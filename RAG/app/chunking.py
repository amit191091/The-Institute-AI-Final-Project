from typing import Dict, List, Optional, Tuple
import os
from app.logger import get_logger

from app.metadata import classify_section_type, extract_keywords
from app.utils import approx_token_len, simple_summarize, truncate_to_tokens, naive_markdown_table


def structure_chunks(elements, file_path: str) -> List[Dict]:
	"""
	Split by natural order: Title/Text, Table, Figure/Image, Appendix.
	Distill each chunk to ~5% core info from its element.
	Save anchors: PageNumber, SectionType, TableId/FigureId (if any), and row/column position for tables.
	Enforce token budgets: avg 250-500 tokens; max 800 for table/figure chunks.
	"""
	chunks: List[Dict] = []
	log = get_logger()
	trace = os.getenv("RAG_TRACE_CHUNKING", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes")
	if trace:
		try:
			log.debug("Chunking: %d input elements from %s", len(elements or []), file_path)
		except Exception:
			pass
	for idx, el in enumerate(elements, start=1):
		kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
		md = getattr(el, "metadata", None)
		page = getattr(md, "page_number", None) if md is not None else None
		anchor = getattr(md, "id", None) if md is not None else None
		table_md_path = getattr(md, "table_md_path", None) if md is not None else None
		table_csv_path = getattr(md, "table_csv_path", None) if md is not None else None
		raw_text = (getattr(el, "text", "") or "").strip()

		section_type = classify_section_type(str(kind), raw_text)
		if trace:
			try:
				log.debug("CHUNK-IN[%d]: kind=%s page=%s section=%s len=%d", idx, kind, page, section_type, len(raw_text))
			except Exception:
				pass

		if str(kind).lower() == "table":
			as_text = raw_text
			# Use the generated summary if available
			table_summary = getattr(md, "table_summary", None) if md is not None else None
			if table_summary:
				distilled = table_summary
			else:
				distilled = simple_summarize(as_text, ratio=0.05)
			
			row_range: Optional[Tuple[int, int]] = (0, 0)
			col_names: Optional[list[str]] = []
			md = naive_markdown_table(as_text)
			content = f"[TABLE]\nSUMMARY:\n{distilled}\nMARKDOWN:\n{md or as_text}\nRAW:\n{as_text}"
			tok = approx_token_len(content)
			if tok > 800:
				content = truncate_to_tokens(content, 800)
			if trace:
				log.debug("CHUNK-OUT[%d]: section=Table tokensג‰ˆ%d anchor=%s", idx, tok, anchor)
			chunks.append(
				{
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Table",
					"anchor": anchor or None,
					"table_row_range": row_range,
					"table_col_names": col_names,
					"table_md_path": table_md_path,
					"table_csv_path": table_csv_path,
					"content": content.strip(),
					"keywords": extract_keywords(content),
				}
			)
			continue

		if str(kind).lower() in ("figure", "image"):
			caption = raw_text or "Figure"
			# Use the generated summary if available
			figure_summary = getattr(md, "figure_summary", None) if md is not None else None
			if figure_summary:
				content = f"[FIGURE]\n{figure_summary}"
			else:
				distilled = simple_summarize(caption, ratio=0.5)
				content = f"[FIGURE]\n{distilled}"
			
			# Try to detect image path from text or metadata
			img_path = getattr(md, "image_path", None) if md is not None else None
			if not img_path:
				try:
					import re as _re
					m = _re.search(r"Image file: (.+)$", caption)
					if m:
						img_path = m.group(1).strip()
				except Exception:
					pass
			
			tok = approx_token_len(content)
			if tok > 800:
				content = truncate_to_tokens(content, 800)
			if trace:
				log.debug("CHUNK-OUT[%d]: section=Figure tokensג‰ˆ%d img=%s", idx, tok, img_path)
			chunks.append(
				{
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Figure",
					"anchor": anchor or None,
					"image_path": img_path,
					"content": content.strip(),
					"keywords": extract_keywords(content),
				}
			)
			continue		# Textual sections (headers/paragraphs/timeline/analysis/conclusion)
		content = simple_summarize(raw_text, ratio=0.05)
		chunks.append(
			{
				"file_name": file_path,
				"page": page,
				"section_type": section_type or "Text",
				"anchor": anchor or None,
				"content": truncate_to_tokens(content, 500).strip(),
				"keywords": extract_keywords(content),
			}
		)
		if trace:
			try:
				log.debug("CHUNK-OUT[%d]: section=%s words=%d", idx, section_type or "Text", len((content or "").split()))
			except Exception:
				pass

	return chunks

