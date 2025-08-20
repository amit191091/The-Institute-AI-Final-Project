from typing import Dict, List, Optional, Tuple

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
	for el in elements:
		kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
		md = getattr(el, "metadata", None)
		page = getattr(md, "page_number", None) if md is not None else None
		anchor = getattr(md, "id", None) if md is not None else None
		raw_text = (getattr(el, "text", "") or "").strip()

		section_type = classify_section_type(str(kind), raw_text)

		if str(kind).lower() == "table":
			as_text = raw_text
			distilled = simple_summarize(as_text, ratio=0.05)
			row_range: Optional[Tuple[int, int]] = (0, 0)
			col_names: Optional[list[str]] = []
			md = naive_markdown_table(as_text)
			content = f"[TABLE]\nSUMMARY:\n{distilled}\nMARKDOWN:\n{md or as_text}\nRAW:\n{as_text}"
			tok = approx_token_len(content)
			if tok > 800:
				content = truncate_to_tokens(content, 800)
			chunks.append(
				{
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Table",
					"anchor": anchor or None,
					"table_row_range": row_range,
					"table_col_names": col_names,
					"content": content.strip(),
					"keywords": extract_keywords(content),
				}
			)
			continue

		if str(kind).lower() in ("figure", "image"):
			caption = raw_text or "Figure"
			distilled = simple_summarize(caption, ratio=0.5)
			content = f"[FIGURE]\n{distilled}"
			# Try to detect image path hint embedded by loader
			img_path = None
			m = None
			try:
				import re as _re
				m = _re.search(r"Extracted image saved at (.+)$", caption)
			except Exception:
				m = None
			if m:
				img_path = m.group(1).strip()
			tok = approx_token_len(content)
			if tok > 800:
				content = truncate_to_tokens(content, 800)
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
			continue

		# Textual sections (headers/paragraphs/timeline/analysis/conclusion)
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

	return chunks

