from typing import Dict, List, Optional, Tuple
import os
from app.logger import get_logger

from app.metadata import classify_section_type, extract_keywords
from app.utils import approx_token_len, simple_summarize, truncate_to_tokens, naive_markdown_table, split_into_sentences, slugify, sha1_short


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
	# Derive doc_id once per file
	try:
		doc_id = slugify(str(os.path.basename(file_path)))
	except Exception:
		doc_id = slugify(str(file_path))

	# Track per-page ordinals for deterministic anchors
	page_ord: Dict[int, int] = {}

	for idx, el in enumerate(elements, start=1):
		kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
		md = getattr(el, "metadata", None)
		page = getattr(md, "page_number", None) if md is not None else None
		# Derive a robust, non-null anchor
		# Priority: explicit table/figure anchors -> element id -> file-based stems -> fallback
		table_anchor = getattr(md, "table_anchor", None) if md is not None else None
		figure_anchor = getattr(md, "figure_anchor", None) if md is not None else None
		anchor = table_anchor or figure_anchor
		if anchor is None:
			anchor = getattr(md, "id", None) if md is not None else None
		extractor = getattr(md, "extractor", None) if md is not None else None
		table_md_path = getattr(md, "table_md_path", None) if md is not None else None
		table_csv_path = getattr(md, "table_csv_path", None) if md is not None else None
		table_number = getattr(md, "table_number", None) if md is not None else None
		table_label = getattr(md, "table_label", None) if md is not None else None
		# If still no anchor, derive from known file paths or numbering
		if anchor is None:
			try:
				if table_number is not None and page is not None:
					anchor = f"table-{int(table_number):02d}"
			except Exception:
				pass
		if anchor is None and table_md_path:
			try:
				import os as _os
				anchor = _os.path.splitext(_os.path.basename(str(table_md_path)))[0]
			except Exception:
				pass
		if anchor is None and table_csv_path:
			try:
				import os as _os
				anchor = _os.path.splitext(_os.path.basename(str(table_csv_path)))[0]
			except Exception:
				pass
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
				log.debug("CHUNK-OUT[%d]: section=Table tokens≈%d anchor=%s", idx, tok, anchor)
			# Final fallback for table anchor if still None
			if anchor is None:
				try:
					if page is not None:
						page_ord[page] = page_ord.get(page, 0) + 1
						ordinal = page_ord[page]
						anchor = f"p{int(page)}-tbl{ordinal}"
					elif table_number is not None:
						anchor = f"tbl{int(table_number)}"
				except Exception:
					anchor = f"tbl{idx}"
			# Build deterministic IDs and metadata
			chunk_preview = (content or "").splitlines()[0][:200]
			content_hash = sha1_short(content)
			chunk_id = f"{doc_id}#p{page}:{section_type or 'Table'}/{anchor}"
			order_val = None
			try:
				order_val = page_ord.get(int(page)) if page is not None else None
			except Exception:
				order_val = None
			chunks.append(
				{
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Table",
					"anchor": anchor or None,
					"order": order_val,
					"doc_id": doc_id,
					"chunk_id": chunk_id,
					"content_hash": content_hash,
					"extractor": extractor,
					"table_number": table_number,
					"table_label": table_label,
					"table_row_range": row_range,
					"table_col_names": col_names,
					"table_md_path": table_md_path,
					"table_csv_path": table_csv_path,
					"content": content.strip(),
					"preview": chunk_preview,
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
			# Derive figure anchor from image path or page index
			if anchor is None:
				try:
					if img_path:
						import os as _os
						anchor = _os.path.splitext(_os.path.basename(str(img_path)))[0]
					elif page is not None:
						anchor = f"fig-p{int(page)}-{idx}"
				except Exception:
					anchor = f"figure-{idx}"
			
			tok = approx_token_len(content)
			if tok > 800:
				content = truncate_to_tokens(content, 800)
			if trace:
				log.debug("CHUNK-OUT[%d]: section=Figure tokens≈%d img=%s", idx, tok, img_path)
			chunk_preview = (content or "").splitlines()[0][:200]
			content_hash = sha1_short(content)
			if anchor is None:
				try:
					if page is not None:
						page_ord[page] = page_ord.get(page, 0) + 1
						anchor = f"p{int(page)}-fig{page_ord[page]}"
					else:
						anchor = f"fig{idx}"
				except Exception:
					anchor = f"fig{idx}"
			chunk_id = f"{doc_id}#p{page}:{section_type or 'Figure'}/{anchor}"
			order_val = None
			try:
				order_val = page_ord.get(int(page)) if page is not None else None
			except Exception:
				order_val = None
			chunks.append(
				{
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Figure",
					"anchor": anchor or None,
					"order": order_val,
					"doc_id": doc_id,
					"chunk_id": chunk_id,
					"content_hash": content_hash,
					"extractor": extractor,
					"image_path": img_path,
					"content": content.strip(),
					"preview": chunk_preview,
					"keywords": extract_keywords(content),
				}
			)
			continue  # Textual sections handled; go next element

		# Sentence-aware chunking to target 200-500 tokens per chunk
		sentences = split_into_sentences(raw_text)
		if not sentences:
			content = simple_summarize(raw_text, ratio=0.2)
		else:
			# Greedy pack sentences until ~350 tokens, cap at 500
			buf: List[str] = []
			cur_tokens = 0
			target = 350
			max_tokens = 500
			for s in sentences:
				s_tok = approx_token_len(s)
				if cur_tokens + s_tok > max_tokens and buf:
					break
				buf.append(s)
				cur_tokens += s_tok
				if cur_tokens >= target:
					break
			content = " ".join(buf) if buf else raw_text
		# Fallback anchor for text content if missing
		if anchor is None:
			try:
				if page is not None:
					page_ord[page] = page_ord.get(page, 0) + 1
					anchor = f"p{int(page)}-t{page_ord[page]}"
				else:
					anchor = f"t{idx}"
			except Exception:
				anchor = f"t{idx}"
		# Build IDs and metadata hygiene fields
		content = truncate_to_tokens(content, 500).strip()
		chunk_preview = (content or "").splitlines()[0][:200]
		content_hash = sha1_short(content)
		chunk_id = f"{doc_id}#p{page}:{section_type or 'Text'}/{anchor}"
		order_val = None
		try:
			order_val = page_ord.get(int(page)) if page is not None else None
		except Exception:
			order_val = None
		chunks.append(
			{
				"file_name": file_path,
				"page": page,
				"section_type": section_type or "Text",
				"anchor": anchor or None,
				"order": order_val,
				"doc_id": doc_id,
				"chunk_id": chunk_id,
				"content_hash": content_hash,
				"content": content,
				"preview": chunk_preview,
				"keywords": extract_keywords(content),
			}
		)
		if trace:
			try:
				log.debug("CHUNK-OUT[%d]: section=%s words=%d", idx, section_type or "Text", len((content or "").split()))
			except Exception:
				pass

	return chunks

