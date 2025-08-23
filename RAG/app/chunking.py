from typing import Dict, List, Optional, Tuple
import os
from app.logger import get_logger

from app.metadata import classify_section_type, extract_keywords
from app.utils import approx_token_len, simple_summarize, truncate_to_tokens, naive_markdown_table, normalize_engineering_tokens


def structure_chunks(elements, file_path: str) -> List[Dict]:
	"""
	Split by natural order: Title/Text, Table, Figure/Image, Appendix.
	Distill each chunk to ~5% core info from its element.
	Save anchors: PageNumber, SectionType, TableId/FigureId (if any), and row/column position for tables.
	Enforce token budgets: avg 250-500 tokens; max 800 for table/figure chunks.
	Improved for better context sensitivity and RAGAS metrics.
	"""
	chunks: List[Dict] = []
	
	# Group elements by page for better context
	page_elements = {}
	for el in elements:
		md = getattr(el, "metadata", None)
		page = getattr(md, "page_number", 1) if md is not None else 1
		if page not in page_elements:
			page_elements[page] = []
		page_elements[page].append(el)
	
	for page_num, page_els in sorted(page_elements.items()):
		# Process elements on this page
		for idx, el in enumerate(page_els, start=1):
			kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
			md = getattr(el, "metadata", None)
			page = getattr(md, "page_number", page_num) if md is not None else page_num
			anchor = getattr(md, "id", None) if md is not None else None
			table_md_path = getattr(md, "table_md_path", None) if md is not None else None
			table_csv_path = getattr(md, "table_csv_path", None) if md is not None else None
			raw_text = (getattr(el, "text", "") or "").strip()

			section_type = classify_section_type(str(kind), raw_text)

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
				md_content = naive_markdown_table(as_text)
				
				# Enhanced table content with better context
				content = f"[TABLE - Page {page}]\nSUMMARY: {distilled}\n\nMARKDOWN:\n{md_content or as_text}\n\nRAW DATA:\n{normalize_engineering_tokens(as_text)}"
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
					content = f"[FIGURE - Page {page}]\n{figure_summary}"
				else:
					distilled = simple_summarize(caption, ratio=0.5)
					content = f"[FIGURE - Page {page}]\n{distilled}"
				
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
		# Enhanced: Create context-sensitive text chunks
		for idx, el in enumerate(page_els, start=1):
			kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
			if str(kind).lower() in ("table", "figure", "image"):
				continue  # Already processed
			
			md = getattr(el, "metadata", None)
			page = getattr(md, "page_number", page_num) if md is not None else page_num
			anchor = getattr(md, "id", None) if md is not None else None
			raw_text = (getattr(el, "text", "") or "").strip()
			
			if not raw_text:
				continue
			
			section_type = classify_section_type(str(kind), raw_text)
			
			# Enhanced text processing with better context preservation
			normalized_text = normalize_engineering_tokens(raw_text)
			
			# Create context-sensitive chunks
			# For longer text, split into smaller chunks while preserving context
			if len(normalized_text) > 1000:
				# Split into smaller chunks with overlap
				chunk_size = 800
				overlap = 200
				text_chunks = []
				
				for i in range(0, len(normalized_text), chunk_size - overlap):
					chunk_text = normalized_text[i:i + chunk_size]
					if len(chunk_text) > 100:  # Only keep substantial chunks
						text_chunks.append(chunk_text)
			else:
				text_chunks = [normalized_text]
			
			for chunk_idx, chunk_text in enumerate(text_chunks):
				# Add page context to each chunk
				content = f"[TEXT - Page {page}]\n{chunk_text}"
				content = truncate_to_tokens(content, 500).strip()
				
				chunks.append(
					{
						"file_name": file_path,
						"page": page,
						"section_type": section_type or "Text",
						"anchor": anchor or None,
						"content": content,
						"keywords": extract_keywords(content),
					}
				)
	
	return chunks