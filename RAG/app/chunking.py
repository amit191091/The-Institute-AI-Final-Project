from typing import Dict, List, Optional, Tuple
import os
from RAG.app.logger import get_logger
from RAG.app.Data_Management.metadata import classify_section_type
from RAG.app.chunking_modules.chunking_table import process_table_chunk
from RAG.app.chunking_modules.chunking_figure import process_figure_chunk
from RAG.app.chunking_modules.chunking_text import process_text_chunk
from RAG.app.chunking_modules.chunking_utils import (
    build_caption_map, derive_anchor, derive_doc_id, 
    associate_figures_with_captions, associate_tables_with_captions, add_keywords_to_chunks
)


def structure_chunks(elements, file_path: str) -> List[Dict]:
	"""
	Split by natural order: Title/Text, Table, Figure/Image, Appendix.
	Distill each chunk to ~5% core info from its element.
	Save anchors: PageNumber, SectionType, TableId/FigureId (if any), and row/column position for tables.
	Enforce token budgets from config:
	- Text: target ~CHUNK_TOK_AVG_RANGE[0] tokens, max CHUNK_TOK_AVG_RANGE[1]
	- Tables: max CHUNK_TOK_MAX tokens
	- Figures: max CHUNK_TOK_MAX tokens
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
	doc_id = derive_doc_id(file_path)

	# Track per-page ordinals for deterministic anchors
	page_ord: Dict[int, int] = {}

	# Pre-scan: collect caption lines per page like "Figure N: ..." or "Fig. N: ..." to align with image elements
	caption_map = build_caption_map(elements)

	# Track which caption indices have been used per page to avoid double-mapping
	caption_used: Dict[int, set[int]] = {}

	# Doc-level sequential order for figures to ensure stable ascending order regardless of caption text
	figure_seq_counter: int = 0
	# Doc-level last assigned figure number to prevent resets (e.g., if a new page restarts numbering)
	last_figure_number_assigned: int = 0

	for idx, el in enumerate(elements, start=1):
		kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
		md = getattr(el, "metadata", None)
		page = getattr(md, "page_number", None) if md is not None else None
		
		# Extract metadata fields
		table_number = getattr(md, "table_number", None) if md is not None else None
		table_md_path = getattr(md, "table_md_path", None) if md is not None else None
		table_csv_path = getattr(md, "table_csv_path", None) if md is not None else None
		
		# Derive a robust, non-null anchor
		anchor = derive_anchor(el, md, page, table_number, table_md_path, table_csv_path)
		
		raw_text = (getattr(el, "text", "") or "").strip()
		section_type = classify_section_type(str(kind), raw_text)
		
		if trace:
			try:
				log.debug("CHUNK-IN[%d]: kind=%s page=%s section=%s len=%d", idx, kind, page, section_type, len(raw_text))
			except Exception:
				pass

		if str(kind).lower() == "table":
			chunk = process_table_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, trace)
			chunks.append(chunk)
			continue

		if str(kind).lower() in ("figure", "image"):
			chunk, figure_seq_counter, last_figure_number_assigned = process_figure_chunk(
				el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path,
				caption_map, caption_used, figure_seq_counter, last_figure_number_assigned, trace
			)
			chunks.append(chunk)
			continue

		# Text chunking
		chunk = process_text_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, trace)
		chunks.append(chunk)
		
		if trace:
			try:
				log.debug("CHUNK-OUT[%d]: section=%s words=%d", idx, section_type or "Text", len((chunk.get("content") or "").split()))
			except Exception:
				pass

	# Post-process: associate figures and tables with their captions
	associate_figures_with_captions(chunks)
	associate_tables_with_captions(chunks)
	
	# Add keywords to all chunks
	add_keywords_to_chunks(chunks)

	return chunks

