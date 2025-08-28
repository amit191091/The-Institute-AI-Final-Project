from typing import Dict, List, Optional, Tuple, Any
import os
from app.logger import get_logger
from app.logger import trace_func

from app.metadata import classify_section_type, extract_keywords
from app.utils import approx_token_len, simple_summarize, truncate_to_tokens, naive_markdown_table, split_into_sentences, slugify, sha1_short

@trace_func
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

	# Helper to read metadata values from either dict-like or attr-like containers
	def _md_get(md: Any, key: str, default: Any = None) -> Any:
		if md is None:
			return default
		try:
			# dict-like first
			if isinstance(md, dict):
				return md.get(key, default)
			# attr-like fallback
			return getattr(md, key, default)
		except Exception:
			return default

	# Pre-scan: collect caption lines per page like "Figure N: ..." or "Fig. N: ..." to align with image elements
	# We rely on PDF reading order when coordinates are unavailable.
	from typing import DefaultDict
	caption_map: DefaultDict[int, List[Tuple[int, int, str]]] = DefaultDict(list)  # page -> list of (idx, num, text)
	for _idx, _el in enumerate(elements, start=1):
		_kind = getattr(_el, "category", getattr(_el, "type", "Text")) or "Text"
		_md = getattr(_el, "metadata", None)
		_page = _md_get(_md, "page_number") if _md is not None else None
		if _page is None:
			continue
		if str(_kind).lower() == "text":
			_cap = (getattr(_el, "text", "") or "").strip()
			if _cap:
				import re as _re
				# Support Figure / Fig. / multi-line text after colon; capture full line for later normalization
				m = _re.search(r"\b(fig(?:\.|ure)?)\s*(\d{1,3})\b\s*:\s*(.+)$", _cap, _re.I)
				if m:
					try:
						num = int(m.group(2))
						caption_map[_page].append((_idx, num, _cap))
					except Exception:
						pass

	# Track which caption indices have been used per page to avoid double-mapping
	caption_used: Dict[int, set[int]] = {}

	# Doc-level sequential order for figures to ensure stable ascending order regardless of caption text
	figure_seq_counter: int = 0
	# Doc-level last assigned figure number to prevent resets (e.g., if a new page restarts numbering)
	last_figure_number_assigned: int = 0

	for idx, el in enumerate(elements, start=1):
		kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
		md = getattr(el, "metadata", None)
		page = _md_get(md, "page_number") if md is not None else None
		# Derive a robust, non-null anchor
		# Priority: explicit table/figure anchors -> element id -> file-based stems -> fallback
		table_anchor = _md_get(md, "table_anchor") if md is not None else None
		figure_anchor = _md_get(md, "figure_anchor") if md is not None else None
		anchor = table_anchor or figure_anchor
		if anchor is None:
			anchor = _md_get(md, "id") if md is not None else None
		extractor = _md_get(md, "extractor") if md is not None else None
		table_md_path = _md_get(md, "table_md_path") if md is not None else None
		table_csv_path = _md_get(md, "table_csv_path") if md is not None else None
		table_number = _md_get(md, "table_number") if md is not None else None
		table_label = _md_get(md, "table_label") if md is not None else None
		table_caption = _md_get(md, "table_caption") if md is not None else None
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
			table_summary = _md_get(md, "table_summary") if md is not None else None
			if table_summary:
				distilled = table_summary
			else:
				distilled = simple_summarize(as_text, ratio=0.05)
			
			row_range: Optional[Tuple[int, int]] = (0, 0)
			col_names: Optional[list[str]] = []
			md = naive_markdown_table(as_text)
			# If a full label exists, include at the top for clarity
			label_hdr = f"LABEL: {table_label}\n" if (table_label and str(table_label).strip()) else ""
			content = f"[TABLE]\n{label_hdr}SUMMARY:\n{distilled}\nMARKDOWN:\n{md or as_text}\nRAW:\n{as_text}"
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
						# keep a consistent table anchor scheme when number is unknown
						anchor = f"p{int(page)}-tbl{ordinal}"
					elif table_number is not None:
						anchor = f"table-{int(table_number):02d}"
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
					"section": "Table",
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
					"table_caption": table_caption,
					"content": content.strip(),
					"preview": chunk_preview,
					"keywords": extract_keywords(content),
				}
			)
			continue

		if str(kind).lower() in ("figure", "image"):
			# Increment document-level figure order
			try:
				figure_seq_counter += 1
			except Exception:
				figure_seq_counter = figure_seq_counter + 1
			caption = raw_text or "Figure"
			# Extract figure number if present in caption (e.g., "Figure 2:")
			fig_num = None
			try:
				import re as _re
				mfn = _re.search(r"\b(fig(?:\.|ure)?)\s*(\d{1,3})\b", caption, _re.I)
				if mfn:
					fig_num = int(mfn.group(2))
			except Exception:
				fig_num = None
			# Align figure to the nearest caption BELOW it on the same page (by element index), if available
			aligned_via = None
			try:
				if page is not None and caption_map.get(page):
					used = caption_used.setdefault(page, set())
					# choose the first caption with element index greater than current figure element index
					cands = [(cidx, cnum, ctext) for (cidx, cnum, ctext) in caption_map[page] if cidx > idx and cidx not in used]
					if cands:
						cidx, cnum, ctext = sorted(cands, key=lambda t: t[0])[0]
						caption = ctext or caption
						if cnum is not None:
							fig_num = cnum
						used.add(cidx)
						aligned_via = "after_same_page"
					else:
						# fallback: pick the next unused caption regardless of relative order (rare PDFs)
						cands2 = [(cidx, cnum, ctext) for (cidx, cnum, ctext) in caption_map[page] if cidx not in used]
						if cands2:
							cidx, cnum, ctext = sorted(cands2, key=lambda t: t[0])[0]
							caption = ctext or caption
							if cnum is not None:
								fig_num = cnum
							used.add(cidx)
							aligned_via = "same_page_any"
			except Exception:
				aligned_via = None
			# Build a clean summary derived from caption (or metadata), but drop any leading "Figure X:" label
			figure_summary = _md_get(md, "figure_summary") if md is not None else None
			try:
				import re as _re
				if figure_summary:
					_tmp = str(figure_summary).replace("[FIGURE]", "")
					_tmp = _re.sub(r"^\s*figure\s*\d{1,3}\s*:\s*", "", _tmp, flags=_re.I)
					# Remove any inline image-file hints
					_tmp = _re.sub(r"(?mi)^\s*image\s*file\s*:\s*.*$", "", _tmp)
					figure_summary = _tmp.strip() or None
			except Exception:
				pass
			if not figure_summary:
				# Provide a brief distilled summary from the cleaned caption text (after removing any image-file hints)
				try:
					import re as _re
					cap_for_sum = _re.sub(r"(?mi)^\s*image\s*file\s*:\s*.*$", "", caption)
					figure_summary = simple_summarize(cap_for_sum, ratio=0.5)
				except Exception:
					figure_summary = simple_summarize(caption, ratio=0.5)
			# Prepare a clean, normalized caption reflecting the final number; keep original for traceability
			caption_original = caption
			try:
				# Strip parser tags like "[FIGURE]" and any leading "Figure X:" label; also drop any "Image file:" lines
				import re as _re
				cap_clean = caption_original.replace("[FIGURE]", "")
				cap_clean = _re.sub(r"^\s*fig(?:\.|ure)?\s*\d{1,3}\s*:\s*", "", cap_clean, flags=_re.I)
				# Remove any lines that declare an image file path
				cap_clean = _re.sub(r"(?mi)^\s*image\s*file\s*:\s*.*$", "", cap_clean)
				# Collapse multiple blank lines and trim
				cap_clean = "\n".join([ln for ln in (cap_clean.splitlines()) if ln.strip()]).strip()
			except Exception:
				cap_clean = caption_original
			# We'll compute the final figure number below; temporarily store cleaned caption
			caption = cap_clean
			
			# Try to detect image path from text or metadata
			img_path = _md_get(md, "image_path") if md is not None else None
			if not img_path:
				try:
					import re as _re
					m = _re.search(r"Image file: (.+)$", caption)
					if m:
						img_path = m.group(1).strip()
				except Exception:
					pass
			
			# Finalize figure number: prefer caption number; if missing, infer from sequence.
			# Ensure monotonically increasing across the document.
			if fig_num is not None and fig_num > last_figure_number_assigned:
				figure_number_final = fig_num
			else:
				figure_number_final = max(last_figure_number_assigned + 1, figure_seq_counter)
			last_figure_number_assigned = figure_number_final
			# Normalize the visible caption to the final number with better handling of long descriptions
			try:
				import re as _re
				if _re.search(r"^\s*fig(?:\.|ure)?\s*\d{1,3}\s*:\s*", caption, _re.I):
					caption_norm = _re.sub(r"^\s*fig(?:\.|ure)?\s*\d{1,3}\s*:\s*", f"Figure {int(figure_number_final)}: ", caption, flags=_re.I)
				else:
					caption_norm = f"Figure {int(figure_number_final)}: {caption}"
				
				# Use full caption for the label (no truncation for metadata)
				figure_label_full = caption_norm
				
				# Create a shorter summary only for the content section shown to LLM 
				# (to manage token budget), but preserve full label in metadata
				try:
					if len(caption_norm.split()) > 25:  # Only summarize if very long
						cap_for_summary = simple_summarize(caption, ratio=0.6)  # Keep more content
						figure_summary_short = f"Figure {int(figure_number_final)}: {cap_for_summary}"
					else:
						figure_summary_short = caption_norm  # Use full for shorter captions
				except Exception:
					figure_summary_short = caption_norm
			except Exception:
				caption_norm = caption
				figure_label_full = caption_norm
				figure_summary_short = caption_norm

			# Derive clean anchor: "figure-N" as requested
			try:
				custom_anchor = f"figure-{int(figure_number_final)}"
			except Exception:
				# Fallback to simple numbering
				custom_anchor = f"figure-{figure_number_final}"

			# Build content with normalized caption shown to the LLM; compute token budget after composing
			content = f"[FIGURE]\nCAPTION:\n{figure_summary_short}\nSUMMARY:\n{figure_summary}"
			tok = approx_token_len(content)
			if tok > 800:
				content = truncate_to_tokens(content, 800)
			if trace:
				log.debug("CHUNK-OUT[%d]: section=Figure tokens≈%d img=%s anchor=%s", idx, tok, img_path, anchor)
			chunk_preview = (content or "").splitlines()[0][:200]
			content_hash = sha1_short(content)
			# Ensure chunk_id uses the finalized custom anchor for figures
			chunk_id = f"{doc_id}#p{page}:{section_type or 'Figure'}/{custom_anchor}"
			order_val = None
			try:
				order_val = page_ord.get(int(page)) if page is not None else None
			except Exception:
				order_val = None
			# Store finalized figure chunk
			chunks.append(
				{
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Figure",
					"section": "Figure",
					"anchor": custom_anchor,  # Use our custom figure anchor
					"order": order_val,
					"doc_id": doc_id,
					"chunk_id": chunk_id,
					"content_hash": content_hash,
					"extractor": extractor,
					"image_path": img_path,
					"figure_number": figure_number_final,
					"figure_order": figure_seq_counter,
					"figure_label": figure_label_full,  # Use full label without truncation
					"figure_caption_original": caption_original,
					"figure_number_source": "caption" if fig_num is not None else "inferred",
					"caption_alignment": aligned_via or "none",
					"content": content.strip(),
					"preview": chunk_preview,  # Keep preview short for overview displays
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
				"section": section_type or "Text",
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

	# Post-process: associate each figure with its caption Text chunk on the same page (preferred)
	# Keep figure_label as the caption string; expose the Text anchor so UIs can link to it.
	# If no caption chunk is found, fallback to the earliest descriptive mention of the figure.
	try:
		import re as _re
		for i, ch in enumerate(chunks):
			if (ch.get("section") or ch.get("section_type")) != "Figure":
				continue
			pg = ch.get("page")
			fig_num = ch.get("figure_number")
			assoc_text = None
			assoc_anchor = None

			# Pass 1: prefer the caption chunk (e.g., a Text chunk starting with "Figure N:" or "Fig. N:")
			if fig_num is not None:
				cap_pat = rf"^\s*fig(?:\.|ure)?\s*{int(fig_num)}\b\s*[:\.-]"
				cap_candidate = None
				for nx in chunks:
					if (nx.get("page") != pg) or (nx.get("section") != "Text"):
						continue
					text_content = (nx.get("content") or "")
					text_preview = (nx.get("preview") or "")
					lines = []
					try:
						lines = (text_content or "").splitlines()
					except Exception:
						lines = []
					if any(_re.search(cap_pat, ln, _re.I) for ln in (lines or [])) or _re.search(cap_pat, text_preview, _re.I):
						cap_candidate = nx
						break
				if cap_candidate is not None:
					assoc_text = cap_candidate.get("preview") or (cap_candidate.get("content") or "")[:200]
					assoc_anchor = cap_candidate.get("anchor")
					# If the current stored figure_label seems shorter than the caption chunk, upgrade it
					try:
						curr_label = ch.get("figure_label") or ""
						if assoc_text and (len(assoc_text) > len(curr_label)) and assoc_text.lower().startswith("figure "):
							ch["figure_label"] = assoc_text
					except Exception:
						pass

			# Pass 2: if no caption chunk found, fallback to earliest descriptive mention on the same page
			if assoc_text is None and fig_num is not None:
				mention_pat = rf"\bfig(?:\.|ure)?\s*{int(fig_num)}\b"
				caption_pat = rf"^\s*fig(?:\.|ure)?\s*{int(fig_num)}\b\s*[:\.-]"
				candidates = []
				for nx in chunks:
					if (nx.get("page") != pg) or (nx.get("section") != "Text"):
						continue
					text_content = ((nx.get("content") or "") + " " + (nx.get("preview") or "")).strip()
					if not text_content:
						continue
					if _re.search(caption_pat, text_content, _re.I):
						continue
					if _re.search(mention_pat, text_content, _re.I):
						candidates.append(nx)
				if candidates:
					def _order_key(x: dict) -> int:
						# Prefer explicit 'order' (page ordinal), fallback to numeric suffix from anchor 'p{pg}-t{n}', else large
						try:
							if isinstance(x.get("order"), int):
								return int(x.get("order") or 10**9)
							anch = str(x.get("anchor") or "")
							import re as __re
							m = __re.search(r"p(\d+)-t(\d+)", anch)
							if m:
								return int(m.group(2))
						except Exception:
							pass
						return 10**9
					cand = sorted(candidates, key=_order_key)[:1]
					if cand:
						cand = cand[0]
						assoc_text = cand.get("preview") or (cand.get("content") or "")[:200]
						assoc_anchor = cand.get("anchor")
			
			# Set the association (but don't override the figure_label)
			if assoc_text:
				ch["figure_associated_text_preview"] = assoc_text
				ch["figure_associated_anchor"] = assoc_anchor
				# If figure_summary_short equals the label (or is too generic), upgrade preview to associated text
				try:
					prev = ch.get("preview") or ""
					label = ch.get("figure_label") or ""
					if not prev or prev.strip().lower() == (label or "").strip().lower():
						ch["preview"] = assoc_text
				except Exception:
					pass
				# Keep the original figure_label from the image caption as-is
				# Don't override it with the associated text
	except Exception:
		pass
		pass
		pass

	# Post-process: associate each table with its caption text chunk on the same page
	try:
		import re as _re
		for ch in chunks:
			if (ch.get("section") or ch.get("section_type")) != "Table":
				continue
			pg = ch.get("page")
			tn = ch.get("table_number")
			assoc_text = None
			assoc_anchor = None
			if tn is None:
				continue
			# Allow variations like "Table N:", "Table N.", "Table N -"
			pattern = rf"^\s*table\s*{int(tn)}\b\s*[:\.-]"
			for nx in chunks:
				if (nx.get("page") != pg) or (nx.get("section") != "Text"):
					continue
				text_content = (nx.get("content") or "") + " " + (nx.get("preview") or "")
				if _re.search(pattern, text_content, _re.I):
					assoc_text = nx.get("preview") or (nx.get("content") or "")[:200]
					assoc_anchor = nx.get("anchor")
					break
			if assoc_text:
				ch["table_associated_text_preview"] = assoc_text
				ch["table_associated_anchor"] = assoc_anchor
				# table_label already carries full caption text from loaders
	except Exception:
		pass

	return chunks

