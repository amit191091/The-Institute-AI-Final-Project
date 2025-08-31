from typing import Dict, List, Optional, Tuple, Any
import os
from app.logger import get_logger
from app.logger import trace_func

from app.metadata import classify_section_type, extract_keywords
from app.utils import approx_token_len, simple_summarize, truncate_to_tokens, naive_markdown_table, split_into_sentences, split_into_paragraphs, slugify, sha1_short

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

	# Import settings from config for dynamic chunking limits
	from app.config import settings
	
	# Optional overrides for text splitting behavior to control chunk counts
	try:
		# Defaults: semantic+multi ON; use config settings with env overrides
		split_multi = os.getenv("RAG_TEXT_SPLIT_MULTI", "1").lower() in ("1", "true", "yes")
		# Text chunks: target middle of range, max at upper bound
		TEXT_TARGET_TOK = int(os.getenv("RAG_TEXT_TARGET_TOKENS", str((settings.CHUNK_TOK_AVG_RANGE[0] + settings.CHUNK_TOK_AVG_RANGE[1]) // 2)))
		TEXT_MAX_TOK = int(os.getenv("RAG_TEXT_MAX_TOKENS", str(settings.CHUNK_TOK_AVG_RANGE[1])))
		# Figure/Table chunks: higher limits for rich content
		FIGURE_TABLE_MAX_TOK = int(os.getenv("RAG_FIGURE_TABLE_MAX_TOKENS", str(settings.CHUNK_TOK_MAX)))
		SEMANTIC = os.getenv("RAG_SEMANTIC_CHUNKING", "1").lower() in ("1", "true", "yes")
		OVERLAP_N = max(0, int(os.getenv("RAG_TEXT_OVERLAP_SENTENCES", "2") or 2))  # Increased overlap for better context
		DISTILL_RATIO = float(os.getenv("RAG_DISTILL_RATIO", "0.05"))  # 5% distillation target
	except Exception:
		split_multi = True
		TEXT_TARGET_TOK, TEXT_MAX_TOK = 375, 500  # Middle and upper of 250-500 range
		FIGURE_TABLE_MAX_TOK = 800
		SEMANTIC, OVERLAP_N = True, 2
		DISTILL_RATIO = 0.05

	# Trace the effective flags for visibility
	if trace:
		try:
			log.info(
				"FLAGS[chunking]: MULTI=%s SEMANTIC=%s TEXT_TARGET=%d TEXT_MAX=%d FIG_TBL_MAX=%d OVERLAP=%d DISTILL=%.2f",
				split_multi,
				SEMANTIC,
				TEXT_TARGET_TOK,
				TEXT_MAX_TOK,
				FIGURE_TABLE_MAX_TOK,
				OVERLAP_N,
				DISTILL_RATIO,
			)
		except Exception:
			pass
	
	# Helper function to get max tokens based on content type
	def _get_max_tokens_for_type(section_type: str) -> int:
		"""Return appropriate max tokens based on content type."""
		if section_type and section_type.lower() in ("table", "figure", "image"):
			return FIGURE_TABLE_MAX_TOK
		return TEXT_MAX_TOK
	
	def _get_target_tokens_for_type(section_type: str) -> int:
		"""Return appropriate target tokens based on content type."""
		if section_type and section_type.lower() in ("table", "figure", "image"):
			return int(FIGURE_TABLE_MAX_TOK * 0.6)  # ~480 for 800 max
		return TEXT_TARGET_TOK
	# Derive doc_id once per file
	try:
		doc_id = slugify(str(os.path.basename(file_path)))
	except Exception:
		doc_id = slugify(str(file_path))

	# Track per-page ordinals for deterministic anchors
	page_ord: Dict[int, int] = {}

	# ---------------------------
	# Lightweight heading detection + hierarchy stack
	# We don't have font sizes here, so use robust text heuristics and known titles.
	# Maintain a stack of current sections by level (1..3) and generate stable IDs.
	# ---------------------------

	def _is_heading_by_style(md: Any) -> bool:
		"""Best-effort: detect heading using metadata style hints (font size/bold)."""
		try:
			if md is None:
				return False
			font_size = None
			bold = False
			if isinstance(md, dict):
				font_size = md.get("font_size")
				bold = bool(md.get("bold"))
			else:
				font_size = getattr(md, "font_size", None)
				bold = bool(getattr(md, "bold", False))
			min_font = float(os.getenv("RAG_HEADING_MIN_FONT", "12") or 12)
			if isinstance(font_size, (int, float)) and float(font_size) >= min_font:
				return True
			if bold:
				return True
		except (ValueError, TypeError):
			return False
		return False

	def _looks_like_heading(line: str) -> bool:
		"""Heuristic: short-ish line, mostly Title Case, not ending with period, not a table/caption marker."""
		ln = (line or "").strip()
		if not ln:
			return False
		# Skip page headers like "1 | P a g e"
		if "| p a g e" in ln.lower() or "| page" in ln.lower():
			return False
		# Skip obvious captions
		low = ln.lower()
		if low.startswith("figure ") or low.startswith("fig.") or low.startswith("table "):
			return False
		# Length constraint
		if len(ln) < 3 or len(ln) > 120:
			return False
		# Avoid lines that look like sentences
		if ln.endswith(('.', '!', '?', ';')):
			return False
		# Title case proportion
		words = [w for w in ln.split() if w.isalpha()]
		if not words:
			return False
		cap = sum(1 for w in words if w[0].isupper())
		if cap / max(1, len(words)) < 0.5:
			return False
		return True

	def _heading_level(line: str) -> int:
		"""Infer heading level based on numbering prefix or known keywords."""
		import re as _re
		ln = (line or "").strip()
		# Numeric like "1.", "2.1", "3.4.2": depth = dot count + 1 (max 3)
		m = _re.match(r"^\d+(?:\.\d+)*\b", ln)
		if m:
			parts = m.group(0).split('.')
			return min(3, len(parts))
		# Advanced heading keywords
		if _re.match(r"^(chapter|section)\s+\d+(?:\.\d+)*\b", ln, _re.I):
			return 1 if ln.lower().startswith("chapter") else 2
		if _re.match(r"^appendix\s+[a-z]", ln, _re.I):
			return 1
		# Known primary sections
		primary = {"introduction","executive summary","summary","system description","conclusion","recommendations"}
		if (ln or "").strip().lower() in primary:
			return 1
		# Fallback medium level
		return 2

	def _detect_heading_in_text(raw: str, md: Any = None) -> Optional[Tuple[str, int]]:
		# Prefer style-based if metadata indicates heading
		try:
			if _is_heading_by_style(md):
				first_line = (raw or "").strip().splitlines()[0] if raw else None
				if first_line:
					return (first_line.strip(), _heading_level(first_line))
		except Exception:
			pass
		# Check first ~10 non-empty lines for a heading candidate
		lines = [l.strip() for l in (raw or "").splitlines()]
		seen = 0
		for l in lines:
			if not l.strip():
				continue
			seen += 1
			if _looks_like_heading(l):
				return (l.strip(), _heading_level(l))
			if seen >= 10:
				break
		return None

	def _make_section_id(doc: str, title: str, counter: int) -> str:
		return f"{doc}#sec-{counter:03d}-{slugify(title)[:40]}"

	# Section hierarchy state
	section_stack: List[Dict[str, Any]] = []  # each: {id,title,level,breadcrumbs}
	section_counter = 0

	def _update_section_context(new_title: str, new_level: int) -> Dict[str, Any]:
		nonlocal section_counter, section_stack
		# Pop deeper or equal levels
		while section_stack and section_stack[-1]["level"] >= new_level:
			section_stack.pop()
		section_counter += 1
		sec_id = _make_section_id(doc_id, new_title, section_counter)
		parent_id = section_stack[-1]["id"] if section_stack else None
		breadcrumbs = (section_stack[-1]["breadcrumbs"] + [new_title]) if section_stack else [new_title]
		ctx = {"id": sec_id, "title": new_title, "level": new_level, "parent_id": parent_id, "breadcrumbs": breadcrumbs}
		section_stack.append(ctx)
		return ctx

	def _current_section_context() -> Optional[Dict[str, Any]]:
		return section_stack[-1] if section_stack else None

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

	# Track indices consumed by cross-element coalescing (to avoid double-processing)
	consumed: set[int] = set()

	for idx, el in enumerate(elements, start=1):
		if idx in consumed:
			continue
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

		# Treat any non-table/figure/image element as textual for coalescing
		is_textual = str(kind).lower() not in ("table", "figure", "image")
		# Coalesce adjacent textual elements to hit TARGET_TOK across elements (when enabled)
		if is_textual and split_multi:
			try:
				block = [raw_text] if raw_text else []
				cur_tok = approx_token_len(raw_text)
				j = idx + 1
				# Soft cap to avoid overly large pre-merge blocks
				cap_tok = int(TEXT_MAX_TOK * 1.5)
				while j <= len(elements) and cur_tok < max(TEXT_TARGET_TOK, 1):
					_nel = elements[j - 1]
					_nkind = getattr(_nel, "category", getattr(_nel, "type", "Text")) or "Text"
					_nmd = getattr(_nel, "metadata", None)
					_npage = _md_get(_nmd, "page_number") if _nmd is not None else None
					if (str(_nkind).lower() in ("table", "figure", "image")) or _npage != page:
						break
					_ntext = (getattr(_nel, "text", "") or "").strip()
					# Avoid merging across detected headings
					_hn = _detect_heading_in_text(_ntext, _nmd)
					if _hn:
						break
					if _ntext:
						block.append(_ntext)
						cur_tok = approx_token_len("\n\n".join(block))
						consumed.add(j)
						if cur_tok >= cap_tok:
							break
					j += 1
				# If still too small, greedily pack following text elements ignoring headings until a minimum
				min_tok = int(os.getenv("RAG_MIN_CHUNK_TOKENS", str(max(250, TEXT_TARGET_TOK // 2))) or max(250, TEXT_TARGET_TOK // 2))
				if cur_tok < min_tok:
					j2 = j
					while j2 <= len(elements) and cur_tok < min(min_tok, TEXT_MAX_TOK):
						_nel = elements[j2 - 1]
						_nkind = getattr(_nel, "category", getattr(_nel, "type", "Text")) or "Text"
						_nmd = getattr(_nel, "metadata", None)
						_npage = _md_get(_nmd, "page_number") if _nmd is not None else None
						if (str(_nkind).lower() in ("table", "figure", "image")) or _npage != page:
							break
						_ntext = (getattr(_nel, "text", "") or "").strip()
						if _ntext:
							block.append(_ntext)
							cur_tok = approx_token_len("\n\n".join(block))
							consumed.add(j2)
							if cur_tok >= TEXT_MAX_TOK:
								break
						j2 += 1
				if block:
					raw_text = "\n\n".join(block)
			except Exception:
				pass

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
			# Structured analysis on markdown to extract basic numeric stats per column
			def _analyze_table_markdown(md_text: str) -> str:
				try:
					lines = [ln for ln in (md_text or "").splitlines() if ln.strip().startswith("|")]
					if len(lines) < 3:
						return ""
					hdr = [c.strip() for c in lines[0].strip("|").split("|")]
					data_rows = []
					for ln in lines[2:]:
						cells = [c.strip() for c in ln.strip("|").split("|")]
						if len(cells) != len(hdr):
							continue
						data_rows.append(cells)
					stats = []
					for ci, name in enumerate(hdr):
						vals: List[float] = []
						for r in data_rows:
							try:
								v = r[ci].replace(",", "")
								if v.endswith("%"):
									v = v[:-1]
								vals.append(float(v))
							except (ValueError, IndexError):
								continue
						if vals:
							stats.append((name or f"col{ci}", min(vals), max(vals)))
					if not stats:
						return ""
					lines_out = ["ANALYSIS:"] + [f"- {n}: min={vmin:g}, max={vmax:g}" for (n, vmin, vmax) in stats[:4]]
					return "\n".join(lines_out)
				except Exception:
					return ""
			analysis = _analyze_table_markdown(md or as_text)
			# If a full label exists, include at the top for clarity
			label_hdr = f"LABEL: {table_label}\n" if (table_label and str(table_label).strip()) else ""
			content = f"[TABLE]\n{label_hdr}SUMMARY:\n{distilled}\n{(analysis + '\n') if analysis else ''}MARKDOWN:\n{md or as_text}\nRAW:\n{as_text}"
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
			# Attach hierarchy context to table chunk
			_sec = _current_section_context()
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
					# hierarchy
					"section_id": _sec.get("id") if _sec else None,
					"section_title": _sec.get("title") if _sec else None,
					"section_level": _sec.get("level") if _sec else None,
					"section_parent_id": _sec.get("parent_id") if _sec else None,
					"section_breadcrumbs": _sec.get("breadcrumbs") if _sec else None,
					"content": content.strip(),
					"preview": chunk_preview,
					"keywords": extract_keywords(content),
				}
			)
			continue

		elif str(kind).lower() in ("figure", "image"):
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
			_sec = _current_section_context()
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
					# hierarchy
					"section_id": _sec.get("id") if _sec else None,
					"section_title": _sec.get("title") if _sec else None,
					"section_level": _sec.get("level") if _sec else None,
					"section_parent_id": _sec.get("parent_id") if _sec else None,
					"section_breadcrumbs": _sec.get("breadcrumbs") if _sec else None,
					"content": content.strip(),
					"preview": chunk_preview,  # Keep preview short for overview displays
					"keywords": extract_keywords(content),
				}
			)
			continue  # Textual sections handled; go next element

			# (no-op: figure handled)

		# Default: narrative text
		# Update section context if this text block begins with a heading
		try:
			h = _detect_heading_in_text(raw_text, md)
			if h:
				h_title, h_level = h
				_update_section_context(h_title, h_level)
		except Exception:
			pass

		# Paragraph-aware then sentence-aware chunking to hit token targets
		paras = split_into_paragraphs(raw_text)
		sentences = split_into_sentences("\n\n".join(paras))
		if not sentences:
			# Single distilled chunk
			if anchor is None:
				try:
					if page is not None:
						page_ord[page] = page_ord.get(page, 0) + 1
						anchor_local = f"p{int(page)}-t{page_ord[page]}"
					else:
						anchor_local = f"t{idx}"
				except Exception:
					anchor_local = f"t{idx}"
			else:
				anchor_local = anchor
			content_single = simple_summarize(raw_text, ratio=DISTILL_RATIO)
			def _create_chunk_dict(content: str, anchor_local: str) -> Dict[str, Any]:
				# Use dynamic max tokens based on content type  
				max_tokens = _get_max_tokens_for_type(section_type)
				content_c = truncate_to_tokens(content, max_tokens).strip()
				chunk_preview = (content_c or "").splitlines()[0][:200]
				content_hash = sha1_short(content_c)
				chunk_id = f"{doc_id}#p{page}:{section_type or 'Text'}/{anchor_local}"
				try:
					order_val = page_ord.get(int(page)) if page is not None else None
				except Exception:
					order_val = None
				_sec = _current_section_context()
				return {
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Text",
					"section": section_type or "Text",
					"anchor": anchor_local,
					"order": order_val,
					"doc_id": doc_id,
					"chunk_id": chunk_id,
					"content_hash": content_hash,
					"section_id": _sec.get("id") if _sec else None,
					"section_title": _sec.get("title") if _sec else None,
					"section_level": _sec.get("level") if _sec else None,
					"section_parent_id": _sec.get("parent_id") if _sec else None,
					"section_breadcrumbs": _sec.get("breadcrumbs") if _sec else None,
					"content": content_c,
					"preview": chunk_preview,
					"keywords": extract_keywords(content_c),
				}
			chunks.append(_create_chunk_dict(content_single, anchor_local))
			if trace:
				try:
					log.debug("CHUNK-OUT[%d]: section=%s words=%d", idx, section_type or "Text", len((content_single or "").split()))
				except Exception:
					pass
		else:
			# Helpers available in this scope
			def _create_chunk_dict(content: str, anchor_local: str) -> Dict[str, Any]:
				# Use dynamic max tokens based on content type
				max_tokens = _get_max_tokens_for_type(section_type) 
				content_c = truncate_to_tokens(content, max_tokens).strip()
				chunk_preview = (content_c or "").splitlines()[0][:200]
				content_hash = sha1_short(content_c)
				chunk_id = f"{doc_id}#p{page}:{section_type or 'Text'}/{anchor_local}"
				try:
					order_val = page_ord.get(int(page)) if page is not None else None
				except Exception:
					order_val = None
				_sec = _current_section_context()
				return {
					"file_name": file_path,
					"page": page,
					"section_type": section_type or "Text",
					"section": section_type or "Text",
					"anchor": anchor_local,
					"order": order_val,
					"doc_id": doc_id,
					"chunk_id": chunk_id,
					"content_hash": content_hash,
					"section_id": _sec.get("id") if _sec else None,
					"section_title": _sec.get("title") if _sec else None,
					"section_level": _sec.get("level") if _sec else None,
					"section_parent_id": _sec.get("parent_id") if _sec else None,
					"section_breadcrumbs": _sec.get("breadcrumbs") if _sec else None,
					"content": content_c,
					"preview": chunk_preview,
					"keywords": extract_keywords(content_c),
				}

			def _semantic_groups(ss: List[str]) -> List[List[str]]:
				if not SEMANTIC:
					return []
				try:
					from sentence_transformers import SentenceTransformer
					model_name = os.getenv("RAG_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
					st = SentenceTransformer(model_name)
					emb = st.encode(ss, normalize_embeddings=True, show_progress_bar=False)
					th = float(os.getenv("RAG_SEMANTIC_SIM_THRESHOLD", "0.35"))
					groups: List[List[str]] = []
					cur: List[str] = []
					cur_tok = 0
					for i in range(len(ss)):
						if not cur:
							cur = [ss[i]]
							cur_tok = approx_token_len(ss[i])
							continue
						sim = float(emb[i] @ emb[i-1])
						s_tok = approx_token_len(ss[i])
						if sim >= th and (cur_tok + s_tok) <= TEXT_MAX_TOK:
							cur.append(ss[i])
							cur_tok += s_tok
						else:
							groups.append(cur)
							cur = [ss[i]]
							cur_tok = s_tok
						if cur_tok >= TEXT_MAX_TOK:
							groups.append(cur)
							cur = []
							cur_tok = 0
					if cur:
						groups.append(cur)
					return groups
				except Exception:
					return []

			# Compute semantic groups on sentences
			groups = _semantic_groups(sentences)
			if groups:
				# Merge adjacent groups to meet MIN/TARGET token thresholds without exceeding MAX
				min_tok = int(os.getenv("RAG_MIN_CHUNK_TOKENS", str(max(250, TEXT_TARGET_TOK // 2))) or max(250, TEXT_TARGET_TOK // 2))
				merged_groups: List[List[str]] = []
				curg: List[str] = []
				cur_tok = 0
				for g in groups:
					g_text = " ".join(g)
					g_tok = approx_token_len(g_text)
					if not curg:
						curg = list(g)
						cur_tok = g_tok
						continue
					if (cur_tok < max(min_tok, TEXT_TARGET_TOK)) and (cur_tok + g_tok) <= TEXT_MAX_TOK:
						curg.extend(g)
						cur_tok += g_tok
					else:
						merged_groups.append(curg)
						curg = list(g)
						cur_tok = g_tok
				if curg:
					merged_groups.append(curg)

				for g in merged_groups:
					if not g:
						continue
					try:
						if page is not None:
							page_ord[page] = page_ord.get(page, 0) + 1
							anchor_local = f"p{int(page)}-t{page_ord[page]}"
						else:
							page_ord[-1] = page_ord.get(-1, 0) + 1
							anchor_local = f"t{idx}-{page_ord[-1]}"
					except Exception:
						anchor_local = f"t{idx}"
					chunks.append(_create_chunk_dict(" ".join(g), anchor_local))
					if trace:
						try:
							log.debug("CHUNK-OUT[%d.%s]: semantic merged sentences=%d", idx, anchor_local, len(g))
						except Exception:
							pass
					if not split_multi:
						break
			else:
				# Greedy packing with overlap
				si = 0
				while si < len(sentences):
					buf: List[str] = []
					cur_tokens = 0
					start_si = si
					while si < len(sentences):
						s = sentences[si]
						s_tok = approx_token_len(s)
						if cur_tokens + s_tok > TEXT_MAX_TOK and buf:
							break
						buf.append(s)
						cur_tokens += s_tok
						si += 1
						if cur_tokens >= TEXT_TARGET_TOK:
							break
					content_local = " ".join(buf) if buf else sentences[si]
					try:
						if page is not None:
							page_ord[page] = page_ord.get(page, 0) + 1
							anchor_local = f"p{int(page)}-t{page_ord[page]}"
						else:
							page_ord[-1] = page_ord.get(-1, 0) + 1
							anchor_local = f"t{idx}-{page_ord[-1]}"
					except Exception:
						anchor_local = f"t{idx}"
					chunks.append(_create_chunk_dict(content_local, anchor_local))
					if trace:
						try:
							log.debug("CHUNK-OUT[%d.%s]: greedy sentences=%d", idx, anchor_local, len(buf))
						except Exception:
							pass
					if not split_multi:
						break
					# If we already consumed to the end, don't generate trailing tiny windows
					if si >= len(sentences):
						break
					# apply sentence overlap
					if OVERLAP_N > 0:
						si = max(si - OVERLAP_N, start_si + 1)

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

	# Post-process: merge adjacent small textual chunks to reach a minimum token size (target ~250)
	try:
		min_tok = int(os.getenv("RAG_MIN_CHUNK_TOKENS", "250") or 250)
		merged: List[Dict[str, Any]] = []
		def _is_textual_chunk(ch: Dict[str, Any]) -> bool:
			sec = (ch.get("section") or ch.get("section_type") or "Text")
			return sec not in ("Table", "Figure")
		for ch in chunks:
			if not merged:
				merged.append(ch)
				continue
			prev = merged[-1]
			if _is_textual_chunk(prev) and _is_textual_chunk(ch) and \
				(prev.get("file_name") == ch.get("file_name")) and \
				(prev.get("page") == ch.get("page")):
				# If both small, merge
				pt = approx_token_len(prev.get("content") or "")
				ct = approx_token_len(ch.get("content") or "")
				if pt < min_tok or ct < min_tok:
					combined = ((prev.get("content") or "").rstrip() + "\n\n" + (ch.get("content") or "")).strip()
					# Respect max token bound for content type
					max_tokens = _get_max_tokens_for_type(prev.get("section", "Text"))
					if approx_token_len(combined) <= max_tokens:
						prev["content"] = combined
						prev["preview"] = (combined.splitlines()[0] if combined else "")[:200]
						prev["keywords"] = extract_keywords(combined)
						# Mark as merged by keeping first anchor; skip appending ch
						continue
			merged.append(ch)
		chunks = merged
	except Exception:
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

	# Final summary log to help diagnose small chunk counts
	try:
		sec_counts = {"Text": 0, "Table": 0, "Figure": 0}
		from statistics import fmean
		toks = []
		for ch in chunks:
			sec = (ch.get("section") or ch.get("section_type") or "Text")
			if sec not in sec_counts:
				sec_counts[sec] = 0
			sec_counts[sec] += 1
			toks.append(approx_token_len(ch.get("content") or ""))
		avg_tok = float(fmean(toks)) if toks else 0.0
		log.info(
			"Chunking summary: total=%d text=%d table=%d figure=%d other=%d avg_tokens≈%.1f",
			len(chunks),
			sec_counts.get("Text", 0),
			sec_counts.get("Table", 0),
			sec_counts.get("Figure", 0),
			max(0, len(chunks) - (sec_counts.get("Text",0)+sec_counts.get("Table",0)+sec_counts.get("Figure",0))),
			avg_tok,
		)
	except Exception:
		pass

	return chunks

