from pathlib import Path
from typing import List
import re
import os
from types import SimpleNamespace
from app.logger import get_logger


def _generate_table_summary(text: str) -> str:
	"""Generate a semantic summary for a table based on its content."""
	import re
	lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
	if not lines:
		return "Data table"
	
	# Look for table title or caption
	for line in lines[:3]:
		if re.search(r"\btable\s*\d+", line, re.I):
			# Extract everything after "Table X:"
			match = re.search(r"\btable\s*\d+[:\.\-\s]*(.+)", line, re.I)
			if match:
				return f"Table: {match.group(1).strip()}"
	
	# Look for column headers to infer content
	header_indicators = []
	for line in lines[:3]:
		if any(term in line.lower() for term in ["case", "wear", "depth", "sensor", "value", "feature"]):
			header_indicators.extend(re.findall(r"\b(case|wear|depth|sensor|value|feature|measurement|data)\w*\b", line.lower()))
	
	if "wear" in header_indicators:
		return "Wear measurement data table"
	elif "sensor" in header_indicators:
		return "Sensor configuration table"
	elif "case" in header_indicators:
		return "Case study data table"
	elif header_indicators:
		return f"Data table ({', '.join(set(header_indicators[:3]))})"
	
	return "Data table"


def _generate_figure_summary(page_text: str, page_num: int, img_index: int) -> str:
	"""Generate a semantic summary for a figure based on surrounding text."""
	import re
	
	# Look for figure captions near this image
	fig_patterns = [
		rf"\bfig\.?\s*{img_index}\b[:\.\-\s]*([^\n\r]+)",
		rf"\bfigure\s*{img_index}\b[:\.\-\s]*([^\n\r]+)",
		r"\bfig\.?\s*\d+[:\.\-\s]*([^\n\r]+)",
		r"\bfigure\s*\d+[:\.\-\s]*([^\n\r]+)"
	]
	
	for pattern in fig_patterns:
		matches = re.findall(pattern, page_text, re.I)
		if matches:
			caption = matches[0].strip()
			if len(caption) > 10:  # Reasonable caption length
				return f"Figure {img_index}: {caption[:100]}"
	
	# Look for technical context on the page
	technical_terms = re.findall(r"\b(gear|vibration|wear|failure|damage|spall|fatigue|bearing|shaft|tooth)\w*\b", page_text.lower())
	if technical_terms:
		unique_terms = list(set(technical_terms[:3]))
		return f"Figure {img_index}: Technical diagram ({', '.join(unique_terms)})"
	
	return f"Figure {img_index}: Image from page {page_num}"


# Placeholder for future LlamaIndex integration
def _try_llamaindex_extraction(path: Path):
	"""Future: Use LlamaIndex/LlamaParse for enhanced PDF parsing.
	This function is a placeholder for when we want to integrate LlamaIndex
	as an alternative to the current extraction pipeline.
	"""
	# TODO: Implement LlamaIndex integration
	# - Check for LLAMA_CLOUD_API_KEY
	# - Use LlamaParse for better table/figure detection
	# - Convert results to our element format
	# - Fall back to current extractors if unavailable
	return []

try:
	from unstructured.partition.pdf import partition_pdf
	from unstructured.partition.docx import partition_docx
	from unstructured.partition.text import partition_text
except Exception:  # pragma: no cover - allow import even if extras missing
	partition_pdf = partition_docx = partition_text = None  # type: ignore


# --- feature flag helpers ----------------------------------------------------
def _is_truthy(val: str | None) -> bool:
	return bool(val) and (val.lower() in ("1", "true", "yes", "on", "y"))


def _env_enabled(name: str, default: bool | None) -> bool | None:
	"""Return True/False if env explicitly set; return default when unset.
	Using default=None means "auto" (try if available).
	"""
	raw = os.getenv(name)
	if raw is None:
		return default
	return _is_truthy(raw)


def _export_tables_to_files(elements, path: Path) -> None:
	"""Persist detected table elements to data/elements as Markdown and CSV files.
	Attach file paths back into element.metadata as table_md_path/table_csv_path when possible.
	Always attempt to produce both formats (md and csv) for convenience.
	"""
	base_dir = Path("data") / "elements"
	base_dir.mkdir(parents=True, exist_ok=True)
	# --- helpers for parsing/normalization ---
	import csv as _csv
	from io import StringIO as _StringIO
	import re as _re

	def _parse_md_rows(text: str) -> list[list[str]]:
		lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
		rows: list[list[str]] = []
		for ln in lines:
			if not ln.startswith("|"):
				continue
			cells = [c.strip() for c in ln.split("|")]
			if cells and cells[0] == "":
				cells = cells[1:]
			if cells and cells[-1] == "":
				cells = cells[:-1]
			# skip separator rows like --- or :---:
			if cells and all(_re.fullmatch(r":?-{3,}:?", c or "-") for c in cells):
				continue
			rows.append(cells)
		return rows

	def _parse_csv_rows(text: str) -> list[list[str]]:
		try:
			f = _StringIO(text or "")
			reader = _csv.reader(f)
			return [list(r) for r in reader if any(c.strip() for c in r)]
		except Exception:
			return []

	def _remove_empty_cols(rows: list[list[str]]) -> list[list[str]]:
		if not rows:
			return rows
		w = max(len(r) for r in rows)
		norm = [r + [""] * (w - len(r)) for r in rows]
		keep_idx = []
		for j in range(w):
			col_vals = [norm[i][j].strip() for i in range(len(norm))]
			if any(col_vals):
				keep_idx.append(j)
		return [[row[j] for j in keep_idx] for row in norm]

	def _normalize_repeated_panel_headers(rows: list[list[str]]) -> list[list[str]] | None:
		"""Detect patterns like [Case, Wear depth, Case, Wear depth, ...] and fold into two columns.
		Returns normalized rows or None if no transformation was applied.
		"""
		if not rows or not rows[0] or len(rows[0]) < 4:
			return None
		head = [h.strip() for h in rows[0]]
		lower = [h.lower() for h in head]
		# find pairs of (Case, Wear depth ...)
		pairs = []
		j = 0
		while j < len(lower) - 1:
			is_case = lower[j].startswith("case")
			is_wear = ("wear" in lower[j + 1]) and ("depth" in lower[j + 1])
			if is_case and is_wear:
				pairs.append((j, j + 1))
				j += 2
			else:
				j += 1
		if len(pairs) < 2:
			return None
		# Build long-format rows: [Case, Wear depth ...] from each pair, stacked vertically
		out: list[list[str]] = []
		# header: take from first pair, preserve units text
		out_head = [rows[0][pairs[0][0]].strip(), rows[0][pairs[0][1]].strip()]
		out.append(out_head)
		for i in range(1, len(rows)):
			for (a, b) in pairs:
				val_a = rows[i][a].strip() if a < len(rows[i]) else ""
				val_b = rows[i][b].strip() if b < len(rows[i]) else ""
				if not val_a and not val_b:
					continue
				out.append([val_a, val_b])
		return out

	def _rows_to_markdown(rows: list[list[str]]) -> str:
		if not rows:
			return ""
		w = max(len(r) for r in rows)
		norm = [r + [""] * (w - len(r)) for r in rows]
		head = norm[0]
		sep = ["---"] * w
		fmt = lambda r: "| " + " | ".join(r) + " |"
		return "\n".join([fmt(head), fmt(sep)] + [fmt(r) for r in norm[1:]])

	def _rows_to_csv(rows: list[list[str]]) -> str:
		f = _StringIO()
		writer = _csv.writer(f)
		for r in rows:
			writer.writerow(r)
		return f.getvalue()

	for i, e in enumerate(elements, start=1):
		if str(getattr(e, "category", "")).lower() != "table":
			continue
		text = (getattr(e, "text", "") or "").strip()
		if not text:
			continue
		# Determine file stems (prefer stable table_number if assigned)
		md_obj = getattr(e, "metadata", None)
		table_no = None
		if md_obj is not None:
			try:
				table_no = getattr(md_obj, "table_number", None)
			except Exception:
				pass
		stem = f"{path.stem}-table-{int(table_no):02d}" if isinstance(table_no, int) else f"{path.stem}-table-{i}"
		md_file = base_dir / f"{stem}.md"
		csv_file = base_dir / f"{stem}.csv"
		# Heuristics and parsing
		looks_markdown = text.lstrip().startswith("|") and "|" in text
		looks_csv = ("," in text and "\n" in text and not looks_markdown)
		rows: list[list[str]] = []
		if looks_markdown:
			rows = _parse_md_rows(text)
		elif looks_csv:
			rows = _parse_csv_rows(text)
		# Attempt normalization if we have parsed rows
		if rows:
			rows = _remove_empty_cols(rows)
			norm = _normalize_repeated_panel_headers(rows)
			if norm:
				rows = norm
		# Write both formats, preferring normalized rows if available
		md_written = False
		csv_written = False
		# Build optional heading from metadata
		head_prefix = ""
		try:
			lbl = getattr(md_obj, "table_label", None)
			cap = getattr(md_obj, "table_caption", None)
			anch = getattr(md_obj, "table_anchor", None)
			if lbl or cap:
				title = (lbl or "Table").strip()
				if cap:
					title = f"{title}: {cap}"
				if anch:
					head_prefix = f"<a name=\"{anch}\"></a>\n### {title}\n\n"
				else:
					head_prefix = f"### {title}\n\n"
		except Exception:
			head_prefix = ""
		if rows:
			try:
				md_file.write_text(head_prefix + _rows_to_markdown(rows), encoding="utf-8")
				md_written = True
			except Exception:
				pass
			try:
				csv_file.write_text(_rows_to_csv(rows), encoding="utf-8")
				csv_written = True
			except Exception:
				pass
		else:
			# Fallback to original text dump as markdown
			try:
				md_file.write_text(head_prefix + text, encoding="utf-8")
				md_written = True
			except Exception:
				pass
		# Attach paths to element metadata if possible
		md_obj = getattr(e, "metadata", None)
		if md_obj is not None:
			try:
				# Some metadata are SimpleNamespace; set attributes dynamically
				if md_written:
					setattr(md_obj, "table_md_path", str(md_file))
				if csv_written:
					setattr(md_obj, "table_csv_path", str(csv_file))
			except Exception:
				pass


def _assign_table_numbers(elements: list, path: Path) -> None:
	"""Assign table numbers by matching page captions to page tables using text overlap.
	Strategy per page:
	1) Extract caption list: [(num, caption_text)...] in reading order.
	2) Collect table elements on that page.
	3) For each caption in order, pick the unassigned table whose text best matches caption tokens.
	4) Any leftover tables get sequential numbers.
	Also store table_label, table_caption, table_anchor.
	"""
	# Map page -> list of (num, caption)
	page_tables: dict[int, list[tuple[int, str | None]]] = {}
	try:
		import fitz  # PyMuPDF
		doc = fitz.open(str(path))
		for pno in range(len(doc)):
			page = doc[pno]
			text = ""
			_get_text = getattr(page, "get_text", None)
			if callable(_get_text):
				try:
					text = _get_text()
				except Exception:
					try:
						text = _get_text("text")
					except Exception:
						text = ""
			import re as _re
			# Ensure string for typing and regex compatibility
			text_str: str = text if isinstance(text, str) else ""
			pairs = []
			for m in _re.finditer(r"\btable\s*(\d+)\s*[:\.\-\s]*([^\n\r]{0,200})", text_str, _re.I):
				try:
					num = int(m.group(1))
					cap = (m.group(2) or "").strip() or None
					pairs.append((num, cap))
				except Exception:
					continue
			if pairs:
				page_tables[pno + 1] = pairs
	except Exception:
		page_tables = {}
	# Group tables by page
	by_page: dict[int, list[object]] = {}
	for e in elements:
		if str(getattr(e, "category", "") or "").lower() != "table":
			continue
		md = getattr(e, "metadata", None)
		p = getattr(md, "page_number", None) if md else None
		page = p if isinstance(p, int) else 10**9
		by_page.setdefault(page, []).append(e)

	def _tok(s: str) -> set[str]:
		import re as _re
		return set(_re.findall(r"[A-Za-z0-9]+", (s or "").lower()))

	used_nums: set[int] = set()
	next_seq = 1
	for page in sorted(by_page.keys()):
		caps = list(page_tables.get(page, []) or [])
		tables = by_page[page]
		assigned: dict[int, tuple[int, str | None]] = {}  # index -> (num, caption)
		# Build token sets for captions and tables
		cap_toks = [(num, cap, _tok(str(cap or ""))) for (num, cap) in caps]
		tbl_toks = [(_tok(getattr(t, "text", "") or "")) for t in tables]
		# Greedy match: for each caption in order, find best table by overlap
		for ci, (num, cap, ctoks) in enumerate(cap_toks):
			best = None
			best_score = -1
			best_idx = None
			for ti in range(len(tables)):
				if ti in assigned:
					continue
				score = len(ctoks & tbl_toks[ti])
				if score > best_score:
					best_score = score
					best_idx = ti
			# Assign this caption number to best table index
			if best_idx is not None and num not in used_nums:
				assigned[best_idx] = (num, cap)
				used_nums.add(num)
		# Assign numbers to tables
		for ti, e in enumerate(tables):
			md = getattr(e, "metadata", None)
			if md is None:
				continue
			if ti in assigned:
				num, cap = assigned[ti]
				# Prefer full label including caption text when available
				label = f"Table {num}: {cap}" if (cap and str(cap).strip()) else f"Table {num}"
				anchor = f"table-{num:02d}"
				try:
					setattr(md, "table_number", int(num))
					setattr(md, "table_label", label)
					setattr(md, "table_caption", cap)
					setattr(md, "table_anchor", anchor)
				except Exception:
					pass
			else:
				# fallback sequential
				while next_seq in used_nums:
					next_seq += 1
				num = next_seq
				next_seq += 1
				used_nums.add(num)
				# No discovered caption; keep simple label
				label = f"Table {num}"
				anchor = f"table-{num:02d}"
				try:
					setattr(md, "table_number", int(num))
					setattr(md, "table_label", label)
					setattr(md, "table_caption", None)
					setattr(md, "table_anchor", anchor)
				except Exception:
					pass


def _estimate_rows_cols_from_text(text: str) -> tuple[int, int]:
	"""Best-effort estimate of rows/cols from markdown/CSV-like text."""
	try:
		lines = [ln for ln in (text or "").splitlines() if ln.strip()]
		if not lines:
			return 0, 0
		# markdown header detection
		if lines[0].lstrip().startswith("|") and "|" in lines[0]:
			cols = max(2, lines[0].count("|") - 1)
			return len(lines), cols
		# CSV heuristic
		if "," in lines[0]:
			cols = max(1, lines[0].count(",") + 1)
			return len(lines), cols
		# whitespace separated columns (at least two wide gaps)
		import re as _re
		wcols = 1
		for ln in lines[:5]:
			wcols = max(wcols, len([m for m in _re.finditer(r"\S\s{2,}\S", ln)]))
		return len(lines), max(1, wcols)
	except Exception:
		return 0, 0


def _score_table_candidate(text: str, extractor: str | None, page: int | None) -> int:
	"""Score table quality for ranking in dedupe/limit steps."""
	prio = {
		"pdfplumber": 4,
		"tabula": 3,
		"camelot": 2,
		"synth": 1,
	}.get((extractor or "").split("-")[0], 1)
	rows, cols = _estimate_rows_cols_from_text(text)
	return prio * 100 + rows * 2 + cols


def _dedupe_and_limit_tables(elements: list, per_page_limit: int = 3) -> list:
	"""Deduplicate table elements by normalized text and limit number per page.
	Prefers higher-quality extractors (pdfplumber > tabula > camelot > synth) and larger tables.
	"""
	log = get_logger()
	# Partition by table vs non-table
	non_tables = [e for e in elements if str(getattr(e, "category", "")).lower() != "table"]
	tables = [e for e in elements if str(getattr(e, "category", "")).lower() == "table"]
	if not tables:
		return elements
	# Group by normalized text hash
	import hashlib, re as _re
	def norm_text(t: str) -> str:
		# Lowercase, collapse whitespace and bars/commas to reduce cosmetic diffs
		t = (t or "").lower()
		t = _re.sub(r"[ \t]{2,}", " ", t)
		t = _re.sub(r"\|{2,}", "|", t)
		return t.strip()
	groups: dict[str, list] = {}
	for e in tables:
		text = (getattr(e, "text", "") or "").strip()
		if not text:
			continue
		key = hashlib.md5(norm_text(text).encode("utf-8")).hexdigest()
		groups.setdefault(key, []).append(e)
	# Pick best per group
	chosen: list = []
	for key, cand in groups.items():
		best = None
		best_score = -1
		for e in cand:
			md = getattr(e, "metadata", None)
			extractor = getattr(md, "extractor", None) if md else None
			page = getattr(md, "page_number", None) if md else None
			s = _score_table_candidate(getattr(e, "text", ""), extractor, page)
			if s > best_score:
				best_score = s
				best = e
		if best is not None:
			chosen.append(best)
	# Enforce per-page limit
	by_page: dict[int | None, list] = {}
	for e in chosen:
		md = getattr(e, "metadata", None)
		page = getattr(md, "page_number", None) if md else None
		by_page.setdefault(page, []).append(e)
	final_tables: list = []
	for page, arr in by_page.items():
		arr.sort(key=lambda e: -_score_table_candidate(getattr(e, "text", ""), getattr(getattr(e, "metadata", None), "extractor", None), getattr(getattr(e, "metadata", None), "page_number", None)))
		final_tables.extend(arr[: per_page_limit])
	pruned = len(tables) - len(final_tables)
	if pruned > 0:
		try:
			log.info("Table dedupe/limit: %d -> %d (pruned %d)", len(tables), len(final_tables), pruned)
		except Exception:
			pass
	return non_tables + final_tables


def _pdf_fallback_elements(path: Path):
	"""Lightweight PDF parsing via pypdf as a fallback for Windows.
	Produces simple elements with .text, .category, and .metadata(page_number,id).
	"""
	try:
		from pypdf import PdfReader
	except Exception as e:  # pragma: no cover
		raise RuntimeError("pypdf not installed; cannot parse PDF without unstructured") from e

	reader = PdfReader(str(path))
	elements = []
	for pi, page in enumerate(reader.pages, start=1):
		try:
			text = page.extract_text() or ""
		except Exception:
			text = ""
		# split by blank lines into blocks
		blocks = re.split(r"\n\s*\n", text) if text else []
		if not blocks:
			# single block fallback
			blocks = [text]
		for bi, block in enumerate(blocks, start=1):
			b = block.strip()
			if not b:
				continue
			head = b.splitlines()[0].strip().lower()
			cat = "Text"
			if head.startswith("table") or "|" in b or re.search(r"\btable\b", b, re.I):
				cat = "Table"
			elif head.startswith("figure") or re.search(r"\bfigure\b|\bfig\.\b", b, re.I):
				cat = "Figure"
			el = SimpleNamespace(
				text=b,
				category=cat,
				metadata=SimpleNamespace(page_number=pi, id=f"{path.name}-p{pi}-b{bi}"),
			)
			elements.append(el)
	if not elements:
		# ensure at least one element exists
		elements.append(SimpleNamespace(text="", category="Text", metadata=SimpleNamespace(page_number=1, id=f"{path.name}-p1-b1")))
	print(f"[INFO] Using pypdf fallback for {path.name}: produced {len(elements)} elements")
	return elements


def _analyze_pdf_pages_for_tables(path: Path):
	"""Lightweight page analysis to find likely table pages.
	Uses PyMuPDF to estimate line density (rulings) and text blocks per page.
	Returns list of dicts: {page, line_count, text_blocks} and candidate page list string for Camelot.
	"""
	try:
		import fitz  # PyMuPDF
	except Exception:
		return [], None
	min_lines = int(os.getenv("RAG_LINE_COUNT_MIN", "20"))
	min_blocks = int(os.getenv("RAG_TEXT_BLOCKS_MIN", "2"))
	stats = []
	try:
		doc = fitz.open(str(path))
		for pno in range(len(doc)):
			page = doc[pno]
			# text blocks
			blocks = []
			_get_text = getattr(page, "get_text", None)
			if callable(_get_text):
				try:
					blocks = _get_text("blocks") or []
				except Exception:
					blocks = []
			else:
				blocks = []
			if not isinstance(blocks, list):
				blocks = []
			# drawings: count path segments (lines/rects)
			drawings = []
			get_drawings = getattr(page, "get_drawings", None)
			if callable(get_drawings):
				try:
					drawings = get_drawings() or []
				except Exception:
					drawings = drawings
				if not isinstance(drawings, list):
					drawings = []
			else:
				drawings = []
			# Coarse heuristic: number of drawing primitives correlates with table rulings
			line_count = len(drawings)
			stats.append({"page": pno + 1, "line_count": line_count, "text_blocks": len(blocks)})
		# Decide candidate pages
		candidates = [s["page"] for s in stats if s["line_count"] >= min_lines and s["text_blocks"] >= min_blocks]
		if candidates:
			pages_str = ",".join(str(p) for p in candidates)
		else:
			pages_str = None
		return stats, pages_str
	except Exception:
		return [], None


def load_elements(path: Path):
	"""Return Unstructured elements for PDF/DOCX/TXT with page metadata kept."""
	ext = path.suffix.lower()
	if ext == ".pdf":
		els = None
		if partition_pdf is not None:
			log = get_logger()
			# Effective feature toggles (unset means auto and will be attempted)
			use_hi_res = _env_enabled("RAG_PDF_HI_RES", True)  # default ON
			use_tabula = _env_enabled("RAG_USE_TABULA", None)  # auto
			use_pdfplumber = _env_enabled("RAG_USE_PDFPLUMBER", True)  # default ON
			use_camelot = _env_enabled("RAG_USE_CAMELOT", None)  # auto
			use_synth = _env_enabled("RAG_SYNTH_TABLES", True)  # default ON
			use_images = _env_enabled("RAG_EXTRACT_IMAGES", True)  # default ON
			# Log feature toggles
			log.info(
				"PDF parse toggles: hi_res=%s, tabula=%s, pdfplumber=%s, camelot=%s, synth_tables=%s, extract_images=%s",
				str(use_hi_res), str(use_tabula), str(use_pdfplumber), str(use_camelot), str(use_synth), str(use_images)
			)
			# Optional env toggle to skip hi_res completely
			if use_hi_res:
				# Pass OCR language(s) using preferred 'languages' kw when available; fallback to 'ocr_languages'
				lang = (os.getenv("RAG_OCR_LANG", "eng") or "eng").strip()
				lang_pref = [s.strip() for s in lang.replace("+", ",").split(",") if s.strip()]
				try:
					els = partition_pdf(
						filename=str(path),
						strategy="hi_res",
						infer_table_structure=True,
						languages=lang_pref,
					)
				except TypeError:
					try:
						els = partition_pdf(
							filename=str(path),
							strategy="hi_res",
							infer_table_structure=True,
							ocr_languages=lang,
						)
					except Exception as e:
						log.warning("hi_res PDF parsing failed (%s); falling back to standard parser", e.__class__.__name__)
				except Exception as e:
					log.warning("hi_res PDF parsing failed (%s); falling back to standard parser", e.__class__.__name__)
			# Try standard parser with/without table inference flags
			if els is None:
				for kwargs in (
					{"filename": str(path), "pdf_infer_table_structure": True},
					{"filename": str(path)},
				):
					try:
						els = partition_pdf(**kwargs)  # type: ignore[arg-type]
						break
					except TypeError:
						# Ignore bad kw on older versions
						continue
					except Exception as e:
						log.warning("standard PDF parsing attempt failed (%s); trying next/fallback", e.__class__.__name__)
		# Fallback: basic text extraction per page using pypdf
		if els is None:
			try:
				from pypdf import PdfReader
			except Exception as e:  # pragma: no cover
				raise RuntimeError("No PDF parser available. Install unstructured[all-docs] or pypdf.") from e
			reader = PdfReader(str(path))
			els = []
			for i, page in enumerate(reader.pages, start=1):
				text = page.extract_text() or ""
				# Create a minimal shim object with .text, .category, .metadata
				class _Shim:
					def __init__(self, text, page_number):
						self.text = text
						self.category = "Text"
						class MD:
							def __init__(self, page_number):
								self.page_number = page_number
								self.id = f"p{i}"
						self.metadata = MD(page_number)
				els.append(_Shim(text, i))
			get_logger().warning("Using pypdf fallback for PDF parsing (limited structure detection).")
		# Optional: enrich with Tabula tables (auto if unset)
		if (use_tabula is None) or (use_tabula is True):
			_added = _try_tabula_tables(path)
			if _added:
				els.extend(_added)
				get_logger().info("%s: Tabula extracted %d tables", path.name, len(_added))
	# Optional: pdfplumber table extraction (default ON)
		if (use_pdfplumber is None) or (use_pdfplumber is True):
			_pdfp = _try_pdfplumber_tables(path)
			if _pdfp:
				els.extend(_pdfp)
				get_logger().info("%s: pdfplumber extracted %d tables", path.name, len(_pdfp))
		# Optional: Camelot table extraction (auto if unset) — adaptive pages
		if (use_camelot is None) or (use_camelot is True):
			# Only run Camelot if pdfplumber found too few tables and pages look promising
			min_tables_target = int(os.getenv("RAG_MIN_TABLES_TARGET", "2"))
			_pdfp_count = len(_pdfp or []) if isinstance(locals().get("_pdfp"), list) else 0
			cam_pages = os.getenv("RAG_CAMELOT_PAGES")
			if not cam_pages:
				_, cam_pages = _analyze_pdf_pages_for_tables(path)
			_cam = _try_camelot_tables(path, pages_override=cam_pages)
			if _pdfp_count < min_tables_target and _cam:
				els.extend(_cam)
				get_logger().info("%s: Camelot extracted %d tables (pages=%s)", path.name, len(_cam), cam_pages or "auto-none")
		# De-duplicate tables and limit per page to avoid over-extraction noise
		try:
			limit = int(os.getenv("RAG_TABLES_PER_PAGE", "3"))
			els = _dedupe_and_limit_tables(els, per_page_limit=max(1, limit))
		except Exception:
			pass
		# Assign stable table numbers post-dedupe for consistent naming across runs
		try:
			_assign_table_numbers(els, path)
		except Exception:
			pass
		# Optional: synthesize table-like elements from text blocks (default ON)
		if (use_synth is None) or (use_synth is True):
			try:
				_synth = _synthesize_tables_from_text(els, path)
				if _synth:
					els.extend(_synth)
					get_logger().info("%s: Synthesized %d table-like elements from text", path.name, len(_synth))
			except Exception:
				pass
		# Optional: extract images as figures via PyMuPDF (default ON)
		if (use_images is None) or (use_images is True):
			_figs = _try_extract_images(path)
			if _figs:
				els.extend(_figs)
				get_logger().info("%s: Extracted %d images as figures", path.name, len(_figs))
		# Optional: export tables to markdown/csv files and attach paths in metadata
		try:
			_export_tables_to_files(els, path)
		except Exception:
			get_logger().debug("table export failed; continuing")
		return els
	if ext in (".docx", ".doc"):
		if partition_docx is None:
			raise RuntimeError("unstructured[all-docs] not installed for DOCX parsing")
		return partition_docx(filename=str(path))
	if ext in (".txt",):
		if partition_text is None:
			raise RuntimeError("unstructured not installed for text parsing")
		return partition_text(filename=str(path))
	raise ValueError(f"Unsupported format: {ext}")


def load_many(paths: List[Path]):
	log = get_logger()
	for p in paths:
		els = load_elements(p)
		# Log tables/figures present in the parsed elements (best-effort)
		try:
			cats = [str(getattr(e, "category", "")).lower() for e in els]
			# histogram by category
			hist = {}
			for c in cats:
				hist[c] = hist.get(c, 0) + 1
			if hist:
				log.info(f"{p.name}: element categories -> {sorted(hist.items(), key=lambda x: (-x[1], x[0]))}")
			tables = [getattr(e, "text", "").strip()[:80] for e in els if str(getattr(e, "category", "")).lower() == "table"]
			figures = [getattr(e, "text", "").strip()[:80] for e in els if str(getattr(e, "category", "")).lower() in ("figure", "image")]
			if tables:
				log.info(f"{p.name}: detected {len(tables)} table elements (sample): {tables[:2]}")
			if figures:
				log.info(f"{p.name}: detected {len(figures)} figure elements (sample): {figures[:2]}")
			# Optional raw elements dump for deep debugging
			import os, json
			if os.getenv("RAG_DUMP_ELEMENTS", "").lower() in ("1", "true", "yes"):
				dump_dir = Path("logs") / "elements"
				dump_dir.mkdir(parents=True, exist_ok=True)
				out_path = dump_dir / f"{p.stem}.jsonl"
				with open(out_path, "w", encoding="utf-8") as f:
					for e in els:
						md = getattr(e, "metadata", None)
						rec = {
							"category": str(getattr(e, "category", "")),
							"page_number": getattr(md, "page_number", None) if md else None,
							"id": getattr(md, "id", None) if md else None,
							"text_head": (getattr(e, "text", "") or "").strip()[:200],
						}
						f.write(json.dumps(rec, ensure_ascii=False) + "\n")
		except Exception:
			pass
		yield p, els


def _synthesize_tables_from_text(els, path: Path):
	"""Heuristically detect table-like blocks in text elements and produce Table elements.
	Useful when Tabula/hi_res are unavailable or PDF is mostly text with delimited rows.
	Now with improved filtering to avoid page headers and simple text.
	"""
	try:
		import re as _re
	except Exception:
		return []
	out = []
	count = 0
	for e in els:
		cat = str(getattr(e, "category", "") or "").lower()
		if cat in ("table",):
			continue
		text = (getattr(e, "text", "") or "").strip()
		if not text:
			continue
		
		# Skip obvious page headers and simple text
		lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
		if not lines or len(lines) < 3:  # Need at least 3 lines for a real table
			continue
			
		# Skip page headers like "5 | P a g e"
		if len(lines) <= 2 and any(_re.match(r"^\d+\s*\|\s*P\s+a\s+g\s+e\s*$", ln, _re.I) for ln in lines):
			continue
			
		# Skip if most content is just page numbers or simple headers
		non_header_lines = [ln for ln in lines if not _re.match(r"^\d+\s*\|\s*P\s+a\s+g\s+e", ln, _re.I)]
		if len(non_header_lines) < 2:
			continue
			
		head = lines[0].lower()
		# More specific table header detection
		looks_like_header = _re.search(r"\btable\s*\d+\b", head) and ":" in text
		
		# Count meaningful separators (ignore single | in page headers)
		sep_counts = sum(("\t" in ln) + ("," in ln) + (ln.count("|") >= 3) for ln in lines[:8])
		wide_cols = sum(1 for ln in lines[:8] if _re.search(r"\S\s{3,}\S.*\S\s{3,}\S", ln))  # At least 2 wide columns
		
		# More data-like patterns (numbers, units, technical terms)
		has_data_patterns = any(_re.search(r"\b\d+\.?\d*\s*(mm|μm|MPa|Hz|°C|N|kN|rpm|%)\b", ln, _re.I) for ln in lines[:5])
		has_multiple_columns = any(ln.count("|") >= 3 or ln.count("\t") >= 2 for ln in lines)
		
		# Only synthesize if we have strong evidence of tabular data
		if (looks_like_header or 
		    (sep_counts >= 3 and has_multiple_columns) or 
		    (wide_cols >= 2 and has_data_patterns) or
		    (has_data_patterns and has_multiple_columns and len(lines) >= 4)):
			
			count += 1
			md = getattr(e, "metadata", None)
			page_no = getattr(md, "page_number", None) if md else None
			
			# Generate a better summary for the table
			summary = _generate_table_summary(text)
			
			out.append(
				SimpleNamespace(
					text=text,
					category="Table",
					metadata=SimpleNamespace(
								page_number=page_no, 
								id=f"synth-{path.name}-{count}",
								table_summary=summary,
								extractor="synth",
					),
				)
			)
	return out


def _try_tabula_tables(path: Path):
	"""Extract tables using tabula-py into shim elements. Requires Java + tabula.
	Returns a list of SimpleNamespace(text, category='Table', metadata.page_number,id).
	"""
	try:
		# Import the function explicitly to satisfy static analysis across tabula-py versions
		from tabula import read_pdf as tabula_read_pdf  # type: ignore
	except Exception:
		get_logger().debug("Tabula not available; skip table extraction")
		return []
	try:
		dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, stream=True)
	except Exception as e1:
		get_logger().warning("Tabula stream mode failed (%s); trying lattice", e1.__class__.__name__)
		try:
			dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, lattice=True)
		except Exception as e2:
			get_logger().warning("Tabula failed to extract tables (%s)", e2.__class__.__name__)
			return []
	elements = []
	idx = 0
	for df in dfs or []:
		try:
			# Prefer markdown; fallback to CSV
			try:
				md_text = df.to_markdown(index=False)  # type: ignore[attr-defined]
			except Exception:
				# pandas DataFrame supports to_csv; add ignore for static checkers
				md_text = df.to_csv(index=False)  # type: ignore[call-arg]
			# Generate a semantic table summary to aid chunking
			summary = _generate_table_summary(md_text)
			idx += 1
			elements.append(
				SimpleNamespace(
					text=md_text,
					category="Table",
					metadata=SimpleNamespace(page_number=None, id=f"tabula-{path.name}-{idx}", extractor="tabula", table_summary=summary),
				)
			)
		except Exception:
			continue
	return elements


def _try_pdfplumber_tables(path: Path):
	"""Extract simple tables using pdfplumber per page. No external deps beyond pdfminer.six.
	Returns shim elements with markdown/CSV-ish text.
	"""
	try:
		import pdfplumber  # type: ignore
	except Exception:
		get_logger().debug("pdfplumber not available; skip table extraction")
		return []
	elements = []
	idx = 0
	try:
		with pdfplumber.open(str(path)) as pdf:
			for pno, page in enumerate(pdf.pages, start=1):
				for table in (page.extract_tables() or []):
					try:
						# Convert to markdown-like text
						rows = [[(c or "").strip() for c in row] for row in table]
						if not rows:
							continue
						width = max(len(r) for r in rows)
						rows = [r + [""] * (width - len(r)) for r in rows]
						header = rows[0]
						sep = ["---"] * len(header)
						body = rows[1:]
						fmt = lambda r: "| " + " | ".join(r) + " |"
						md_text = "\n".join([fmt(header), fmt(sep)] + [fmt(r) for r in body])
						summary = _generate_table_summary(md_text)
						idx += 1
						elements.append(
							SimpleNamespace(
								text=md_text,
								category="Table",
								metadata=SimpleNamespace(page_number=pno, id=f"pdfplumber-{path.name}-{idx}", extractor="pdfplumber", table_summary=summary),
							)
						)
					except Exception:
						continue
	except Exception as e:
		get_logger().warning("pdfplumber failed (%s)", e.__class__.__name__)
		return []
	return elements


def _try_camelot_tables(path: Path, pages_override: str | None = None):
	"""Extract tables using Camelot if available. Requires Ghostscript/Poppler; optional.
	"""
	try:
		import camelot  # type: ignore
	except Exception:
		get_logger().debug("Camelot not available; skip table extraction")
		return []
	log = get_logger()
	# Env-driven configuration
	flavors_raw = os.getenv("RAG_CAMELOT_FLAVORS", "lattice,stream")
	flavors = [f.strip() for f in flavors_raw.split(",") if f.strip()]
	pages = pages_override or os.getenv("RAG_CAMELOT_PAGES", "all")
	min_rows = int(os.getenv("RAG_CAMELOT_MIN_ROWS", "3"))
	min_cols = int(os.getenv("RAG_CAMELOT_MIN_COLS", "2"))
	min_acc = float(os.getenv("RAG_CAMELOT_MIN_ACCURACY", "70"))  # 0-100
	max_empty_col_ratio = float(os.getenv("RAG_CAMELOT_MAX_EMPTY_COL_RATIO", "0.6"))
	min_numeric_ratio = float(os.getenv("RAG_CAMELOT_MIN_NUMERIC_RATIO", "0.0"))  # fraction 0-1
	elements = []
	idx = 0
	for flavor in (flavors or ["lattice", "stream"]):
		try:
			# pages can be 'all' or '1,2,3' or '1-3'
			tables = camelot.read_pdf(
				str(path),
				pages=pages,
				flavor=flavor,
				strip_text="\n",
				process_background=True if flavor == "lattice" else False,
			)
		except Exception as e:
			log.warning("Camelot %s failed (%s)", flavor, e.__class__.__name__)
			continue
		for t in getattr(tables, "df", []) or []:
			try:
				# t is DataFrame when accessed as tables[i].df; iterate tables properly
				pass
			except Exception:
				pass
		# Camelot returns a TableList; iterate explicitly
		for ti, tbl in enumerate(tables, start=1):
			try:
				df = getattr(tbl, "df", None)
				if df is None:
					continue
				# Quality thresholds to reduce false positives
				rows, cols = getattr(df, "shape", (0, 0))
				if rows < min_rows or cols < min_cols:
					continue
				# Accuracy from parsing_report if available
				acc = None
				try:
					rep = getattr(tbl, "parsing_report", None)
					if isinstance(rep, dict):
						acc = float(rep.get("accuracy", 0))
				except Exception:
					acc = None
				if acc is not None and acc < min_acc:
					continue
				# Empty column ratio and numeric density
				try:
					import pandas as _pd  # type: ignore
					import re as _re
					_nonempty = []
					_numeric_cells = 0
					_total_cells = 0
					for c in range(cols):
						col = df.iloc[:, c].astype(str).fillna("")
						ne = int(col.str.strip().astype(bool).sum())
						_nonempty.append(ne)
						# numeric detection on non-header rows
						vals = col[1:] if rows > 1 else col
						for v in vals:
							_s = str(v).strip()
							if _s:
								_total_cells += 1
								if _re.search(r"^[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|mm|μm|MPa|Hz|°C|N|kN|rpm))?$", _s, _re.I):
									_numeric_cells += 1
					if cols > 0:
						empty_col_ratio = sum(1 for x in _nonempty if x == 0) / float(cols)
						if empty_col_ratio > max_empty_col_ratio:
							continue
					if _total_cells > 0 and (numeric_ratio := (_numeric_cells / _total_cells)) < min_numeric_ratio:
						continue
				except Exception:
					pass
				try:
					md_text = df.to_markdown(index=False)  # type: ignore[attr-defined]
				except Exception:
					md_text = df.to_csv(index=False)
				summary = _generate_table_summary(md_text)
				idx += 1
				elements.append(
					SimpleNamespace(
						text=md_text,
						category="Table",
						metadata=SimpleNamespace(page_number=None, id=f"camelot-{flavor}-{path.name}-{idx}", extractor=f"camelot-{flavor}", table_summary=summary, camelot_accuracy=acc),
					)
				)
			except Exception:
				continue
	return elements


def _try_extract_images(path: Path):
	"""Extract images from PDF pages as Figure elements using PyMuPDF (fitz).
	Saves images to data/images and creates proper Figure elements with OCR summaries.
	"""
	try:
		import fitz  # PyMuPDF
	except Exception:
		get_logger().debug("PyMuPDF not available; skip image extraction")
		return []
	out_dir = Path("data") / "images"
	out_dir.mkdir(parents=True, exist_ok=True)
	elements = []
	try:
		doc = fitz.open(str(path))
		for pno in range(len(doc)):
			page = doc[pno]
			for img_index, img in enumerate(page.get_images(full=True), start=1):
				xref = img[0]
				pix = fitz.Pixmap(doc, xref)
				# Save as PNG
				fname = f"{path.stem}-p{pno+1}-img{img_index}.png"
				fpath = out_dir / fname
				try:
					if pix.alpha:  # convert RGBA to RGB
						pix = fitz.Pixmap(fitz.csRGB, pix)
					pix.save(fpath)
				finally:
					pix = None  # release
				
				# Try to get text around the image for context (PyMuPDF compatible across versions)
				page_text = ""
				_get_text = getattr(page, "get_text", None)
				if callable(_get_text):
					try:
						# Newer PyMuPDF versions support get_text() defaulting to 'text'
						page_text = _get_text()
					except TypeError:
						# Older signatures require an explicit type
						try:
							page_text = _get_text("text")
						except Exception:
							page_text = ""
				else:
					# Very old PyMuPDF used getText
					_getText = getattr(page, "getText", None)
					if callable(_getText):
						try:
							page_text = _getText("text")
						except Exception:
							page_text = ""
				figure_summary = _generate_figure_summary(str(page_text), pno + 1, img_index)
				
				elements.append(
					SimpleNamespace(
						text=f"[FIGURE] {figure_summary}\nImage file: {fpath}",
						category="Figure",
						metadata=SimpleNamespace(
							page_number=pno + 1, 
							id=f"img-{path.name}-{pno+1}-{img_index}", 
							image_path=str(fpath),
							figure_summary=figure_summary
						),
					)
				)
	except Exception as e:
		get_logger().warning("Image extraction failed (%s)", e.__class__.__name__)
		return []
	return elements

