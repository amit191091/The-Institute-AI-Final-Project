# CRITICAL: Import configuration BEFORE any other imports
from RAG.app.config import settings

# Performance optimizations - use centralized configuration
# These can be overridden by environment variables for backward compatibility

# Now import lightweight libraries only
from pathlib import Path
from typing import List
import re
import warnings
from types import SimpleNamespace
from RAG.app.logger import get_logger

# Suppress all warnings from external libraries
import logging
import sys
import os

# Redirect stderr to suppress OCR warnings
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("camelot").setLevel(logging.ERROR)
logging.getLogger("tabula").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("Pillow").setLevel(logging.ERROR)

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

def _export_tables_to_files(elements, path: Path) -> None:
	"""Persist detected table elements to data/elements as Markdown and CSV files.
	Attach file paths back into element.metadata as table_md_path/table_csv_path when possible.
	Always attempt to produce both formats (md and csv) for convenience.
	"""
	from RAG.app.config import settings
	base_dir = settings.paths.DATA_DIR / "elements"
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
			log.debug("Table dedupe/limit: %d -> %d (pruned %d)", len(tables), len(final_tables), pruned)
		except Exception:
			pass
	return non_tables + final_tables
