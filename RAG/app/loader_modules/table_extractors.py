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
		get_logger().debug("Tabula stream mode failed (%s); trying lattice", e1.__class__.__name__)
		try:
			dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, lattice=True)
		except Exception as e2:
			get_logger().debug("Tabula failed to extract tables (%s)", e2.__class__.__name__)
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
			from RAG.app.loader_modules.table_utils import _generate_table_summary
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
						from RAG.app.loader_modules.table_utils import _generate_table_summary
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
		get_logger().debug("pdfplumber failed (%s)", e.__class__.__name__)
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
	# Configuration-driven settings
	flavors_raw = settings.loader.CAMELOT_FLAVORS
	flavors = [f.strip() for f in flavors_raw.split(",") if f.strip()]
	pages = pages_override or settings.loader.CAMELOT_PAGES
	min_rows = settings.loader.CAMELOT_MIN_ROWS
	min_cols = settings.loader.CAMELOT_MIN_COLS
	min_acc = settings.loader.CAMELOT_MIN_ACCURACY  # 0-100
	max_empty_col_ratio = settings.loader.CAMELOT_MAX_EMPTY_COL_RATIO
	min_numeric_ratio = settings.loader.CAMELOT_MIN_NUMERIC_RATIO  # fraction 0-1
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
			# Suppress common Camelot errors - these are expected for many PDFs
			if "ValueError" in str(e.__class__.__name__) or "No tables found" in str(e):
				log.debug("Camelot %s: No tables found or unsupported format (expected for many PDFs)", flavor)
			else:
				log.debug("Camelot %s failed (%s)", flavor, e.__class__.__name__)
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
								if _re.search(r"^[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|mm|╬╝m|MPa|Hz|┬░C|N|kN|rpm))?$", _s, _re.I):
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
				from RAG.app.loader_modules.table_utils import _generate_table_summary
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
		has_data_patterns = any(_re.search(r"\b\d+\.?\d*\s*(mm|╬╝m|MPa|Hz|┬░C|N|kN|rpm|%)\b", ln, _re.I) for ln in lines[:5])
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
			from RAG.app.loader_modules.table_utils import _generate_table_summary
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
