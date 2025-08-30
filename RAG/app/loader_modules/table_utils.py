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
from RAG.app.loader_modules.element_types import Element

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


def _table_to_markdown(table) -> str:
	"""Convert a sequence of rows (list[list]) into GitHub-flavored markdown.
	
	This function is extracted from app/loaders.py for enhanced table processing.
	"""
	if not table or len(table) < 2:
		return ""
	
	lines = []
	def _clean(cell) -> str:
		# Replace newlines and pipes inside cells to avoid breaking markdown rows
		s = str(cell or "").replace("\r", " ").replace("\n", " ")
		s = " ".join(s.split())  # collapse repeated spaces
		s = s.replace("|", "/")  # avoid pipe-conflicts in markdown formatting
		return s

	# Header (single logical line after cleaning)
	header = "| " + " | ".join(_clean(cell) for cell in table[0]) + " |"
	lines.append(header)
	
	# Separator
	separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
	lines.append(separator)
	
	# Data rows
	for row in table[1:]:
		data_row = "| " + " | ".join(_clean(cell) for cell in row) + " |"
		lines.append(data_row)
	
	return "\n".join(lines)


def _export_page_text_to_files(elements, pdf_path: Path) -> None:
	"""Write per-page raw text files under data/elements/text/<stem>/p{page}.txt.
	
	This is lossless and does not truncate. If multiple text elements exist per page,
	they will be appended in order of appearance.
	"""
	try:
		from RAG.app.config import settings
		out_dir = settings.paths.DATA_DIR / "elements" / "text" / pdf_path.stem
		out_dir.mkdir(parents=True, exist_ok=True)
		# Build map page -> list[text]
		from collections import defaultdict
		page_texts = defaultdict(list)
		for el in elements:
			if getattr(el, "category", getattr(el, "type", "")) in ("text", "Text"):
				md = getattr(el, "metadata", None) or {}
				page = md.get("page_number")
				if page is None:
					continue
				t = (getattr(el, "text", "") or "")
				if t:
					page_texts[int(page)].append(str(t))
		# Write files and an index.json
		import json as _json
		index = []
		for page in sorted(page_texts.keys()):
			fp = out_dir / f"p{int(page)}.txt"
			try:
				fp.write_text("\n\n".join(page_texts[page]), encoding="utf-8")
			except Exception:
				# Best-effort; continue
				continue
			try:
				txt = fp.read_text(encoding="utf-8")
				index.append({
					"page": int(page),
					"chars": len(txt),
					"words": len(txt.split()),
					"file": fp.as_posix(),
				})
			except Exception:
				index.append({"page": int(page), "file": fp.as_posix()})
		# Write index.json
		try:
			(out_dir / "index.json").write_text(_json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
		except Exception:
			pass
	except Exception:
		# Non-fatal
		pass


def _dump_elements_jsonl(elements, pdf_path: Path) -> None:
	"""Write all extracted elements as raw JSONL to logs/elements/<stem>-elements.jsonl.
	
	Fields: category, extractor, page_number, table_number (if any), image_path (if any),
	and "text" with full content (no truncation).
	
	Controlled by RAG_DUMP_ELEMENTS environment variable (default: on).
	"""
	from RAG.app.loader_modules.loader_utils import _env_enabled
	if not _env_enabled("RAG_DUMP_ELEMENTS", True):
		return
		
	try:
		from RAG.app.config import settings
		out_dir = settings.paths.LOGS_DIR / "elements"
		out_dir.mkdir(parents=True, exist_ok=True)
		out_path = out_dir / f"{pdf_path.stem}-elements.jsonl"
		import json as _json
		with open(out_path, "w", encoding="utf-8") as f:
			for el in elements:
				md = getattr(el, "metadata", None) or {}
				rec = {
					"file": pdf_path.name,
					"category": getattr(el, "category", getattr(el, "type", "Text")),
					"extractor": md.get("extractor"),
					"page_number": md.get("page_number"),
					"table_number": md.get("table_number"),
					"table_md_path": md.get("table_md_path"),
					"table_csv_path": md.get("table_csv_path"),
					"image_path": md.get("image_path"),
					"figure_order": md.get("figure_order"),
					"text": (getattr(el, "text", "") or ""),
				}
				try:
					f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
				except Exception:
					# Attempt a minimal fallback if JSON serialization barfs
					try:
						rec.pop("text", None)
						f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
					except Exception:
						pass
	except Exception:
		# Non-fatal
		pass


def _export_tables_to_files(elements: List[Element], path: Path) -> None:
    """Persist detected table elements to data/elements as Markdown and CSV files.

    For each table element, write two files:
      - Markdown: <stem>-table-XX.md (always)
      - CSV:      <stem>-table-XX.csv (best-effort parsed from markdown rows)

    Attach file paths back into element.metadata as table_md_path/table_csv_path
    and set a stable table_number (1-based) if not already present.

    Controlled by env RAG_EXPORT_TABLES (default: on).
    """
    from RAG.app.loader_modules.loader_utils import _env_enabled
    if not _env_enabled("RAG_EXPORT_TABLES", True):
        return
    try:
        from RAG.app.config import settings
        out_dir = settings.paths.DATA_DIR / "elements"
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    # Determine document-order tables
    tbl_indices = [i for i, e in enumerate(elements) if str(getattr(e, "category", "")).lower() == "table"]
    for order, idx in enumerate(tbl_indices, start=1):
        e = elements[idx]
        text = (getattr(e, "text", "") or "").strip()
        if not text:
            continue
        stem = f"{path.stem}-table-{order:02d}"
        md_path = out_dir / f"{stem}.md"
        csv_path = out_dir / f"{stem}.csv"
        # Write Markdown as-is
        try:
            md_path.write_text(text, encoding="utf-8")
        except Exception:
            continue
        # Best-effort CSV from markdown row lines
        try:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            csv_rows = []
            for ln in lines:
                if not ln.startswith("|"):
                    continue
                if "---" in ln:
                    # header separator
                    continue
                cells = [c.strip() for c in ln.strip("|").split("|")]
                # CSV escaping of quotes
                csv_rows.append(
                    ",".join(f'"{c.replace("\"", "\"\"")}"' for c in cells)
                )
            if csv_rows:
                csv_path.write_text("\n".join(csv_rows), encoding="utf-8")
        except Exception:
            # ignore CSV failures; MD is still useful
            pass
        # Attach metadata
        try:
            meta = getattr(e, "metadata", None)
            if isinstance(meta, dict):
                meta.setdefault("table_number", order)
                meta["table_md_path"] = md_path.as_posix()
                if csv_path.exists():
                    meta["table_csv_path"] = csv_path.as_posix()
            elif meta is not None:
                # SimpleNamespace or other attr-based containers
                try:
                    if getattr(meta, "table_number", None) is None:
                        setattr(meta, "table_number", order)
                    setattr(meta, "table_md_path", md_path.as_posix())
                    if csv_path.exists():
                        setattr(meta, "table_csv_path", csv_path.as_posix())
                except Exception:
                    pass
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
