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

# Import from split modules
from RAG.app.loader_modules.pdf_loaders import (
    _import_unstructured as _import_unstructured_pdf,
    _pdf_fallback_elements,
    _analyze_pdf_pages_for_tables,
    _try_extract_images,
    partition_pdf
)

from RAG.app.loader_modules.table_utils import (
    _generate_table_summary,
    _generate_figure_summary,
    _export_tables_to_files,
    _assign_table_numbers,
    _estimate_rows_cols_from_text,
    _score_table_candidate,
    _dedupe_and_limit_tables
)

from RAG.app.loader_modules.table_extractors import (
    _try_tabula_tables,
    _try_pdfplumber_tables,
    _try_camelot_tables,
    _synthesize_tables_from_text
)

from RAG.app.loader_modules.loader_utils import (
    _import_unstructured,
    _is_truthy,
    _get_loader_setting,
    _try_llamaindex_extraction,
    partition_docx,
    partition_text
)

def load_elements(path: Path):
	"""Return Unstructured elements for PDF/DOCX/TXT with page metadata kept."""
	ext = path.suffix.lower()
	if ext == ".pdf":
		els = None
		# Lazy import unstructured if needed
		_import_unstructured_pdf()
		if partition_pdf is not None:
			log = get_logger()
			# Effective feature toggles (unset means auto and will be attempted)
			use_hi_res = _get_loader_setting("PDF_HI_RES", True)  # default ON
			use_tabula = _get_loader_setting("USE_TABULA", False)  # disabled by default
			use_pdfplumber = _get_loader_setting("USE_PDFPLUMBER", True)  # default ON
			use_camelot = _get_loader_setting("USE_CAMELOT", None)  # auto
			use_synth = _get_loader_setting("SYNTH_TABLES", True)  # default ON
			use_images = _get_loader_setting("EXTRACT_IMAGES", True)  # default ON
			# Log feature toggles
			log.info(
				"PDF parse toggles: hi_res=%s, tabula=%s, pdfplumber=%s, camelot=%s, synth_tables=%s, extract_images=%s",
				str(use_hi_res), str(use_tabula), str(use_pdfplumber), str(use_camelot), str(use_synth), str(use_images)
			)
			# Optional env toggle to skip hi_res completely
			if use_hi_res:
				# Pass OCR language(s) using preferred 'languages' kw when available; fallback to 'ocr_languages'
				lang = (os.getenv("RAG_OCR_LANG", settings.loader.OCR_LANG) or settings.loader.OCR_LANG).strip()
				lang_pref = [s.strip() for s in lang.replace("+", ",").split(",") if s.strip()]
				
				# Suppress OCR language warnings
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", message="No languages specified")
					warnings.filterwarnings("ignore", message="defaulting to English")
					
					with SuppressStderr():
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
								log.debug("hi_res PDF parsing failed (%s); falling back to standard parser", e.__class__.__name__)
						except Exception as e:
							log.debug("hi_res PDF parsing failed (%s); falling back to standard parser", e.__class__.__name__)
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
						log.debug("standard PDF parsing attempt failed (%s); trying next/fallback", e.__class__.__name__)
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
			get_logger().debug("Using pypdf fallback for PDF parsing (limited structure detection).")
		
		# Get environment settings for feature toggles
		use_tabula = _get_loader_setting("USE_TABULA", False)
		use_pdfplumber = _get_loader_setting("USE_PDFPLUMBER", True)
		use_camelot = _get_loader_setting("USE_CAMELOT", None)
		use_synth = _get_loader_setting("SYNTH_TABLES", True)
		use_images = _get_loader_setting("EXTRACT_IMAGES", True)
		
		# Optional: enrich with Tabula tables (disabled by default due to Java dependency)
		if use_tabula is True:
			_added = _try_tabula_tables(path)
			if _added:
				els.extend(_added)
				get_logger().debug("%s: Tabula extracted %d tables", path.name, len(_added))
	# Optional: pdfplumber table extraction (default ON)
		if (use_pdfplumber is None) or (use_pdfplumber is True):
			_pdfp = _try_pdfplumber_tables(path)
			if _pdfp:
				els.extend(_pdfp)
				get_logger().debug("%s: pdfplumber extracted %d tables", path.name, len(_pdfp))
		# Optional: Camelot table extraction (auto if unset) - only when needed
		if (use_camelot is None) or (use_camelot is True):
			# Only run Camelot if pdfplumber found too few tables and pages look promising
			min_tables_target = settings.loader.MIN_TABLES_TARGET
			_pdfp_count = len(_pdfp or []) if isinstance(locals().get("_pdfp"), list) else 0
			
			# Skip Camelot if pdfplumber already found enough tables
			if _pdfp_count >= min_tables_target:
				log.debug("%s: Skipping Camelot - pdfplumber found %d tables (target: %d)", path.name, _pdfp_count, min_tables_target)
			else:
				# Only run Camelot if pages look promising for tables
				cam_pages = os.getenv("RAG_CAMELOT_PAGES", settings.loader.CAMELOT_PAGES)
				if not cam_pages:
					_, cam_pages = _analyze_pdf_pages_for_tables(path)
				
				# Skip if no promising pages were found
				if not cam_pages:
					log.debug("%s: Skipping Camelot - no table-like pages detected", path.name)
				else:
					_cam = _try_camelot_tables(path, pages_override=cam_pages)
					if _cam:
						els.extend(_cam)
						get_logger().debug("%s: Camelot extracted %d tables (pages=%s)", path.name, len(_cam), cam_pages or "auto-none")
		# De-duplicate tables and limit per page to avoid over-extraction noise
		try:
			limit = settings.loader.TABLES_PER_PAGE
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
					get_logger().debug("%s: Synthesized %d table-like elements from text", path.name, len(_synth))
			except Exception:
				pass
		# Optional: extract images as figures via PyMuPDF (default ON)
		if (use_images is None) or (use_images is True):
			_figs = _try_extract_images(path)
			if _figs:
				els.extend(_figs)
				get_logger().debug("%s: Extracted %d images as figures", path.name, len(_figs))
		# Optional: export tables to markdown/csv files and attach paths in metadata
		try:
			_export_tables_to_files(els, path)
		except Exception:
			get_logger().debug("table export failed; continuing")
		return els
	if ext in (".docx", ".doc"):
		_import_unstructured()
		if partition_docx is None:
			raise RuntimeError("unstructured[all-docs] not installed for DOCX parsing")
		return partition_docx(filename=str(path))
	if ext in (".txt",):
		_import_unstructured()
		if partition_text is None:
			raise RuntimeError("unstructured not installed for text parsing")
		return partition_text(filename=str(path))
	if ext in (".md",):
		_import_unstructured()
		if partition_text is None:
			raise RuntimeError("unstructured not installed for text parsing")
		return partition_text(filename=str(path))
	if ext in (".csv",):
		_import_unstructured()
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
				log.debug(f"{p.name}: element categories -> {sorted(hist.items(), key=lambda x: (-x[1], x[0]))}")
			tables = [getattr(e, "text", "").strip()[:80] for e in els if str(getattr(e, "category", "")).lower() == "table"]
			figures = [getattr(e, "text", "").strip()[:80] for e in els if str(getattr(e, "category", "")).lower() in ("figure", "image")]
			if tables:
				log.debug(f"{p.name}: detected {len(tables)} table elements (sample): {tables[:2]}")
			if figures:
				log.debug(f"{p.name}: detected {len(figures)} figure elements (sample): {figures[:2]}")
			# Optional raw elements dump for deep debugging
			import os, json
			if settings.loader.DUMP_ELEMENTS:
				from RAG.app.config import settings
				dump_dir = settings.LOGS_DIR / "elements"
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
