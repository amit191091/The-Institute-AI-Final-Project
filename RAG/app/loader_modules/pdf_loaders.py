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

# Initialize variables for lazy imports
partition_pdf = None

def _generate_figure_summary_inline(page_text: str, page_num: int, img_index: int) -> str:
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

def _import_unstructured():
    """Lazy import of unstructured libraries - only when needed"""
    global partition_pdf
    
    if partition_pdf is not None:
        return  # Already imported
    
    try:
        # Only import PDF parser if hi_res is enabled
        if settings.loader.PDF_HI_RES:
            try:
                from unstructured.partition.pdf import partition_pdf as _partition_pdf
                partition_pdf = _partition_pdf
            except Exception:
                partition_pdf = None
        else:
            partition_pdf = None
        
        # Suppress common OCR warnings globally
        warnings.filterwarnings("ignore", message="No languages specified")
        warnings.filterwarnings("ignore", message="defaulting to English")
        warnings.filterwarnings("ignore", message=".*language.*specified.*")
        warnings.filterwarnings("ignore", message=".*defaulting.*English.*")
        warnings.filterwarnings("ignore", message="short text:")
        warnings.filterwarnings("ignore", message=".*short text.*")
        warnings.filterwarnings("ignore", message=".*PDFInfoNotInstalledError.*")
        warnings.filterwarnings("ignore", message=".*hi_res PDF parsing failed.*")
        warnings.filterwarnings("ignore", message=".*falling back to standard parser.*")
        
    except Exception:  # pragma: no cover - allow import even if extras missing
        partition_pdf = None  # type: ignore

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
	min_lines = settings.loader.LINE_COUNT_MIN
	min_blocks = settings.loader.TEXT_BLOCKS_MIN
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

def _try_extract_images(path: Path):
	"""Extract images from PDF pages as Figure elements using PyMuPDF (fitz).
	Saves images to data/images and creates proper Figure elements with OCR summaries.
	"""
	try:
		import fitz  # PyMuPDF
	except Exception:
		get_logger().debug("PyMuPDF not available; skip image extraction")
		return []
	from RAG.app.config import settings
	out_dir = settings.paths.DATA_DIR / "images"
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
				
				# Generate figure summary inline
				figure_summary = _generate_figure_summary_inline(str(page_text), pno + 1, img_index)
				
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
		get_logger().debug("Image extraction failed (%s)", e.__class__.__name__)
		return []
	return elements
