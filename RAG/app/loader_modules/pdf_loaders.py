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

# OCR capabilities for enhanced figure text extraction
try:
    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore
    OCR_AVAILABLE = True
except Exception as e:
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    OCR_AVAILABLE = False

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
            except Exception as e:
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
        
    except Exception as e:  # pragma: no cover - allow import even if extras missing
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
		except Exception as e:
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
	except Exception as e:
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
				except Exception as e:
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
				except Exception as e:
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
	except Exception as e:
		return [], None

def _extract_text_via_ocr(image_path: Path) -> str:
    """Extract text from image using OCR (Tesseract).
    
    Returns extracted text or empty string if OCR fails or is unavailable.
    """
    if not OCR_AVAILABLE or not Image or not pytesseract or not image_path.exists():
        return ""
    
    try:
        # Open image and perform OCR
        with Image.open(image_path) as img:
            # Convert to RGB if needed for better OCR results
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(img, config='--psm 6').strip()
            
            # Clean up the text - remove excessive whitespace
            cleaned_text = " ".join(extracted_text.split())
            
            # Only return if we got meaningful text (more than just noise)
            if len(cleaned_text) > 10 and any(c.isalpha() for c in cleaned_text):
                return cleaned_text
                
    except Exception as e:
        get_logger().debug(f"OCR extraction failed for {image_path}: {e}")
    
    return ""

def _try_extract_images(path: Path):
    """Extract embedded images using PyMuPDF (fitz).

    Saves images to data/images/<stem>-p<page>-img<idx>.png and returns Figure elements
    with metadata: extractor, page_number, image_path, figure_order (per-page index).
    Controlled by RAG_USE_PYMUPDF (default: on).
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        get_logger().debug("PyMuPDF not available; skip image extraction")
        return []
    
    from RAG.app.loader_modules.loader_utils import _env_enabled
    if not _env_enabled("RAG_USE_PYMUPDF", True):
        return []
    
    log = get_logger()
    from RAG.app.config import settings
    out_dir = settings.paths.DATA_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    elements = []
    
    try:
        doc = fitz.open(str(path))  # type: ignore[misc]
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            images = page.get_images(full=True)
            for idx, img in enumerate(images, start=1):
                try:
                    xref = img[0]
                    base = f"{path.stem}-p{pno+1}-img{idx}.png"
                    out_path = out_dir / base
                    pix = doc.extract_image(xref)
                    iw = pix.get("width") if isinstance(pix, dict) else None
                    ih = pix.get("height") if isinstance(pix, dict) else None
                    # Some images are already PNG/JPEG; keep extension consistent for consumer simplicity
                    with open(out_path, "wb") as f:
                        f.write(pix.get("image", b""))
                    meta = {
                        "extractor": "pymupdf",
                        "page_number": pno + 1,
                        "image_path": str(out_path.as_posix()),
                        "figure_order": idx,
                        "image_width": iw,
                        "image_height": ih,
                    }
                    
                    # Enhanced text extraction: try OCR if enabled
                    ocr_text = ""
                    if _env_enabled("RAG_USE_OCR", True):
                        ocr_text = _extract_text_via_ocr(out_path)
                        if ocr_text:
                            meta["ocr_text"] = ocr_text
                    
                    # Try to find associated text around the figure
                    figure_context = ""
                    try:
                        # Extract text blocks from the page for context
                        _gt = getattr(page, "get_text", None) or getattr(page, "getText", None)
                        text_dict: dict = {}
                        if callable(_gt):
                            try:
                                _res = _gt("dict")
                                if isinstance(_res, dict):
                                    text_dict = _res
                                else:
                                    text_dict = {}
                            except Exception as e:
                                try:
                                    _res2 = _gt()
                                    text_dict = _res2 if isinstance(_res2, dict) else {}
                                except Exception as e2:
                                    text_dict = {}
                        text_blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
                        page_text_parts = []
                        for block in text_blocks:
                            if "lines" in block:
                                for line in block["lines"]:
                                    for span in line.get("spans", []):
                                        text = span.get("text", "").strip()
                                        if text:
                                            page_text_parts.append(text)
                        
                        page_text = " ".join(page_text_parts)
                        # Look for figure references (Figure 1, Fig. 2, etc.)
                        fig_refs = re.findall(rf"(?:Figure|Fig\.?)\s*{idx}[:\.]?\s*([^.!?]*[.!?])", page_text, re.IGNORECASE)
                        if fig_refs:
                            figure_context = " ".join(fig_refs[:2])  # Take first 2 matches
                            meta["figure_context"] = figure_context
                    except Exception as e:
                        pass
                    
                    # Build comprehensive text description
                    txt_parts = [f"[FIGURE]"]
                    
                    # Add OCR text if available
                    if ocr_text:
                        txt_parts.append(f"OCR Text: {ocr_text}")
                    
                    # Add figure context if found
                    if figure_context:
                        txt_parts.append(f"Context: {figure_context}")
                    
                    # Add basic metadata
                    txt_parts.append(f"Image file: {out_path.as_posix()}")
                    txt_parts.append(f"Page: {pno+1}")
                    txt_parts.append(f"Figure {idx}")
                    
                    txt = "\n".join(txt_parts)
                    # Mark tiny images as assets (icons, bullets)
                    try:
                        if isinstance(iw, int) and isinstance(ih, int) and (iw < 128 or ih < 128):
                            meta["is_asset"] = True
                    except Exception as e:
                        pass
                    
                    elements.append(
                        SimpleNamespace(
                            text=txt,
                            category="Figure",
                            metadata=SimpleNamespace(**meta),
                        )
                    )
                except Exception as e:
                    continue
    except Exception as e:
        log.warning(f"PyMuPDF image extraction failed: {e}")
    return elements
