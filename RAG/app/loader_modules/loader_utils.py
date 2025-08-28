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
partition_docx = None
partition_text = None

def _import_unstructured():
    """Lazy import of unstructured libraries - only when needed"""
    global partition_docx, partition_text
    
    if partition_docx is not None:
        return  # Already imported
    
    try:
        # Always import DOCX and text parsers (needed for DOCX files)
        from unstructured.partition.docx import partition_docx as _partition_docx
        from unstructured.partition.text import partition_text as _partition_text
        
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
        
        partition_docx = _partition_docx
        partition_text = _partition_text
        
    except Exception:  # pragma: no cover - allow import even if extras missing
        partition_docx = partition_text = None  # type: ignore

# --- feature flag helpers ----------------------------------------------------
def _is_truthy(val: str | None) -> bool:
	return bool(val) and (val.lower() in ("1", "true", "yes", "on", "y"))


def _get_loader_setting(name: str, default: bool | None) -> bool | None:
	"""Get loader setting from centralized config with environment variable override.
	Using default=None means "auto" (try if available).
	"""
	# Check environment variable first (for backward compatibility)
	raw = os.getenv(f"RAG_{name}")
	if raw is not None:
		return _is_truthy(raw)
	
	# Use centralized configuration
	config_map = {
		"PDF_HI_RES": settings.loader.PDF_HI_RES,
		"USE_TABULA": settings.loader.USE_TABULA,
		"USE_CAMELOT": settings.loader.USE_CAMELOT,
		"USE_PDFPLUMBER": settings.loader.USE_PDFPLUMBER,
		"EXTRACT_IMAGES": settings.loader.EXTRACT_IMAGES,
		"SYNTH_TABLES": settings.loader.SYNTH_TABLES,
	}
	
	return config_map.get(name, default)

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
