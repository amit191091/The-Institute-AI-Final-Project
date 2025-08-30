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
		"USE_PYMUPDF": settings.loader.USE_PYMUPDF,
		"USE_LLAMA_PARSE": settings.loader.USE_LLAMA_PARSE,
		"EXTRACT_IMAGES": settings.loader.EXTRACT_IMAGES,
		"SYNTH_TABLES": settings.loader.SYNTH_TABLES,
		"EXPORT_TABLES": settings.loader.EXPORT_TABLES,
		"DUMP_ELEMENTS": settings.loader.DUMP_ELEMENTS,
		"USE_PYMUPDF_TEXT": settings.loader.USE_PYMUPDF_TEXT,
		"PDFPLUMBER_DEBUG": settings.loader.PDFPLUMBER_DEBUG,
		"CAMELOT_DEBUG": settings.loader.CAMELOT_DEBUG,
		"TABULA_DEBUG": settings.loader.TABULA_DEBUG,
	}
	
	return config_map.get(name, default)


def _get_loader_string_setting(name: str, default: str = "") -> str:
	"""Get loader string setting from centralized config with environment variable override."""
	# Check environment variable first (for backward compatibility)
	raw = os.getenv(f"RAG_{name}")
	if raw is not None:
		return raw
	
	# Use centralized configuration
	config_map = {
		"OCR_LANG": settings.loader.OCR_LANG,
		"CAMELOT_FLAVORS": settings.loader.CAMELOT_FLAVORS,
		"CAMELOT_PAGES": settings.loader.CAMELOT_PAGES,
		"EXCLUSIVE_EXTRACTOR": settings.loader.EXCLUSIVE_EXTRACTOR,
	}
	
	return config_map.get(name, default)


def _get_loader_int_setting(name: str, default: int = 0) -> int:
	"""Get loader integer setting from centralized config with environment variable override."""
	# Check environment variable first (for backward compatibility)
	raw = os.getenv(f"RAG_{name}")
	if raw is not None:
		try:
			return int(raw)
		except ValueError:
			return default
	
	# Use centralized configuration
	config_map = {
		"MIN_TABLES_TARGET": settings.loader.MIN_TABLES_TARGET,
		"TABLES_PER_PAGE": settings.loader.TABLES_PER_PAGE,
		"LINE_COUNT_MIN": settings.loader.LINE_COUNT_MIN,
		"TEXT_BLOCKS_MIN": settings.loader.TEXT_BLOCKS_MIN,
		"CAMELOT_MIN_ROWS": settings.loader.CAMELOT_MIN_ROWS,
		"CAMELOT_MIN_COLS": settings.loader.CAMELOT_MIN_COLS,
	}
	
	return config_map.get(name, default)


def _get_loader_float_setting(name: str, default: float = 0.0) -> float:
	"""Get loader float setting from centralized config with environment variable override."""
	# Check environment variable first (for backward compatibility)
	raw = os.getenv(f"RAG_{name}")
	if raw is not None:
		try:
			return float(raw)
		except ValueError:
			return default
	
	# Use centralized configuration
	config_map = {
		"CAMELOT_MIN_ACCURACY": settings.loader.CAMELOT_MIN_ACCURACY,
		"CAMELOT_MAX_EMPTY_COL_RATIO": settings.loader.CAMELOT_MAX_EMPTY_COL_RATIO,
		"CAMELOT_MIN_NUMERIC_RATIO": settings.loader.CAMELOT_MIN_NUMERIC_RATIO,
	}
	
	return config_map.get(name, default)


def _env_enabled(var_name: str, default: bool = False) -> bool:
	"""Return True if the env var is a truthy flag.
	
	Truthy values: 1, true, yes, on (case-insensitive). Defaults to `default`.
	This function is extracted from app/loaders.py for backward compatibility.
	"""
	value = os.environ.get(var_name, "1" if default else "0")
	return str(value).lower() in ("1", "true", "yes", "on")

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
