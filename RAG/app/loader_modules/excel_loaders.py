# CRITICAL: Import configuration BEFORE any other imports
from RAG.app.config import settings

# Performance optimizations - use centralized configuration
# These can be overridden by environment variables for backward compatibility

# Now import lightweight libraries only
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import warnings
from types import SimpleNamespace
from RAG.app.logger import get_logger

# Suppress all warnings from external libraries
import logging
import sys
import os

# Redirect stderr to suppress warnings
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

logging.getLogger("openpyxl").setLevel(logging.ERROR)
logging.getLogger("xlrd").setLevel(logging.ERROR)

# Initialize variables for lazy imports
pandas = None
openpyxl = None
xlrd = None

def _import_excel_libraries():
    """Lazy import of Excel libraries - only when needed"""
    global pandas, openpyxl, xlrd
    
    if pandas is not None:
        return  # Already imported
    
    try:
        import pandas as _pandas
        import openpyxl as _openpyxl
        import xlrd as _xlrd
        
        pandas = _pandas
        openpyxl = _openpyxl
        xlrd = _xlrd
        
        # Suppress common warnings globally
        warnings.filterwarnings("ignore", message=".*openpyxl.*")
        warnings.filterwarnings("ignore", message=".*xlrd.*")
        
    except ImportError as e:
        log = get_logger()
        log.error(f"Excel libraries not available: {e}")
        pandas = openpyxl = xlrd = None

def load_excel_file(path: Path) -> List[Dict[str, Any]]:
    """
    Load Excel file and extract all sheets as structured data.
    
    Args:
        path: Path to Excel file
        
    Returns:
        List[Dict[str, Any]]: List of elements representing sheets and data
    """
    log = get_logger()
    
    # Lazy import
    _import_excel_libraries()
    if pandas is None:
        log.error("Excel libraries not available")
        return []
    
    elements = []
    
    try:
        with SuppressStderr():
            # Read all sheets from Excel file
            excel_file = pandas.ExcelFile(path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read sheet as DataFrame
                    df = pandas.read_excel(path, sheet_name=sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Convert DataFrame to structured data
                    sheet_data = {
                        "type": "table",
                        "content": df.to_dict('records'),
                        "columns": df.columns.tolist(),
                        "shape": df.shape,
                        "sheet_name": sheet_name,
                        "file_name": path.name,
                        "file_path": str(path)
                    }
                    
                    # Add metadata
                    sheet_data["metadata"] = {
                        "file_name": path.name,
                        "file_path": str(path),
                        "sheet_name": sheet_name,
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                        "data_type": "excel_table"
                    }
                    
                    elements.append(sheet_data)
                    
                except Exception as e:
                    log.warning(f"Failed to read sheet '{sheet_name}' from {path.name}: {e}")
                    continue
                    
    except Exception as e:
        log.error(f"Failed to load Excel file {path.name}: {e}")
        return []
    
    log.info(f"Loaded {len(elements)} sheets from Excel file: {path.name}")
    return elements

def load_excel_as_text(path: Path) -> List[Dict[str, Any]]:
    """
    Load Excel file and convert to text representation.
    
    Args:
        path: Path to Excel file
        
    Returns:
        List[Dict[str, Any]]: List of text elements
    """
    log = get_logger()
    
    # Lazy import
    _import_excel_libraries()
    if pandas is None:
        log.error("Excel libraries not available")
        return []
    
    elements = []
    
    try:
        with SuppressStderr():
            # Read all sheets from Excel file
            excel_file = pandas.ExcelFile(path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read sheet as DataFrame
                    df = pandas.read_excel(path, sheet_name=sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Convert DataFrame to text
                    text_content = f"Sheet: {sheet_name}\n\n"
                    text_content += df.to_string(index=False)
                    
                    # Create text element
                    text_data = {
                        "type": "text",
                        "content": text_content,
                        "sheet_name": sheet_name,
                        "file_name": path.name,
                        "file_path": str(path)
                    }
                    
                    # Add metadata
                    text_data["metadata"] = {
                        "file_name": path.name,
                        "file_path": str(path),
                        "sheet_name": sheet_name,
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                        "data_type": "excel_text"
                    }
                    
                    elements.append(text_data)
                    
                except Exception as e:
                    log.warning(f"Failed to read sheet '{sheet_name}' from {path.name}: {e}")
                    continue
                    
    except Exception as e:
        log.error(f"Failed to load Excel file {path.name}: {e}")
        return []
    
    log.info(f"Converted {len(elements)} sheets to text from Excel file: {path.name}")
    return elements

def get_excel_summary(path: Path) -> Dict[str, Any]:
    """
    Get summary information about Excel file.
    
    Args:
        path: Path to Excel file
        
    Returns:
        Dict[str, Any]: Summary information
    """
    log = get_logger()
    
    # Lazy import
    _import_excel_libraries()
    if pandas is None:
        log.error("Excel libraries not available")
        return {}
    
    try:
        with SuppressStderr():
            excel_file = pandas.ExcelFile(path)
            
            summary = {
                "file_name": path.name,
                "file_path": str(path),
                "total_sheets": len(excel_file.sheet_names),
                "sheet_names": excel_file.sheet_names,
                "sheets": {}
            }
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pandas.read_excel(path, sheet_name=sheet_name)
                    summary["sheets"][sheet_name] = {
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                        "column_names": df.columns.tolist()
                    }
                except Exception as e:
                    log.warning(f"Failed to read sheet '{sheet_name}' for summary: {e}")
                    summary["sheets"][sheet_name] = {"error": str(e)}
                    
    except Exception as e:
        log.error(f"Failed to get Excel summary for {path.name}: {e}")
        return {}
    
    return summary
