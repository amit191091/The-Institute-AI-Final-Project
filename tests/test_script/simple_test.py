#!/usr/bin/env python3
"""Test extractors individually to debug table extraction issues."""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pdfplumber_simple():
    """Test pdfplumber directly without the complex loader."""
    try:
        import pdfplumber
    except ImportError:
        print("FAIL: pdfplumber not installed")
        return
    
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"FAIL: PDF not found: {pdf_path}")
        return
    
    print(f"Testing pdfplumber on: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_tables = 0
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                print(f"  Page {page_num}: {len(tables)} tables")
                
                for i, table in enumerate(tables):
                    if table and len(table) > 1:
                        total_tables += 1
                        first_row = table[0] if table else []
                        print(f"    Table {i+1}: {len(table)} rows, header: {first_row[:3]}...")
            
            print(f"SUCCESS: Total tables found: {total_tables}")
    
    except Exception as e:
        print(f"FAIL: Error: {e}")

def test_camelot_simple():
    """Test camelot directly."""
    try:
        import camelot
    except ImportError:
        print("FAIL: camelot not installed")
        return
    
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"FAIL: PDF not found: {pdf_path}")
        return
    
    print(f"Testing camelot on: {pdf_path}")
    
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all")
        print(f"SUCCESS: Camelot found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            print(f"  Table {i+1}: Page {table.page}, Shape: {table.df.shape}")
            # Show first few cells of each table
            if not table.df.empty:
                first_row = table.df.iloc[0].values.tolist()[:3] if len(table.df) > 0 else []
                print(f"    Sample: {first_row}")
    
    except Exception as e:
        print(f"FAIL: Error: {e}")

def test_tabula_simple():
    """Test tabula directly."""
    try:
        import tabula
    except ImportError:
        print("FAIL: tabula not installed")
        return
    
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"FAIL: PDF not found: {pdf_path}")
        return
    
    print(f"Testing tabula on: {pdf_path}")
    
    try:
        dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
        print(f"SUCCESS: Tabula found {len(dfs)} tables")
        
        for i, df in enumerate(dfs):
            print(f"  Table {i+1}: Shape: {df.shape}")
            if not df.empty:
                first_row = df.iloc[0].values.tolist()[:3] if len(df) > 0 else []
                print(f"    Sample: {first_row}")
    
    except Exception as e:
        print(f"FAIL: Error: {e}")

if __name__ == "__main__":
    print("Testing PDF table extractors individually")
    print("=" * 60)
    
    test_pdfplumber_simple()
    print()
    test_camelot_simple()
    print()
    test_tabula_simple()
