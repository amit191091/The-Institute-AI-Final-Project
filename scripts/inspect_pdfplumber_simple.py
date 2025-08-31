#!/usr/bin/env python3
"""
Simple pdfplumber inspector with focused debugging for the 4th table issue.
This is a lean, targeted debugging script to understand table extraction.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def debug_pdfplumber_extraction():
    """Debug pdfplumber table extraction with minimal overhead."""
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not available")
        return
    
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return
    
    print(f"=== Debugging {pdf_path} ===")
    
    # Open PDF and extract tables page by page
    with pdfplumber.open(pdf_path) as pdf:
        total_tables = 0
        
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"\n--- Page {page_num} ---")
            
            # Try basic table extraction
            tables = page.extract_tables()
            print(f"Raw tables found: {len(tables)}")
            
            for i, table in enumerate(tables, 1):
                if not table:
                    continue
                    
                total_tables += 1
                print(f"\nTable {total_tables} (Page {page_num}, Index {i}):")
                print(f"  Dimensions: {len(table)} rows x {len(table[0]) if table else 0} cols")
                
                # Show first few rows to identify content
                for row_idx, row in enumerate(table[:3]):
                    print(f"  Row {row_idx}: {row}")
                
                # Check for obvious issues
                if len(table) < 2:
                    print("  WARNING: Less than 2 rows - might be caption/header only")
                
                if table and len(table[0]) < 2:
                    print("  WARNING: Less than 2 columns - might be malformed")
                
                # Look for page header contamination
                first_row = table[0] if table else []
                if any(cell and "P a g e" in str(cell) for cell in first_row):
                    print("  WARNING: Contains 'P a g e' - likely page header contamination")
                
                # Look for table caption contamination
                if any(cell and "Table" in str(cell) and ":" in str(cell) for cell in first_row):
                    print("  WARNING: Contains 'Table N:' - likely caption contamination")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total tables extracted: {total_tables}")
    print("Expected: 3 tables")
    if total_tables != 3:
        print(f"ISSUE: Found {total_tables} instead of 3 - need to investigate extra tables")

if __name__ == "__main__":
    debug_pdfplumber_extraction()
