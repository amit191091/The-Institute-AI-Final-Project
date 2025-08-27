#!/usr/bin/env python3
"""
Minimal table extraction debugging - trace where the 4th table comes from.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def debug_table_pipeline():
    """Debug the full table extraction pipeline to find the 4th table source."""
    
    # Enable debug logging
    os.environ["RAG_PDFPLUMBER_DEBUG"] = "1"
    os.environ["RAG_TABLE_DEDUP_DEBUG"] = "1" 
    os.environ["RAG_EXPORT_TABLES_DEBUG"] = "1"
    
    # Use only pdfplumber to isolate the issue
    os.environ["RAG_USE_PDFPLUMBER_ONLY"] = "1"
    os.environ["RAG_USE_TABULA"] = "0"
    os.environ["RAG_USE_CAMELOT"] = "0"
    os.environ["RAG_SYNTH_TABLES"] = "0"
    os.environ["RAG_USE_LLAMAPARSE"] = "0"
    
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return
    
    print(f"=== Debugging table pipeline for {pdf_path} ===")
    
    # Try to import and run basic pdfplumber extraction
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not available")
        return
    
    all_elements = []
    
    with pdfplumber.open(pdf_path) as pdf:
        table_count = 0
        
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"\n--- Processing Page {page_num} ---")
            
            # Extract tables with basic settings
            tables = page.extract_tables()
            print(f"Found {len(tables)} raw tables")
            
            for table_idx, table in enumerate(tables, 1):
                if not table or len(table) < 2:
                    print(f"  Skipping table {table_idx}: too small ({len(table) if table else 0} rows)")
                    continue
                
                table_count += 1
                print(f"\n  Table {table_count} (page {page_num}, index {table_idx}):")
                print(f"    Dimensions: {len(table)} x {len(table[0]) if table else 0}")
                print(f"    First row: {table[0]}")
                
                # Convert to markdown format for comparison
                md_lines = []
                for row in table:
                    clean_row = [str(cell or "").strip() for cell in row]
                    md_lines.append("| " + " | ".join(clean_row) + " |")
                
                # Add separator after header
                if len(md_lines) > 0:
                    sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
                    md_lines.insert(1, sep)
                
                md_text = "\n".join(md_lines)
                
                # Create element
                element = SimpleNamespace(
                    text=md_text,
                    category="Table",
                    metadata=SimpleNamespace(
                        page_number=page_num,
                        id=f"pdfplumber-{page_num}-{table_idx}",
                        extractor="pdfplumber",
                        table_number=table_count
                    )
                )
                
                all_elements.append(element)
                print(f"    Element created with ID: {element.metadata.id}")
    
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total elements created: {len(all_elements)}")
    
    # Check for duplicates by comparing first rows
    print(f"\n=== DUPLICATE CHECK ===")
    for i, elem1 in enumerate(all_elements):
        first_line1 = elem1.text.split('\n')[0] if elem1.text else ""
        for j, elem2 in enumerate(all_elements[i+1:], i+1):
            first_line2 = elem2.text.split('\n')[0] if elem2.text else ""
            
            # Simple similarity check
            if first_line1 and first_line2:
                words1 = set(first_line1.split())
                words2 = set(first_line2.split())
                overlap = len(words1 & words2)
                total = len(words1 | words2)
                similarity = overlap / total if total > 0 else 0
                
                if similarity > 0.5:
                    print(f"  SIMILAR: Element {i+1} and {j+1} (similarity: {similarity:.2f})")
                    print(f"    {i+1}: {first_line1}")
                    print(f"    {j+1}: {first_line2}")

if __name__ == "__main__":
    debug_table_pipeline()
