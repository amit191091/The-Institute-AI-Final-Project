"""
Lean table extraction - focused fix for the 4th table issue.
This replaces the broken loaders.py with a minimal, working version.
"""

from pathlib import Path
from types import SimpleNamespace
import os
import re


def _normalize_text(text):
    """Simple text normalization for deduplication."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    # Remove markdown table formatting for comparison
    normalized = re.sub(r'\|', '', normalized)
    normalized = re.sub(r'-{3,}', '', normalized)
    return normalized


def _are_tables_similar(table1_text, table2_text, threshold=0.7):
    """Check if two tables are similar enough to be considered duplicates."""
    text1 = _normalize_text(table1_text)
    text2 = _normalize_text(table2_text)
    
    if not text1 or not text2:
        return False
    
    # Simple word-based similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if len(words1) == 0 and len(words2) == 0:
        return True
    if len(words1) == 0 or len(words2) == 0:
        return False
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    similarity = intersection / union if union > 0 else 0
    
    return similarity >= threshold


def extract_tables_lean(pdf_path):
    """Lean table extraction with built-in deduplication."""
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not available")
        return []
    
    print(f"Extracting tables from {pdf_path}")
    
    elements = []
    seen_tables = []  # Track tables to prevent duplicates
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables, 1):
                if not table or len(table) < 2:
                    continue
                
                # Convert to markdown
                md_lines = []
                for row in table:
                    clean_row = [str(cell or "").strip() for cell in row]
                    md_lines.append("| " + " | ".join(clean_row) + " |")
                
                # Add header separator
                if len(md_lines) > 0:
                    sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
                    md_lines.insert(1, sep)
                
                md_text = "\n".join(md_lines)
                
                # Check for duplicates
                is_duplicate = False
                for seen_text in seen_tables:
                    if _are_tables_similar(md_text, seen_text):
                        print(f"  DUPLICATE: Skipping table {page_num}-{table_idx} (similar to existing)")
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                # Track this table
                seen_tables.append(md_text)
                
                table_num = len(seen_tables)
                print(f"  Table {table_num}: Page {page_num}, {len(table)} rows x {len(table[0])} cols")
                
                element = SimpleNamespace(
                    text=md_text,
                    category="Table",
                    metadata=SimpleNamespace(
                        page_number=page_num,
                        table_number=table_num,
                        extractor="pdfplumber-lean"
                    )
                )
                
                elements.append(element)
    
    print(f"Extracted {len(elements)} unique tables")
    return elements


def export_tables_lean(elements, pdf_path):
    """Simple table export without overcomplication."""
    output_dir = Path("data") / "elements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for element in elements:
        if element.category.lower() != "table":
            continue
        
        table_num = element.metadata.table_number
        stem = f"{pdf_path.stem}-table-{table_num:02d}"
        
        # Write markdown
        md_file = output_dir / f"{stem}.md"
        md_file.write_text(element.text, encoding="utf-8")
        
        # Write CSV
        csv_file = output_dir / f"{stem}.csv"
        lines = element.text.split('\n')
        csv_lines = []
        for line in lines:
            if line.startswith('|') and not '---' in line:
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                csv_lines.append(','.join(f'"{cell}"' for cell in cells))
        
        if csv_lines:
            csv_file.write_text('\n'.join(csv_lines), encoding="utf-8")
        
        print(f"Exported: {md_file.name}")


def main():
    """Main extraction function."""
    pdf_path = Path("Gear wear Failure.pdf")
    
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return
    
    # Extract tables with deduplication
    elements = extract_tables_lean(pdf_path)
    
    # Export tables
    export_tables_lean(elements, pdf_path)
    
    print("\n=== SUMMARY ===")
    print(f"Successfully extracted and exported {len(elements)} tables")
    if len(elements) != 3:
        print(f"WARNING: Expected 3 tables, got {len(elements)}")


if __name__ == "__main__":
    main()
