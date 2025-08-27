"""
Minimal, clean table extraction for debugging the 4th table issue.
This is a lean, focused version that strips out all complexity.
"""

from pathlib import Path
from types import SimpleNamespace
import os

def extract_tables_minimal(pdf_path: Path):
    """Minimal table extraction using only pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not available")
        return []
    
    elements = []
    
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
                
                element = SimpleNamespace(
                    text=md_text,
                    category="Table",
                    metadata=SimpleNamespace(
                        page_number=page_num,
                        id=f"minimal-{page_num}-{table_idx}",
                        extractor="pdfplumber-minimal"
                    )
                )
                
                elements.append(element)
    
    return elements

def export_tables_minimal(elements, pdf_path: Path):
    """Minimal table export - just write files in order."""
    output_dir = Path("data") / "elements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    table_count = 0
    for element in elements:
        if element.category.lower() != "table":
            continue
            
        table_count += 1
        
        # Create file names
        stem = f"{pdf_path.stem}-table-{table_count:02d}"
        md_file = output_dir / f"{stem}.md"
        csv_file = output_dir / f"{stem}.csv"
        
        # Write markdown
        md_file.write_text(element.text, encoding="utf-8")
        
        # Convert to CSV (basic)
        lines = element.text.split('\n')
        csv_lines = []
        for line in lines:
            if line.startswith('|') and not line.strip().startswith('| ---'):
                # Remove leading/trailing pipes and split
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                csv_lines.append(','.join(f'"{cell}"' for cell in cells))
        
        if csv_lines:
            csv_file.write_text('\n'.join(csv_lines), encoding="utf-8")
        
        print(f"Exported table {table_count}: {md_file.name}")

def debug_minimal_pipeline():
    """Run minimal pipeline to see if we still get 4 tables."""
    pdf_path = Path("Gear wear Failure.pdf")
    
    print("=== Minimal Pipeline Debug ===")
    
    # Extract tables
    elements = extract_tables_minimal(pdf_path)
    print(f"Extracted {len(elements)} table elements")
    
    # Export tables  
    export_tables_minimal(elements, pdf_path)
    
    # Check output
    output_dir = Path("data") / "elements"
    table_files = list(output_dir.glob("*.md"))
    print(f"Created {len(table_files)} table files:")
    for f in sorted(table_files):
        print(f"  {f.name}")

if __name__ == "__main__":
    debug_minimal_pipeline()
