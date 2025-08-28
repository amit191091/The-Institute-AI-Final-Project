#!/usr/bin/env python3
"""Test individual extractors to debug table extraction issues."""

import os
from pathlib import Path
from app.loaders import load_elements

def test_extractors():
    """Test different extractors individually."""
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"🔍 Testing extractors on: {pdf_path}")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        ("pdfplumber only", {"RAG_USE_PDFPLUMBER": "1", "RAG_USE_CAMELOT": "0", "RAG_USE_TABULA": "0", "RAG_USE_LLAMAPARSE": "0", "RAG_USE_UNSTRUCTURED": "0"}),
        ("camelot only", {"RAG_USE_PDFPLUMBER": "0", "RAG_USE_CAMELOT": "1", "RAG_USE_TABULA": "0", "RAG_USE_LLAMAPARSE": "0", "RAG_USE_UNSTRUCTURED": "0"}),
        ("tabula only", {"RAG_USE_PDFPLUMBER": "0", "RAG_USE_CAMELOT": "0", "RAG_USE_TABULA": "1", "RAG_USE_LLAMAPARSE": "0", "RAG_USE_UNSTRUCTURED": "0"}),
        ("all extractors", {"RAG_USE_PDFPLUMBER": "1", "RAG_USE_CAMELOT": "1", "RAG_USE_TABULA": "1", "RAG_USE_LLAMAPARSE": "0", "RAG_USE_UNSTRUCTURED": "1"}),
    ]
    
    for scenario_name, env_vars in scenarios:
        print(f"\n📊 Testing: {scenario_name}")
        print("-" * 40)
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Enable debug logging
        os.environ["RAG_PDFPLUMBER_DEBUG"] = "1"
        os.environ["RAG_CAMELOT_DEBUG"] = "1" 
        os.environ["RAG_TABULA_DEBUG"] = "1"
        os.environ["RAG_TABLE_DEDUP_DEBUG"] = "1"
        os.environ["RAG_EXPORT_TABLES_DEBUG"] = "1"
        
        try:
            elements = load_elements(pdf_path)
            
            # Count by category
            categories = {}
            table_extractors = set()
            
            for e in elements:
                cat = str(getattr(e, "category", "")).lower()
                categories[cat] = categories.get(cat, 0) + 1
                
                if cat == "table":
                    md = getattr(e, "metadata", None)
                    if md:
                        ext = getattr(md, "extractor", "unknown")
                        table_extractors.add(ext)
            
            print(f"  📋 Categories: {dict(sorted(categories.items()))}")
            print(f"  🔧 Table extractors: {sorted(table_extractors)}")
            
            # List table details
            tables = [e for e in elements if str(getattr(e, "category", "")).lower() == "table"]
            for i, table in enumerate(tables, 1):
                md = getattr(table, "metadata", None)
                if md:
                    ext = getattr(md, "extractor", "?")
                    page = getattr(md, "page_number", "?")
                    first_line = (getattr(table, "text", "").splitlines() or [""])[0][:60]
                    print(f"    Table {i}: {ext} (page {page}) - {first_line}...")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Clean up env vars
        for key in env_vars:
            if key in os.environ:
                del os.environ[key]
    
    # Check exported files
    elements_dir = Path("data/elements")
    if elements_dir.exists():
        table_files = list(elements_dir.glob("*.md"))
        print(f"\n📁 Exported files: {len(table_files)} markdown files")
        for f in sorted(table_files):
            size = f.stat().st_size
            print(f"  - {f.name} ({size} bytes)")

if __name__ == "__main__":
    test_extractors()
