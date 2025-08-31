#!/usr/bin/env python3
"""Test script to check document extraction."""

from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
from RAG.app.loaders import load_elements

def main():
    print("Testing document extraction...")
    
    # Discover input files
    paths = discover_input_paths()
    print(f"Found {len(paths)} files: {[p.name for p in paths]}")
    
    if not paths:
        print("No input files found!")
        return
    
    # Load elements from first file
    pdf_path = paths[0]
    print(f"\nLoading elements from: {pdf_path.name}")
    
    try:
        elements = load_elements(pdf_path)
        print(f"Loaded {len(elements)} elements")
        
        # Show first 10 elements
        print("\nFirst 10 elements:")
        for i, el in enumerate(elements[:10]):
            category = getattr(el, 'category', 'Unknown')
            text = getattr(el, 'text', '')[:100]
            print(f"{i+1}: {category} - {text}...")
        
        # Count by category
        categories = {}
        for el in elements:
            cat = getattr(el, 'category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nElement categories: {categories}")
        
        # Look for tables specifically
        tables = [el for el in elements if getattr(el, 'category', '').lower() == 'table']
        print(f"\nFound {len(tables)} table elements")
        
        for i, table in enumerate(tables[:3]):
            text = getattr(table, 'text', '')[:200]
            print(f"Table {i+1}: {text}...")
            
    except Exception as e:
        print(f"Error loading elements: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
