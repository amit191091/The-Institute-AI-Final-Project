#!/usr/bin/env python3
"""Test script to check table content for transmission ratio."""

from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
from RAG.app.loaders import load_elements
from RAG.app.chunking import structure_chunks

def main():
    print("Testing table content for transmission ratio...")
    
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
        
        # Process chunks
        chunks = structure_chunks(elements, str(pdf_path))
        print(f"Created {len(chunks)} chunks")
        
        # Look for transmission ratio in table chunks
        print("\nLooking for transmission ratio in table chunks:")
        for i, chunk in enumerate(chunks):
            section = chunk.get('section', 'Unknown')
            if section == 'Table':
                content = chunk.get('content', '')
                label = chunk.get('table_label', '')
                print(f"\nTable {i+1}: {label}")
                print(f"Content: {content[:500]}...")
                
                # Check for transmission ratio
                if 'transmission' in content.lower() or 'ratio' in content.lower() or '18/35' in content:
                    print(f"*** CONTAINS TRANSMISSION RATIO INFO ***")
                    print(f"Full content: {content}")
                    
    except Exception as e:
        print(f"Error processing chunks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
