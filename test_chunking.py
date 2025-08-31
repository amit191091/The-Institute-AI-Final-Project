#!/usr/bin/env python3
"""Test script to check chunking process."""

from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
from RAG.app.loaders import load_elements
from RAG.app.chunking import structure_chunks

def main():
    print("Testing chunking process...")
    
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
        
        # Show chunk details
        print("\nChunk details:")
        for i, chunk in enumerate(chunks):
            section = chunk.get('section', 'Unknown')
            content = chunk.get('content', '')[:200]
            print(f"{i+1}: {section} - {content}...")
        
        # Look for transmission ratio in chunks
        print("\nLooking for transmission ratio in chunks:")
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '').lower()
            if 'transmission ratio' in content or '18/35' in content:
                print(f"Found in chunk {i+1}: {chunk.get('section')}")
                print(f"Content: {chunk.get('content', '')[:300]}...")
                
    except Exception as e:
        print(f"Error processing chunks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
