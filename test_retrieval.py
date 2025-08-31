#!/usr/bin/env python3
"""Test script to check retrieval process."""

from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
from RAG.app.loaders import load_elements
from RAG.app.chunking import structure_chunks
from RAG.app.pipeline_modules.pipeline_core import build_pipeline
from RAG.app.retrieve import rerank_candidates

def main():
    print("Testing retrieval process...")
    
    # Discover input files
    paths = discover_input_paths()
    print(f"Found {len(paths)} files: {[p.name for p in paths]}")
    
    if not paths:
        print("No input files found!")
        return
    
    try:
        # Build pipeline
        docs, hybrid_retriever, llm = build_pipeline(paths)
        print(f"Built pipeline with {len(docs)} documents")
        
        # Test question
        question = "What is the transmission ratio (driving/driven)?"
        print(f"\nTesting question: {question}")
        
        # Get candidates from hybrid retriever
        candidates = hybrid_retriever.invoke(question)
        print(f"Retrieved {len(candidates)} candidates")
        
        # Show top candidates
        print("\nTop 5 candidates:")
        for i, doc in enumerate(candidates[:5]):
            content = doc.page_content[:200]
            metadata = doc.metadata
            print(f"{i+1}: {metadata.get('section', 'Unknown')} - {content}...")
        
        # Rerank candidates
        reranked = rerank_candidates(question, candidates, top_n=5)
        print(f"\nReranked to {len(reranked)} candidates")
        
        # Show reranked results
        print("\nTop 5 reranked candidates:")
        for i, doc in enumerate(reranked[:5]):
            content = doc.page_content[:200]
            metadata = doc.metadata
            print(f"{i+1}: {metadata.get('section', 'Unknown')} - {content}...")
            
            # Check if this contains transmission ratio
            if 'transmission ratio' in content.lower() or '18/35' in content:
                print(f"  *** CONTAINS TRANSMISSION RATIO ***")
                
    except Exception as e:
        print(f"Error in retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
