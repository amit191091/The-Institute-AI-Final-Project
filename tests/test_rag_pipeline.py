#!/usr/bin/env python3
"""
Test script for RAG pipeline with real data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from RAG.app.rag_service import RAGService

def test_rag_pipeline():
    """Test the RAG pipeline with real queries."""
    
    print("🧪 Testing RAG Pipeline with Real Data")
    print("=" * 50)
    
    # Initialize RAG service
    print("1. Initializing RAG service...")
    service = RAGService()
    
    # Check status
    print("2. Checking system status...")
    status = service.get_system_status()
    print(f"   Initialized: {status['initialized']}")
    print(f"   Document count: {status['doc_count']}")
    
    if not status['initialized']:
        print("❌ System not initialized. Building pipeline...")
        try:
            # Try with normalized documents first
            print("   Trying with normalized documents...")
            result = service.run_pipeline(use_normalized=True)
            print(f"✅ Pipeline built: {result['doc_count']} documents loaded")
        except Exception as e:
            print(f"❌ Failed to build pipeline with normalized docs: {e}")
            try:
                # Try with regular documents
                print("   Trying with regular documents...")
                result = service.run_pipeline(use_normalized=False)
                print(f"✅ Pipeline built: {result['doc_count']} documents loaded")
            except Exception as e2:
                print(f"❌ Failed to build pipeline: {e2}")
                return
    
    # Test queries
    test_queries = [
        "What is the wear depth for case W15?",
        "Show me the gear wear progression over time",
        "What are the main findings of the investigation?",
        "What is the RMS level in the baseline condition?",
        "Describe the failure progression timeline"
    ]
    
    print("\n3. Testing queries...")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            result = service.query(query, use_agent=True)
            
            print(f"Answer: {result.get('answer', 'No answer')}")
            print(f"Method: {result.get('method', 'Unknown')}")
            
            sources = result.get('sources', [])
            if sources:
                print(f"Sources: {len(sources)} documents")
                for j, source in enumerate(sources[:2], 1):  # Show first 2 sources
                    if hasattr(source, 'page_content'):
                        preview = source.page_content[:100] + "..." if len(source.page_content) > 100 else source.page_content
                        print(f"  Source {j}: {preview}")
            else:
                print("Sources: No sources found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ RAG Pipeline Test Completed!")

if __name__ == "__main__":
    test_rag_pipeline()
