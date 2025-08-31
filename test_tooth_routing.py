#!/usr/bin/env python3
"""Test script to check tooth routing functionality."""

from RAG.app.pipeline import build_pipeline, answer_with_contexts
from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths

def test_tooth_routing():
    print("Testing tooth routing functionality...")
    
    # Build pipeline
    paths = discover_input_paths()
    docs, hybrid, llm = build_pipeline(paths)
    
    # Test tooth 1 query (should route to main report)
    print("\n1. Testing tooth 1 query:")
    ans, ctx = answer_with_contexts(docs, hybrid, llm, 'What is the wear depth for tooth 1?')
    print(f'Answer: {ans}')
    print(f'Context docs: {len(ctx)}')
    for i, doc in enumerate(ctx[:3]):
        print(f'  - Doc {i}: {doc.metadata.get("file_name", "unknown")}: {doc.page_content[:100]}...')
    
    # Test tooth 2 query (should route to database)
    print("\n2. Testing tooth 2 query:")
    ans, ctx = answer_with_contexts(docs, hybrid, llm, 'What is the wear depth for tooth 2?')
    print(f'Answer: {ans}')
    print(f'Context docs: {len(ctx)}')
    for i, doc in enumerate(ctx[:3]):
        print(f'  - Doc {i}: {doc.metadata.get("file_name", "unknown")}: {doc.page_content[:100]}...')
    
    # Test general wear depth query
    print("\n3. Testing general wear depth query:")
    ans, ctx = answer_with_contexts(docs, hybrid, llm, 'What are the wear depth measurements?')
    print(f'Answer: {ans}')
    print(f'Context docs: {len(ctx)}')
    for i, doc in enumerate(ctx[:3]):
        print(f'  - Doc {i}: {doc.metadata.get("file_name", "unknown")}: {doc.page_content[:100]}...')

if __name__ == "__main__":
    test_tooth_routing()
