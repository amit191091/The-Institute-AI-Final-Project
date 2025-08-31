#!/usr/bin/env python3
"""Test script to verify retrieval quality improvements - prose prioritization."""

import os
import sys
from typing import List

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from langchain.schema import Document
from app.retrieve import rerank_candidates, calculate_content_quality_score, calculate_semantic_relevance_boost

def create_test_documents() -> List[Document]:
    """Create test documents that simulate the retrieval failure scenario."""
    
    # This is the HIGH-QUALITY content that should be ranked first
    # Contains the actual answer about spectral domain analysis
    good_content = Document(
        page_content="""In Spectral Domain Analysis of the baseline condition, the system exhibited characteristic behavior patterns. The analysis revealed that peaks were sharp and of low amplitude, with minimal broadband energy and no sidebands around the Gear Mesh Frequencies. This indicates optimal operating conditions with well-maintained gear sets showing minimal wear characteristics. The baseline measurements provide a reference point for comparing subsequent wear progression stages.""",
        metadata={
            "file_name": "Gear wear Failure.pdf",
            "page": "2", 
            "section": "Analysis",
            "section_type": "Analysis"
        }
    )
    
    # These are LOW-QUALITY distractors that currently pollute the results
    figure_caption = Document(
        page_content="Figure 13: Spectral analysis display showing frequency domain representation",
        metadata={
            "file_name": "Gear wear Failure.pdf", 
            "page": "13",
            "section": "Figure",
            "section_type": "Figure"
        }
    )
    
    table_metadata = Document(
        page_content="Wear Depth Measurements - Baseline Analysis\nCase ID | Depth (µm) | Status\nW1 | 0.5 | Normal",
        metadata={
            "file_name": "Gear wear Failure.pdf",
            "page": "11", 
            "section": "Table",
            "section_type": "Table"
        }
    )
    
    generic_description = Document(
        page_content="The mild wear stage represents the initial phase of gear degradation. During this stage, baseline measurements are typically taken for comparison purposes.",
        metadata={
            "file_name": "Gear wear Failure.pdf",
            "page": "7",
            "section": "Analysis", 
            "section_type": "Analysis"
        }
    )
    
    short_caption = Document(
        page_content="p12 Figure: Analysis results",
        metadata={
            "file_name": "Gear wear Failure.pdf",
            "page": "12",
            "section": "Figure",
            "section_type": "Figure"  
        }
    )
    
    return [good_content, figure_caption, table_metadata, generic_description, short_caption]

def test_content_quality_scoring():
    """Test the content quality scoring function."""
    print("=" * 60)
    print("TESTING CONTENT QUALITY SCORING")
    print("=" * 60)
    
    docs = create_test_documents()
    
    for i, doc in enumerate(docs):
        quality_score = calculate_content_quality_score(doc.page_content, doc.metadata)
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:80]}...")
        print(f"Section: {doc.metadata.get('section')}")
        print(f"Quality Score: {quality_score:.3f}")
        print("-" * 40)

def test_semantic_relevance():
    """Test the semantic relevance boost function."""
    print("\n" + "=" * 60)
    print("TESTING SEMANTIC RELEVANCE BOOST")
    print("=" * 60)
    
    query = "What did the spectral domain analysis of the baseline show?"
    docs = create_test_documents()
    
    for i, doc in enumerate(docs):
        relevance_boost = calculate_semantic_relevance_boost(query, doc.page_content, doc.metadata)
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:80]}...")
        print(f"Relevance Boost: {relevance_boost:.3f}")
        print("-" * 40)

def test_full_reranking():
    """Test the complete reranking with quality improvements."""
    print("\n" + "=" * 60)
    print("TESTING FULL RERANKING PIPELINE")
    print("=" * 60)
    
    query = "What did the spectral domain analysis of the baseline show?"
    docs = create_test_documents()
    
    print("BEFORE RERANKING (original order):")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.page_content[:60]}... (p{doc.metadata.get('page')} {doc.metadata.get('section')})")
    
    print("\nAFTER RERANKING:")
    reranked = rerank_candidates(query, docs, top_n=5)
    
    for i, doc in enumerate(reranked):
        score = doc.metadata.get('_score', 0.0)
        quality = doc.metadata.get('_quality', 1.0)
        semantic = doc.metadata.get('_semantic', 0.0)
        section = doc.metadata.get('section', 'Unknown')
        page = doc.metadata.get('page', '?')
        
        print(f"{i+1}. SCORE: {score:.4f} (Q:{quality:.2f}, S:{semantic:.2f}) | p{page} {section}")
        print(f"   Content: {doc.page_content[:80]}...")
        print()
    
    # Check if the analytical content (doc 0) is now ranked first
    best_doc = reranked[0]
    if "peaks were sharp and of low amplitude" in best_doc.page_content:
        print("✅ SUCCESS: High-quality analytical prose is now ranked first!")
        print("   The retrieval quality issue has been resolved.")
    else:
        print("❌ FAILURE: Low-quality content is still ranked higher than analytical prose.")
        print("   The scoring may need further tuning.")

def test_edge_cases():
    """Test edge cases and boundary conditions.""" 
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    edge_docs = [
        Document(page_content="", metadata={"section": "Text"}),  # Empty content
        Document(page_content="Fig. 1", metadata={"section": "Figure"}),  # Very short
        Document(page_content="A" * 2000, metadata={"section": "Analysis"}),  # Very long
        Document(page_content="Analysis shows that measurement indicates results demonstrate findings.", 
                metadata={"section": "Analysis"}),  # Analytical terms without substance
    ]
    
    query = "What does the analysis show?"
    
    for i, doc in enumerate(edge_docs):
        quality = calculate_content_quality_score(doc.page_content, doc.metadata)
        relevance = calculate_semantic_relevance_boost(query, doc.page_content, doc.metadata)
        print(f"Edge case {i+1}: Quality={quality:.3f}, Relevance={relevance:.3f}")
        print(f"  Content: {repr(doc.page_content[:50])}...")

def main():
    """Run all tests."""
    print("RETRIEVAL QUALITY IMPROVEMENT TESTS")
    print("Testing prose prioritization over metadata content")
    
    test_content_quality_scoring()
    test_semantic_relevance()
    test_full_reranking()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
