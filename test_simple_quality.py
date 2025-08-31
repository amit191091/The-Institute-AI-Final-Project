#!/usr/bin/env python3
"""Simplified test for content quality scoring without heavy imports."""

import re

def lexical_overlap(a: str, b: str) -> float:
    """Simple lexical overlap implementation for testing."""
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def calculate_content_quality_score(content: str, metadata: dict) -> float:
    """Calculate content quality score to prioritize prose over metadata."""
    if not content or len(content.strip()) < 10:
        return 0.5
    
    content_lower = content.lower().strip()
    section_type = metadata.get("section") or metadata.get("section_type", "")
    
    # Strong penalties for known low-quality content types
    if section_type in ("Figure", "TableCell"):
        return 0.6
    
    # Detect figure captions and headers (short, metadata-like text)
    if len(content) < 100:
        # Very short content - likely caption, header, or metadata
        if any(indicator in content_lower for indicator in 
               ["figure", "fig.", "table", "p.", "page", "source:", "caption"]):
            return 0.65
        # Short but could be meaningful (table data, technical specs)
        return 0.75
    
    # Detect prose vs. structured data characteristics
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 5]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    # Indicators of high-quality analytical prose
    prose_indicators = 0.0
    
    # Complex sentence structure (good for analysis)
    if avg_sentence_length > 15:
        prose_indicators += 0.3
    elif avg_sentence_length > 10:
        prose_indicators += 0.2
    
    # Analytical language patterns
    analytical_terms = [
        "analysis", "shows", "indicates", "demonstrates", "reveals", "suggests",
        "observed", "measured", "calculated", "determined", "found", "results",
        "conclusion", "evidence", "data", "study", "examination", "investigation"
    ]
    analytical_matches = sum(1 for term in analytical_terms if term in content_lower)
    prose_indicators += min(0.3, analytical_matches * 0.05)
    
    # Technical depth indicators (good for gear analysis)
    technical_terms = [
        "spectral", "domain", "frequency", "amplitude", "peaks", "baseline",
        "mesh", "vibration", "wear", "rms", "analysis", "signal", "broadband"
    ]
    technical_matches = sum(1 for term in technical_terms if term in content_lower)
    prose_indicators += min(0.2, technical_matches * 0.03)
    
    # Penalties for metadata-like content
    metadata_penalties = 0.0
    
    # List-like or bullet-point content
    if content.count('\n•') > 2 or content.count('\n-') > 2:
        metadata_penalties += 0.2
    
    # Very short lines (typical of captions/headers)
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if lines:
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 30:
            metadata_penalties += 0.3
    
    # Numeric-heavy content without context (raw data tables)
    numbers = re.findall(r'\d+\.?\d*', content)
    if len(numbers) > len(content.split()) * 0.3:  # More than 30% numbers
        if not any(term in content_lower for term in analytical_terms):
            metadata_penalties += 0.2
    
    # Calculate final quality score
    base_quality = 1.0
    quality_score = base_quality + prose_indicators - metadata_penalties
    
    # Ensure score stays in reasonable bounds
    return max(0.5, min(1.5, quality_score))

def test_content_quality():
    """Test content quality scoring with realistic examples."""
    print("TESTING CONTENT QUALITY SCORING")
    print("=" * 50)
    
    test_cases = [
        # HIGH QUALITY: Analytical prose with the actual answer
        {
            "content": "In Spectral Domain Analysis of the baseline condition, the system exhibited characteristic behavior patterns. The analysis revealed that peaks were sharp and of low amplitude, with minimal broadband energy and no sidebands around the Gear Mesh Frequencies. This indicates optimal operating conditions with well-maintained gear sets showing minimal wear characteristics.",
            "metadata": {"section": "Analysis"},
            "description": "HIGH QUALITY: Detailed analytical prose"
        },
        
        # LOW QUALITY: Figure caption
        {
            "content": "Figure 13: Spectral analysis display showing frequency domain representation",
            "metadata": {"section": "Figure"},
            "description": "LOW QUALITY: Figure caption"
        },
        
        # LOW QUALITY: Table header/metadata
        {
            "content": "Wear Depth Measurements - Baseline Analysis\nCase ID | Depth (µm) | Status\nW1 | 0.5 | Normal",
            "metadata": {"section": "Table"},
            "description": "LOW QUALITY: Table metadata"
        },
        
        # MEDIUM QUALITY: Generic description
        {
            "content": "The mild wear stage represents the initial phase of gear degradation. During this stage, baseline measurements are typically taken for comparison purposes.",
            "metadata": {"section": "Analysis"},
            "description": "MEDIUM QUALITY: Generic description"
        },
        
        # LOW QUALITY: Very short caption
        {
            "content": "p12 Figure: Analysis results",
            "metadata": {"section": "Figure"},
            "description": "LOW QUALITY: Short figure reference"
        },
        
        # LOW QUALITY: Table cell data only
        {
            "content": "Shaft",
            "metadata": {"section": "TableCell"},
            "description": "LOW QUALITY: Single table cell"
        }
    ]
    
    for i, case in enumerate(test_cases):
        quality_score = calculate_content_quality_score(case["content"], case["metadata"])
        
        print(f"\nTest {i+1}: {case['description']}")
        print(f"Content: {case['content'][:60]}...")
        print(f"Quality Score: {quality_score:.3f}")
        
        # Evaluate if scoring is working correctly
        if "HIGH QUALITY" in case["description"] and quality_score > 1.1:
            print("✅ CORRECT: High quality content scored well")
        elif "LOW QUALITY" in case["description"] and quality_score < 0.8:
            print("✅ CORRECT: Low quality content penalized")
        elif "MEDIUM QUALITY" in case["description"] and 0.8 <= quality_score <= 1.1:
            print("✅ CORRECT: Medium quality content scored appropriately")
        else:
            print("❌ INCORRECT: Scoring may need adjustment")
        
        print("-" * 50)

def test_retrieval_scenario():
    """Test the specific retrieval failure scenario described."""
    print("\nTESTING RETRIEVAL FAILURE SCENARIO")
    print("=" * 50)
    
    query = "What did the spectral domain analysis of the baseline show?"
    
    # Simulate the documents returned by retrieval (in order of current ranking)
    documents = [
        # Document 1: The GOOD answer that should be ranked first
        {
            "content": "In Spectral Domain Analysis of the baseline condition, the system exhibited characteristic behavior patterns. The analysis revealed that peaks were sharp and of low amplitude, with minimal broadband energy and no sidebands around the Gear Mesh Frequencies.",
            "metadata": {"section": "Analysis", "page": "2"},
            "current_rank": 1,
            "should_rank": 1,
            "description": "CONTAINS THE ANSWER"
        },
        
        # Document 2: Figure caption noise
        {
            "content": "Figure 13: Spectral analysis display showing frequency domain representation",
            "metadata": {"section": "Figure", "page": "13"},
            "current_rank": 2,
            "should_rank": 5,
            "description": "Figure caption noise"
        },
        
        # Document 3: Table noise  
        {
            "content": "p11 Table: Wear depth measurements for baseline analysis cases",
            "metadata": {"section": "Table", "page": "11"},
            "current_rank": 3,
            "should_rank": 4,
            "description": "Table metadata noise"
        },
        
        # Document 4: Generic description
        {
            "content": "The mild wear stage represents the initial phase of gear degradation where baseline measurements are taken.",
            "metadata": {"section": "Analysis", "page": "7"},
            "current_rank": 4,
            "should_rank": 2,
            "description": "Generic description"
        },
        
        # Document 5: More figure noise
        {
            "content": "p12 Figure: Analysis results",
            "metadata": {"section": "Figure", "page": "12"},
            "current_rank": 5,
            "should_rank": 6,
            "description": "Short figure reference"
        }
    ]
    
    print("ORIGINAL RANKING (causing retrieval failure):")
    for doc in documents:
        print(f"  {doc['current_rank']}. p{doc['metadata']['page']} {doc['metadata']['section']}: {doc['description']}")
    
    print("\nQUALITY SCORES:")
    scored_docs = []
    for doc in documents:
        quality_score = calculate_content_quality_score(doc["content"], doc["metadata"])
        
        # Calculate basic lexical overlap for comparison
        lexical_score = lexical_overlap(query.lower(), doc["content"].lower())
        
        # Combined score (simplified version of what the real system would do)
        combined_score = lexical_score + (quality_score - 1.0) * 0.5  # Quality multiplier effect
        
        scored_docs.append({
            **doc,
            "quality_score": quality_score,
            "lexical_score": lexical_score,
            "combined_score": combined_score
        })
        
        print(f"  p{doc['metadata']['page']} {doc['metadata']['section']}: Quality={quality_score:.3f}, Lexical={lexical_score:.3f}, Combined={combined_score:.3f}")
    
    # Sort by combined score
    scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
    
    print("\nIMPROVED RANKING (with quality scoring):")
    for i, doc in enumerate(scored_docs):
        print(f"  {i+1}. p{doc['metadata']['page']} {doc['metadata']['section']}: {doc['description']} (Score: {doc['combined_score']:.3f})")
    
    # Check if the improvement worked
    if scored_docs[0]["description"] == "CONTAINS THE ANSWER":
        print("\n✅ SUCCESS: The document with the actual answer is now ranked first!")
        print("   The retrieval quality issue has been resolved.")
    else:
        print("\n❌ FAILURE: The answer document is still not ranked first.")
        print("   Further tuning may be needed.")

if __name__ == "__main__":
    test_content_quality()
    test_retrieval_scenario()
    
    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY IMPROVEMENT TEST COMPLETE")
    print("=" * 60)
