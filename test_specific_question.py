#!/usr/bin/env python3
"""Quick test of specific evaluation question that was failing"""

import sys
sys.path.append('.')

from langchain.schema import Document

def test_specific_question():
    """Test the exact question from eval_per_question.jsonl that was failing"""
    
    print("=== Testing Specific Failing Question ===")
    print()
    
    # Create mock context similar to what would be retrieved
    context_docs = [
        Document(
            page_content="""Contemporaneous vibration records reinforced this interpretation. At 45 RPS, an afternoon sequence from 14:12 to 14:54 and a subsequent evening set at 17:58–18:36 exhibited RMS levels elevated by roughly 10–15% above the April 9 reference, with the increases stable across consecutive runs. Spectrally, the shaft and gear-mesh lines remained dominant, but sideband families around the mesh frequencies, which had been faint feathering in early April, now emerged as systematic low-amplitude bands separated by shaft frequency.""",
            metadata={'source': 'Gear wear Failure.pdf', 'page': '4'}
        )
    ]
    
    # Test our entity extraction
    from app.agents import _extract_simple_entities
    
    question = "By approximately how much did RMS rise above April 9 levels at 45 RPS during moderate wear?"
    
    result = _extract_simple_entities(question, context_docs)
    
    print(f"Question: {question}")
    print(f"Expected Answer: About 10–15%")
    print(f"Our Extraction: {result}")
    print()
    
    if result and "10" in result and "15" in result:
        print("✅ SUCCESS: Our entity extraction correctly found the percentage!")
    else:
        print("❌ NEEDS WORK: Entity extraction didn't find the answer")
        print("   The answer is clearly in the context: 'elevated by roughly 10–15%'")
    
    print()

if __name__ == "__main__":
    test_specific_question()
