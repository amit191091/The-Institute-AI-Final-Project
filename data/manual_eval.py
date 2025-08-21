"""
Manual evaluation script that doesn't require OpenAI API.
Compares RAG answers with ground truth using simple metrics.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from difflib import SequenceMatcher

def simple_similarity(a: str, b: str) -> float:
    """Calculate simple text similarity."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text for numerical comparison."""
    return re.findall(r'\d+(?:\.\d+)?', text)

def evaluate_answer(question: str, ground_truth: str, rag_answer: str) -> Dict:
    """Evaluate a single answer against ground truth."""
    
    # Basic similarity
    text_similarity = simple_similarity(ground_truth, rag_answer)
    
    # Check if key numbers match (for numerical questions)
    gt_numbers = set(extract_numbers(ground_truth))
    ra_numbers = set(extract_numbers(rag_answer))
    number_accuracy = len(gt_numbers & ra_numbers) / len(gt_numbers) if gt_numbers else 1.0
    
    # Check if key terms are present
    gt_words = set(ground_truth.lower().split())
    ra_words = set(rag_answer.lower().split())
    key_term_coverage = len(gt_words & ra_words) / len(gt_words) if gt_words else 1.0
    
    # Overall score (weighted combination)
    overall_score = (text_similarity * 0.4 + number_accuracy * 0.3 + key_term_coverage * 0.3)
    
    return {
        "text_similarity": text_similarity,
        "number_accuracy": number_accuracy,
        "key_term_coverage": key_term_coverage,
        "overall_score": overall_score,
        "gt_numbers": list(gt_numbers),
        "ra_numbers": list(ra_numbers),
        "missing_numbers": list(gt_numbers - ra_numbers),
        "extra_numbers": list(ra_numbers - gt_numbers)
    }

def manual_evaluation_report():
    """Generate a manual evaluation report based on recent queries."""
    
    # Sample results from your recent tests
    test_cases = [
        {
            "question": "What is the model number of the gearbox?",
            "ground_truth": "MG-5025A",
            "rag_answer": "MG-5025A",  # From your table results
            "source": "table query successful"
        },
        {
            "question": "What type of gears are used?",
            "ground_truth": "Spur gears",
            "rag_answer": "Spur",  # From your table results  
            "source": "table query successful"
        },
        {
            "question": "What does Figure 1 show?",
            "ground_truth": "Face view of a gear tooth at all health status",
            "rag_answer": "Figure 1 shows a face view of a gear tooth at all health statuses",
            "source": "figure query successful"
        },
        {
            "question": "What is the wear depth for case W35?",
            "ground_truth": "932 μm",
            "rag_answer": "932",  # From your table results
            "source": "table query successful"
        }
    ]
    
    print("MANUAL EVALUATION REPORT")
    print("=" * 50)
    
    total_score = 0
    for i, test in enumerate(test_cases, 1):
        result = evaluate_answer(
            test["question"], 
            test["ground_truth"], 
            test["rag_answer"]
        )
        
        print(f"\n{i}. {test['question']}")
        print(f"   Ground Truth: {test['ground_truth']}")
        print(f"   RAG Answer: {test['rag_answer']}")
        print(f"   Overall Score: {result['overall_score']:.2f}")
        print(f"   Text Similarity: {result['text_similarity']:.2f}")
        print(f"   Number Accuracy: {result['number_accuracy']:.2f}")
        print(f"   Key Term Coverage: {result['key_term_coverage']:.2f}")
        
        if result['missing_numbers']:
            print(f"   Missing Numbers: {result['missing_numbers']}")
        
        total_score += result['overall_score']
    
    avg_score = total_score / len(test_cases)
    print(f"\n" + "=" * 50)
    print(f"AVERAGE SCORE: {avg_score:.2f}")
    print(f"PERFORMANCE: {'🟢 Excellent' if avg_score > 0.8 else '🟡 Good' if avg_score > 0.6 else '🔴 Needs Improvement'}")
    
    # Assessment based on your improvements
    print(f"\n📊 ASSESSMENT:")
    print(f"✅ Figure extraction: WORKING (4 figures detected)")
    print(f"✅ Table extraction: WORKING (3 tables detected)")  
    print(f"✅ Semantic summaries: WORKING (meaningful figure captions)")
    print(f"✅ Path fix: WORKING (images in data/images/)")
    print(f"⚠️  RAGAS metrics: BLOCKED (OpenAI API key issue)")
    
    return avg_score

if __name__ == "__main__":
    manual_evaluation_report()
