#!/usr/bin/env python3
"""
Test script for manual evaluation system
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.eval_manual import run_manual_evaluation, EvaluationThresholds, pretty_manual_metrics

def test_manual_evaluation():
    """Test the manual evaluation system with sample data"""
    
    # Sample dataset
    test_dataset = {
        "question": [
            "On what date was the first onset of wear detected by visual inspection?",
            "When did the system reach the failure stage?",
            "Between which dates did the severe wear stage occur?"
        ],
        "answer": [
            "April 9 [Gear wear Failure.pdf p10]",
            "June 15 [Gear wear Failure.pdf p5]",
            "Between May 14 and June 11 [Gear wear Failure.pdf p10]."
        ],
        "reference": [
            "2023-04-09",
            "2023-06-15",
            "2023-05-14 to 2023-06-11"
        ],
        "contexts": [
            [
                "The first signs of wear appeared on April 9, initially detected through tooth profile images.",
                "A small deviation from the healthy baseline was visible on the flanks of several teeth."
            ],
            [
                "On June 15, the system experienced a critical failure directly attributed to the severe progression of wear.",
                "The failure manifested as a sudden reduction in energy observed across both time and spectral analyses."
            ],
            [
                "The severe wear stage was observed between May 14 and June 11.",
                "During this period, the mechanical degradation accelerated significantly."
            ]
        ]
    }
    
    # Test with default thresholds
    print("🧪 Testing Manual Evaluation System...\n")
    
    try:
        summary, per_q = run_manual_evaluation(test_dataset)
        
        if "error" in summary:
            print(f"❌ Test failed: {summary['error']}")
            return False
        
        print("✅ Manual evaluation completed successfully!")
        print("\n" + "="*60)
        print(pretty_manual_metrics(summary))
        print("="*60)
        
        print(f"\n📊 Per-question results:")
        for i, result in enumerate(per_q, 1):
            print(f"\n**Question {i}:**")
            print(f"  - Answer Correctness: {result['answer_correctness']:.3f}")
            print(f"  - Context Precision: {result['context_precision']:.3f}")
            print(f"  - Context Recall: {result['context_recall']:.3f}")
            print(f"  - Faithfulness: {result['faithfulness']:.3f}")
            print(f"  - Passes Thresholds: {'✅' if result['passes_thresholds'] else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_manual_evaluation()
    sys.exit(0 if success else 1)
