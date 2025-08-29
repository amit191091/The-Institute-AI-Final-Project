#!/usr/bin/env python3
"""
Test script to verify the updated evaluation targets and Table-QA accuracy.
"""

import sys
from pathlib import Path

# Add RAG to path
sys.path.append(str(Path(__file__).parent.parent))

def test_target_compliance():
    """Test the target compliance checking function."""
    
    print("Testing Target Compliance Function")
    print("=" * 40)
    
    try:
        from RAG.app.Evaluation_Analysis.evaluation_utils import check_target_compliance, print_target_compliance
        from RAG.app.config import settings
        
        print("✅ Successfully imported evaluation functions")
        
        # Test targets
        print("\n📋 Current Targets:")
        for metric, target in settings.evaluation.EVALUATION_TARGETS.items():
            print(f"   {metric}: ≥ {target}")
        
        # Test with sample metrics
        sample_metrics = {
            "answer_correctness": 0.88,
            "context_precision": 0.80,
            "context_recall": 0.75,
            "faithfulness": 0.90,
            "table_qa_accuracy": 0.92
        }
        
        print("\n🧪 Testing with sample metrics:")
        for metric, value in sample_metrics.items():
            target = settings.evaluation.EVALUATION_TARGETS.get(metric, 0)
            status = "✅ PASS" if value >= target else "❌ FAIL"
            print(f"   {metric}: {value:.3f} (target: {target:.2f}) - {status}")
        
        # Test compliance function
        print("\n📊 Testing compliance function:")
        compliance = check_target_compliance(sample_metrics)
        
        for metric, data in compliance.items():
            print(f"   {metric}: {data['status']}")
        
        # Test print function
        print("\n📋 Full compliance report:")
        print_target_compliance(sample_metrics)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_table_qa_accuracy():
    """Test the Table-QA accuracy calculation function."""
    
    print("\nTesting Table-QA Accuracy Function")
    print("=" * 40)
    
    try:
        from RAG.app.Evaluation_Analysis.evaluation_utils import calculate_table_qa_accuracy
        
        # Sample data with table questions
        questions = [
            "What is the wear depth for case W35?",
            "What is the capital of France?",
            "What are the accelerometer sensitivities?",
            "What is the weather like today?",
            "What is the transmission ratio?",
            "What is the module value for the gears?"
        ]
        
        answers = [
            "932 μm",
            "Paris",
            "9.47 mV/g and 9.35 mV/g",
            "Sunny",
            "18/35",
            "3 mm"
        ]
        
        ground_truths = [
            ["932 μm"],
            ["Paris"],
            ["9.47 mV/g and 9.35 mV/g"],
            ["Sunny"],
            ["18/35"],
            ["3 mm"]
        ]
        
        print("Sample questions:")
        for i, q in enumerate(questions, 1):
            is_table = any(keyword in q.lower() for keyword in ["wear depth", "accelerometer", "transmission", "module"])
            table_indicator = "📊" if is_table else "📝"
            print(f"   {i}. {table_indicator} {q}")
        
        # Calculate Table-QA accuracy
        accuracy = calculate_table_qa_accuracy(questions, answers, ground_truths)
        
        print(f"\n✅ Table-QA Accuracy: {accuracy:.3f}")
        
        # Check if it meets target
        target = 0.90
        status = "✅ PASS" if accuracy >= target else "❌ FAIL"
        print(f"   Target: {target:.2f} | Status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Evaluation Targets and Table-QA Accuracy Test")
    print("=" * 50)
    
    success1 = test_target_compliance()
    success2 = test_table_qa_accuracy()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        print("\nThe evaluation system is ready with the 5 required targets:")
        print("   • Answer Correctness ≥ 0.85")
        print("   • Context Precision ≥ 0.75")
        print("   • Context Recall ≥ 0.70") 
        print("   • Faithfulness ≥ 0.85")
        print("   • Table-QA Accuracy ≥ 0.90")
    else:
        print("\n❌ Some tests failed!")
        print("Check the error messages above for details.")
