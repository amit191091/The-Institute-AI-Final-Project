#!/usr/bin/env python3
"""
Test script to diagnose RAGAS scoring issues
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def test_ragas_setup():
    """Test RAGAS setup and configuration"""
    print("🔍 Testing RAGAS Setup...")
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"✅ Google API Key: {'SET' if google_key else 'NOT SET'}")
    print(f"✅ OpenAI API Key: {'SET' if openai_key else 'NOT SET'}")
    
    if not google_key and not openai_key:
        print("❌ No API keys found! RAGAS LLM metrics will fail.")
        print("   Set either GOOGLE_API_KEY or OPENAI_API_KEY in your .env file")
        return False
    
    # Check ground truth file
    gt_file = Path("data/gear_wear_QA_groundtruth_EVAL.json")
    if gt_file.exists():
        print(f"✅ Ground truth file found: {gt_file}")
        try:
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
            print(f"✅ Ground truth loaded: {len(gt_data)} questions")
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return False
    else:
        print(f"❌ Ground truth file not found: {gt_file}")
        return False
    
    # Test RAGAS import
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        print("✅ RAGAS imports successful")
    except ImportError as e:
        print(f"❌ RAGAS import failed: {e}")
        print("   Install with: pip install ragas datasets evaluate")
        return False
    
    return True

def test_simple_evaluation():
    """Test a simple RAGAS evaluation"""
    print("\n🧪 Testing Simple RAGAS Evaluation...")
    
    # Simple test dataset
    test_dataset = {
        "question": ["What is the sampling rate?"],
        "answer": ["50 kHz"],
        "contexts": [["The sampling rate was 50 kHz according to the sensor specifications."]],
        "ground_truths": [["50 kHz"]],
        "reference": ["50 kHz"]
    }
    
    try:
        from app.eval_ragas import run_eval_detailed
        summary, per_q = run_eval_detailed(test_dataset)
        print("✅ RAGAS evaluation successful!")
        print(f"📊 Results: {summary}")
        
        # Check for NaN values
        nan_count = sum(1 for v in summary.values() if str(v) == 'nan')
        if nan_count > 0:
            print(f"⚠️  {nan_count} metrics returned NaN - likely API key issue")
        else:
            print("✅ All metrics returned valid scores")
        
        # Check for answer correctness metric
        if "answer_correctness" in summary:
            print(f"✅ Answer correctness metric: {summary['answer_correctness']}")
        else:
            print("⚠️  Answer correctness metric not found")
            
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 RAGAS Diagnostic Test\n")
    
    # Test setup
    if not test_ragas_setup():
        print("\n❌ Setup failed - fix issues above")
        return
    
    # Test evaluation
    if not test_simple_evaluation():
        print("\n❌ Evaluation failed - check API keys and configuration")
        return
    
    print("\n✅ All tests passed! RAGAS should work properly now.")

if __name__ == "__main__":
    main()
