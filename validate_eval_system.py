"""
Simple validation for manual evaluation system
"""

import os
import sys
from pathlib import Path

def validate_evaluation_system():
    """Validate that the evaluation system can be imported and basic components work"""
    
    print("🔍 Validating Manual Evaluation System...")
    
    try:
        # Test imports
        print("  ✓ Testing imports...")
        from app.eval_manual import (
            run_manual_evaluation, 
            EvaluationThresholds, 
            pretty_manual_metrics,
            _is_table_question
        )
        print("  ✅ Imports successful")
        
        # Test threshold creation
        print("  ✓ Testing threshold creation...")
        thresholds = EvaluationThresholds()
        assert thresholds.context_precision == 0.75
        assert thresholds.context_recall == 0.70
        assert thresholds.faithfulness == 0.85
        print("  ✅ Thresholds created successfully")
        
        # Test table question detection
        print("  ✓ Testing table question detection...")
        assert _is_table_question("What is the wear depth?") == True
        assert _is_table_question("What happened?") == False
        print("  ✅ Table detection working")
        
        # Test with empty dataset
        print("  ✓ Testing empty dataset handling...")
        empty_dataset = {"question": [], "answer": [], "reference": [], "contexts": []}
        summary, results = run_manual_evaluation(empty_dataset)
        assert "error" in summary
        print("  ✅ Empty dataset handled correctly")
        
        print("\n✅ Manual evaluation system validation passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_evaluation_system()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
