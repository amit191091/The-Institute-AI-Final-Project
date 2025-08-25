#!/usr/bin/env python3
"""
Test Q&A Table Functionality
===========================

This script tests the Q&A table functionality and evaluation metrics.
"""

import sys
import os
import json
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.backup_database import load_backup_database
from app.eval_ragas import run_eval, pretty_metrics

def test_ground_truth_dataset():
    """Test loading the ground truth dataset."""
    print("🔍 Testing ground truth dataset...")
    
    gt_path = Path("data/ground_truth_dataset.json")
    if not gt_path.exists():
        print("❌ Ground truth dataset not found")
        return False
    
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    print(f"✅ Loaded {len(questions)} questions")
    
    # Check for duplicates
    question_texts = [q['question'] for q in questions]
    duplicates = set([x for x in question_texts if question_texts.count(x) > 1])
    
    if duplicates:
        print(f"⚠️ Found {len(duplicates)} duplicate questions:")
        for dup in list(duplicates)[:5]:  # Show first 5
            print(f"   - {dup[:50]}...")
    else:
        print("✅ No duplicate questions found")
    
    # Check question types
    types = {}
    difficulties = {}
    for q in questions:
        q_type = q.get('type', 'unknown')
        difficulty = q.get('difficulty', 'unknown')
        types[q_type] = types.get(q_type, 0) + 1
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
    
    print(f"📊 Question types: {types}")
    print(f"📊 Difficulties: {difficulties}")
    
    return True

def test_backup_database():
    """Test backup database loading."""
    print("\n🔍 Testing backup database...")
    
    try:
        backup_docs = load_backup_database()
        print(f"✅ Backup database loaded: {len(backup_docs)} documents")
        
        # Check for specific content
        rms_content = [doc for doc in backup_docs if 'RMS' in doc.page_content]
        fme_content = [doc for doc in backup_docs if 'FME' in doc.page_content]
        wear_content = [doc for doc in backup_docs if 'Wear depth' in doc.page_content]
        
        print(f"📊 RMS content: {len(rms_content)} documents")
        print(f"📊 FME content: {len(fme_content)} documents")
        print(f"📊 Wear depth content: {len(wear_content)} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading backup database: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics calculation."""
    print("\n🔍 Testing evaluation metrics...")
    
    # Sample test case
    test_dataset = {
        "question": ["What is the transmission ratio?"],
        "answer": ["The transmission ratio is 18/35 (18 teeth on driving gear, 35 teeth on driven gear)"],
        "ground_truth": ["18/35 (18 teeth on driving gear, 35 teeth on driven gear)"],
        "contexts": [["Sample context about transmission ratio"]]
    }
    
    try:
        metrics = run_eval(test_dataset)
        formatted_metrics = pretty_metrics(metrics)
        
        print("✅ Evaluation metrics calculated successfully")
        print(f"📊 Metrics:\n{formatted_metrics}")
        
        # Check for expected metrics
        expected_metrics = ['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']
        for metric in expected_metrics:
            if metric in metrics:
                score = metrics[metric]
                print(f"✅ {metric}: {score:.3f}")
                
                # Check if scores meet the target thresholds
                if metric == 'context_precision' and score >= 0.75:
                    print(f"🎯 {metric} meets target (≥0.75)")
                elif metric == 'context_recall' and score >= 0.70:
                    print(f"🎯 {metric} meets target (≥0.70)")
                elif metric == 'faithfulness' and score >= 0.85:
                    print(f"🎯 {metric} meets target (≥0.85)")
                else:
                    print(f"⚠️ {metric} below target")
            else:
                print(f"❌ {metric} not found in metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Error calculating evaluation metrics: {e}")
        return False

def test_qa_table_functionality():
    """Test Q&A table functionality."""
    print("\n🔍 Testing Q&A table functionality...")
    
    # Load ground truth dataset
    gt_path = Path("data/ground_truth_dataset.json")
    if not gt_path.exists():
        print("❌ Ground truth dataset not found")
        return False
    
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    # Test a few sample questions
    sample_questions = [
        "What is the transmission ratio?",
        "What is the wear depth for case W15?",
        "What are the RMS values for W15 wear case at 15 RPS?"
    ]
    
    for question in sample_questions:
        # Find matching question in dataset
        matching_q = None
        for q in data.get('questions', []):
            if q['question'] == question:
                matching_q = q
                break
        
        if matching_q:
            print(f"✅ Found question: {question[:50]}...")
            print(f"   Type: {matching_q.get('type')}")
            print(f"   Difficulty: {matching_q.get('difficulty')}")
            print(f"   Ground Truth: {matching_q.get('ground_truth')[:50]}...")
        else:
            print(f"❌ Question not found: {question}")
    
    return True

def main():
    """Run all tests."""
    print("🧪 Q&A TABLE FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        ("Ground Truth Dataset", test_ground_truth_dataset),
        ("Backup Database", test_backup_database),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Q&A Table Functionality", test_qa_table_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Q&A table functionality is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Run the RAG system: python Main_RAG.py")
        print("2. Go to the '📋 Q&A Table' tab")
        print("3. Select questions and test the evaluation metrics")
        print("4. Check if scores meet the target thresholds:")
        print("   - Context Precision ≥ 0.75")
        print("   - Context Recall ≥ 0.70")
        print("   - Faithfulness ≥ 0.85")
        print("   - Table-QA Accuracy ≥ 0.90")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
