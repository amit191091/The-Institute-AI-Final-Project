#!/usr/bin/env python3
"""Compare routing for specific failed questions from our evaluation analysis."""

import sys
import os
sys.path.append('.')

try:
    from app.agents import route_question_ex
    import json
    
    print("🎯 Testing specific questions that were failing in evaluation")
    print("=" * 65)
    
    # Load some of our evaluation questions to test
    eval_file = "logs/eval_per_question.jsonl"
    if os.path.exists(eval_file):
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = [json.loads(line) for line in f]
        
        # Find some questions that had answer_correctness = 0.0 (complete failures)
        failed_questions = [item for item in eval_data if item.get('answer_correctness') == 0.0]
        
        print(f"Found {len(failed_questions)} completely failed questions in evaluation")
        print("\nTesting routing for failed questions:")
        print("-" * 50)
        
        # Test first 5 failed questions
        for i, item in enumerate(failed_questions[:5]):
            question = item['question']
            old_answer = item['answer'][:100] + "..." if len(item['answer']) > 100 else item['answer']
            
            route, trace = route_question_ex(question)
            
            print(f"\n{i+1}. QUESTION: {question}")
            print(f"   OLD ANSWER: {old_answer}")
            print(f"   NEW ROUTE: {route}")
            print(f"   MATCHED RULES: {trace.get('matched', [])}")
            
            # Check if this is a percentage question that was incorrectly routed before
            if any(word in question.lower() for word in ['percent', '%', 'delta', 'increase', 'by how much']):
                if route == 'needle':
                    print("   ✅ IMPROVEMENT: Percentage question correctly routed to needle")
                else:
                    print(f"   ⚠️  CHECK: Percentage question routed to {route}")
    
    else:
        print(f"Evaluation file {eval_file} not found, testing with hardcoded examples")
        
        # Test some specific percentage questions that were problematic
        test_cases = [
            "What percentage of the wear occurred in the first 100 hours?",
            "By what percentage did the RMS values increase?", 
            "What is the delta between the two measurements?",
        ]
        
        for question in test_cases:
            route, trace = route_question_ex(question)
            print(f"\nQ: {question}")
            print(f"   Route: {route} (should be 'needle' for percentage/delta questions)")
            print(f"   Rules: {trace.get('matched', [])}")
    
    print("\n" + "=" * 65)
    print("SUMMARY: Direct routing without simplify_question should improve:")
    print("✅ Percentage questions → needle agent (extractive answers)")
    print("✅ Delta questions → needle agent (specific numeric values)")
    print("✅ Table questions → table agent (structured data)")
    print("✅ Figure questions → needle agent (figure navigation)")
    print("✅ Summary questions → summary agent (overviews)")
    print("\nThis should resolve the '15 and 45 RPS' type wrong answers!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
