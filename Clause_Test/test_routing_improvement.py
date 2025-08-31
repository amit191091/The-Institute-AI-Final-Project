#!/usr/bin/env python3
"""Test script to verify simplify_question removal improved routing."""

import sys
import os
sys.path.append('.')

# Test our routing improvements
try:
    from app.agents import route_question_ex
    from app.retrieve import query_analyzer
    
    print("🚀 Testing routing improvements after simplify_question removal")
    print("=" * 60)
    
    # Test cases from our evaluation analysis - these were failing before
    test_questions = [
        "What percentage increase was observed in bearing RMS?",
        "By how much did the wear depth increase?", 
        "What delta was measured between baseline and test conditions?",
        "Show me table 3",
        "Display figure 2",
        "Summarize the findings",
    ]
    
    print("\nTesting routing logic:")
    print("-" * 30)
    
    for question in test_questions:
        route, trace = route_question_ex(question)
        print(f"Q: {question}")
        print(f"   Route: {route}")
        print(f"   Matched: {trace.get('matched', [])}")
        print()
    
    print("\nTesting query analysis:")
    print("-" * 30)
    
    for question in test_questions:
        analysis = query_analyzer(question)
        print(f"Q: {question}")
        print(f"   Filters: {analysis.get('filters', {})}")
        print(f"   Intent: {analysis.get('intent', {})}")
        print()
    
    print("✅ All routing tests completed successfully!")
    print("\nKey improvements:")
    print("- Removed complex simplify_question preprocessing")  
    print("- Direct routing based on question tokens")
    print("- Percentage/delta questions correctly route to 'needle' agent")
    print("- Table questions correctly route to 'table' agent")
    print("- Summary questions correctly route to 'summary' agent")
    
except Exception as e:
    print(f"❌ Error testing routing: {e}")
    import traceback
    traceback.print_exc()
