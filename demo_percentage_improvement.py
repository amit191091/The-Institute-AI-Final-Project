#!/usr/bin/env python3
"""Demonstrate the key improvement: percentage questions routing correctly."""

import sys
import os
sys.path.append('.')

try:
    from app.agents import route_question_ex
    
    print("🔍 DEMONSTRATION: Key Routing Improvement")
    print("=" * 50)
    print("BEFORE: Percentage questions were confused by simplify_question")
    print("AFTER: Direct routing sends percentage questions to needle agent")
    print()
    
    # Test percentage-related questions that were causing issues
    percentage_questions = [
        "What percentage increase was observed in bearing RMS?",
        "By what percentage did the wear depth change?", 
        "What is the percent change between baseline and final measurements?",
        "By how much did the vibration levels increase?",
        "What delta was measured in the gear mesh frequency?",
        "How much did the RMS values exceed baseline levels?",
    ]
    
    print("TESTING PERCENTAGE/DELTA QUESTIONS:")
    print("-" * 40)
    
    for question in percentage_questions:
        route, trace = route_question_ex(question)
        rules = trace.get('matched', [])
        
        print(f"Q: {question}")
        print(f"   → Route: {route}")
        print(f"   → Rules: {rules}")
        
        # Verify this is the correct routing
        if 'delta_percent_needle' in rules:
            print("   ✅ CORRECT: Percentage question routed to needle for extractive answer")
        elif route == 'needle':
            print("   ✅ ACCEPTABLE: Routed to needle via fallback")
        else:
            print(f"   ⚠️  REVIEW: Unexpected route {route}")
        print()
    
    print("EXPECTED IMPROVEMENT:")
    print("Before removal: Questions like 'What percentage...' would get preprocessed")
    print("                by simplify_question and potentially misrouted, leading to")
    print("                answers like '15 and 45 RPS' instead of actual percentages")
    print()
    print("After removal:  Direct routing detects percentage/delta keywords and")
    print("               routes to needle agent for precise extractive answers")
    print()
    print("✅ This should significantly improve factual accuracy!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
