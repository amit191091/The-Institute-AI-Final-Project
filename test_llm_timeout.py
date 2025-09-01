#!/usr/bin/env python3
"""
Test script to identify LLM timeout issues during evaluation
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def test_llm_timeout():
    """Test if LLM is timing out during evaluation"""
    print("🔍 TESTING LLM TIMEOUT ISSUES")
    print("="*50)
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    
    print(f"✅ OpenAI API Key: {'Set' if openai_key else 'NOT SET'}")
    print(f"✅ OpenAI Model: {openai_model}")
    
    if not openai_key:
        print("❌ No OpenAI API key found! This explains the timeout issues.")
        return
    
    # Test a single LLM call with timeout
    try:
        from app.pipeline import _LLM
        
        print("\n🤖 Testing single LLM call...")
        llm = _LLM()
        
        # Test with a simple question
        test_question = "What is the sampling rate?"
        test_context = "The sampling rate was 50 kHz according to the specifications."
        
        start_time = time.time()
        
        # This should trigger an LLM call
        response = llm.answer_with_contexts(test_question, [test_context])
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ LLM Response: {response}")
        print(f"✅ Response Time: {duration:.2f} seconds")
        
        if duration > 30:
            print("⚠️  SLOW RESPONSE: LLM call took more than 30 seconds")
        elif duration > 60:
            print("🚨 TIMEOUT ISSUE: LLM call took more than 60 seconds")
        else:
            print("✅ NORMAL RESPONSE TIME")
            
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        import traceback
        traceback.print_exc()

def check_evaluation_timeout():
    """Check if evaluation is configured with proper timeouts"""
    print("\n📊 CHECKING EVALUATION TIMEOUT CONFIGURATION")
    print("="*50)
    
    # Check current timeout settings
    timeout_settings = {
        "RAG_EVAL": os.getenv("RAG_EVAL", "0"),
        "RAG_TRACE_EVAL": os.getenv("RAG_TRACE_EVAL", "0"),
        "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "RAG_CONTEXT_TOP_N": os.getenv("RAG_CONTEXT_TOP_N", "10"),
        "RAG_DENSE_K": os.getenv("RAG_DENSE_K", "10"),
        "RAG_SPARSE_K": os.getenv("RAG_SPARSE_K", "10"),
    }
    
    for key, value in timeout_settings.items():
        print(f"   {key}: {value}")
    
    # Estimate evaluation time
    questions_count = 46
    avg_time_per_question = 30  # seconds (conservative estimate)
    total_estimated_time = questions_count * avg_time_per_question
    
    print(f"\n⏱️  ESTIMATED EVALUATION TIME:")
    print(f"   Questions: {questions_count}")
    print(f"   Avg time per question: {avg_time_per_question} seconds")
    print(f"   Total estimated time: {total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
    
    if total_estimated_time > 1800:  # 30 minutes
        print("🚨 VERY LONG EVALUATION: Consider using faster model or batch processing")

def suggest_fixes():
    """Suggest fixes for timeout issues"""
    print("\n🔧 SUGGESTED FIXES FOR TIMEOUT ISSUES")
    print("="*50)
    
    print("1. 🚀 USE FASTER MODEL:")
    print("   Set OPENAI_CHAT_MODEL=gpt-3.5-turbo in .env file")
    print("   This is much faster than gpt-4o-mini for evaluation")
    
    print("\n2. ⏱️  ADD TIMEOUT SETTINGS:")
    print("   Add to .env file:")
    print("   OPENAI_REQUEST_TIMEOUT=60")
    print("   OPENAI_MAX_RETRIES=2")
    
    print("\n3. 📦 BATCH PROCESSING:")
    print("   Process questions in smaller batches (e.g., 10 questions at a time)")
    print("   This prevents long-running evaluations from timing out")
    
    print("\n4. 🔄 RETRY MECHANISM:")
    print("   Implement retry logic for failed LLM calls")
    print("   Use exponential backoff for rate limits")
    
    print("\n5. 💾 CACHE RESPONSES:")
    print("   Cache LLM responses to avoid repeated calls")
    print("   Save intermediate results during evaluation")

if __name__ == "__main__":
    test_llm_timeout()
    check_evaluation_timeout()
    suggest_fixes()
