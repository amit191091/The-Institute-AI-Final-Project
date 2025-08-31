#!/usr/bin/env python3
"""Simple evaluation runner to test our improvements."""

import sys
import os
sys.path.append('.')

# Enable enhanced logging for our improvements
os.environ["RAG_TRACE"] = "1"
os.environ["RAG_TRACE_EVAL"] = "1"

try:
    from app.pipeline import run_evaluation, load_docs, load_hybrid, load_llm
    
    print("🚀 Running evaluation with improved table parsing...")
    print("Loading pipeline components...")
    
    # Load the required components 
    docs = load_docs()
    hybrid = load_hybrid(docs)
    llm = load_llm()
    
    print(f"✅ Loaded {len(docs)} documents")
    print("📊 Starting evaluation...")
    
    # Run the evaluation
    run_evaluation(docs, hybrid, llm)
    
    print("✅ Evaluation complete! Check logs/ folder for detailed results.")
    
except Exception as e:
    print(f"❌ Error running evaluation: {e}")
    import traceback
    traceback.print_exc()
