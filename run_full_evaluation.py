#!/usr/bin/env python3
"""
Full evaluation script for all 46 questions with Answer Correctness metric
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def run_full_evaluation():
    """Run full evaluation using the existing pipeline infrastructure"""
    print("🚀 FULL EVALUATION WITH ANSWER CORRECTNESS")
    print("="*60)
    
    try:
        # Import the pipeline components
        from app.pipeline import run, build_pipeline, _discover_input_paths, _LLM
        
        # Set environment variables for evaluation
        os.environ.setdefault("RAG_EVAL", "1")
        os.environ.setdefault("RAG_TRACE_EVAL", "1")
        
        print("📋 Discovering input files...")
        paths = _discover_input_paths()
        print(f"✅ Found {len(paths)} input files")
        
        print("🔧 Building pipeline...")
        docs, hybrid, debug = build_pipeline(paths)
        print(f"✅ Built pipeline with {len(docs)} documents")
        
        print("🤖 Setting up LLM...")
        llm = _LLM()
        print("✅ LLM ready")
        
        print("📊 Running full evaluation...")
        # The run_evaluation function will automatically:
        # 1. Load all questions from data/gear_wear_qa_context_free.jsonl
        # 2. Load ground truth from data/gear_wear_QA_groundtruth_EVAL.json
        # 3. Generate answers for all questions
        # 4. Run RAGAS evaluation with Answer Correctness metric
        # 5. Save results to logs/
        
        from app.pipeline import run_evaluation
        run_evaluation(docs, hybrid, llm)
        
        print("\n✅ Full evaluation completed successfully!")
        print("\n📁 Results saved to:")
        print("   - logs/eval_ragas_per_question.jsonl (detailed per-question results)")
        print("   - logs/eval_ragas_summary.json (summary metrics)")
        
        # Display summary if available
        summary_file = Path("logs/eval_ragas_summary.json")
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print("\n" + "="*60)
            print("📊 EVALUATION SUMMARY")
            print("="*60)
            
            from app.eval_ragas import pretty_metrics, TARGETS
            print(pretty_metrics(summary))
            
            print("\n" + "="*60)
            print("🎯 TARGET THRESHOLDS")
            print("="*60)
            for metric, target in TARGETS.items():
                current = summary.get(metric)
                if current is not None:
                    status = "✅ PASS" if current >= target else "❌ FAIL"
                    print(f"{metric}: {current:.3f} (target: {target:.3f}) {status}")
                else:
                    print(f"{metric}: n/a (target: {target:.3f})")
        
    except Exception as e:
        print(f"\n❌ Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_evaluation()
