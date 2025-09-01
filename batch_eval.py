#!/usr/bin/env python3
"""
Batch evaluation script - run with: python batch_eval.py <batch_number>
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def run_batch_evaluation(batch_num):
    """Run evaluation on a specific batch"""
    print(f"🚀 RUNNING EVALUATION FOR BATCH {batch_num}")
    print("="*50)
    
    # Set environment variables for faster evaluation
    os.environ.setdefault("RAG_EVAL", "1")
    os.environ.setdefault("RAG_TRACE_EVAL", "1")
    os.environ.setdefault("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")  # Faster model
    
    # Set batch-specific files
    batches_dir = Path("data") / "evaluation_batches"
    qa_file = batches_dir / f"batch_{batch_num}_questions.jsonl"
    gt_file = batches_dir / f"batch_{batch_num}_groundtruth.json"
    
    if not qa_file.exists():
        print(f"❌ Batch questions file not found: {qa_file}")
        return
    
    if not gt_file.exists():
        print(f"❌ Batch ground truth file not found: {gt_file}")
        return
    
    try:
        # Import pipeline components
        from app.pipeline import build_pipeline, _discover_input_paths, _LLM, run_evaluation
        
        print("📋 Discovering input files...")
        paths = _discover_input_paths()
        print(f"✅ Found {len(paths)} input files")
        
        print("🔧 Building pipeline...")
        docs, hybrid, debug = build_pipeline(paths)
        print(f"✅ Built pipeline with {len(docs)} documents")
        
        print("🤖 Setting up LLM...")
        llm = _LLM()
        print("✅ LLM ready")
        
        print(f"📊 Running evaluation for batch {batch_num}...")
        
        # Override the default files for this batch
        from app.pipeline import run_evaluation_with_files
        # Set batch-specific output file names
        os.environ.setdefault("RAGAS_OUTPUT_PREFIX", f"batch_{batch_num}")
        run_evaluation_with_files(docs, hybrid, llm, qa_file, gt_file)
        
        print(f"\n✅ Batch {batch_num} evaluation completed!")
        print(f"📁 Results saved to logs/")
        
    except Exception as e:
        print(f"\n❌ Batch {batch_num} evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_eval.py <batch_number>")
        print("Example: python batch_eval.py 1")
        sys.exit(1)
    
    try:
        batch_num = int(sys.argv[1])
        if batch_num < 1 or batch_num > 4:
            print("Batch number must be between 1 and 4")
            sys.exit(1)
    except ValueError:
        print("Batch number must be an integer")
        sys.exit(1)
    
    run_batch_evaluation(batch_num)
