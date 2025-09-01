#!/usr/bin/env python3
"""
Split evaluation into 4 batches to avoid LLM timeout issues
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def split_questions_into_batches():
    """Split the 46 questions into 4 batches of ~12 questions each"""
    print("📦 SPLITTING EVALUATION INTO 4 BATCHES")
    print("="*50)
    
    # Load the original questions
    qa_file = Path("data") / "gear_wear_qa_context_free.jsonl"
    gt_file = Path("data") / "gear_wear_QA_groundtruth_EVAL.json"
    
    if not qa_file.exists():
        print(f"❌ Questions file not found: {qa_file}")
        return
    
    if not gt_file.exists():
        print(f"❌ Ground truth file not found: {gt_file}")
        return
    
    # Load questions
    questions = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    # Load ground truth
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"✅ Loaded {len(questions)} questions")
    print(f"✅ Loaded {len(ground_truth)} ground truth entries")
    
    # Create batches directory
    batches_dir = Path("data") / "evaluation_batches"
    batches_dir.mkdir(exist_ok=True)
    
    # Split into 4 batches
    batch_size = len(questions) // 4
    remainder = len(questions) % 4
    
    batches = []
    start_idx = 0
    
    for i in range(4):
        # Add one extra question to first 'remainder' batches
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        batch_questions = questions[start_idx:end_idx]
        batches.append(batch_questions)
        
        print(f"   Batch {i+1}: {len(batch_questions)} questions (indices {start_idx}-{end_idx-1})")
        start_idx = end_idx
    
    # Create batch files
    for i, batch_questions in enumerate(batches):
        batch_num = i + 1
        
        # Create questions file for this batch
        qa_batch_file = batches_dir / f"batch_{batch_num}_questions.jsonl"
        with open(qa_batch_file, 'w', encoding='utf-8') as f:
            for q in batch_questions:
                f.write(json.dumps(q) + '\n')
        
        # Create ground truth file for this batch
        gt_batch_file = batches_dir / f"batch_{batch_num}_groundtruth.json"
        batch_gt = []
        
        for q in batch_questions:
            question_text = q.get('question', '')
            # Find matching ground truth
            for gt_entry in ground_truth:
                if gt_entry.get('question', '') == question_text:
                    batch_gt.append(gt_entry)
                    break
        
        with open(gt_batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_gt, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created batch {batch_num}: {len(batch_questions)} questions, {len(batch_gt)} ground truths")
    
    print(f"\n📁 Batch files created in: {batches_dir}")
    return batches_dir

def create_batch_evaluation_script():
    """Create a script to run evaluation on a specific batch"""
    script_content = '''#!/usr/bin/env python3
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
        
        print(f"\\n✅ Batch {batch_num} evaluation completed!")
        print(f"📁 Results saved to logs/")
        
    except Exception as e:
        print(f"\\n❌ Batch {batch_num} evaluation failed: {e}")
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
'''
    
    with open("batch_eval.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created batch_eval.py script")

def create_combined_results_script():
    """Create a script to combine results from all batches"""
    script_content = '''#!/usr/bin/env python3
"""
Combine results from all 4 evaluation batches
"""

import json
from pathlib import Path

def combine_batch_results():
    """Combine results from all batches into final summary"""
    print("🔗 COMBINING BATCH RESULTS")
    print("="*50)
    
    all_per_question = []
    all_summaries = []
    
    # Load results from each batch
    for batch_num in range(1, 5):
        per_question_file = Path("logs") / f"eval_ragas_per_question_batch_{batch_num}.jsonl"
        summary_file = Path("logs") / f"eval_ragas_summary_batch_{batch_num}.json"
        
        if per_question_file.exists():
            with open(per_question_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_per_question.append(json.loads(line))
            print(f"✅ Loaded {per_question_file}")
        
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                all_summaries.append(summary)
            print(f"✅ Loaded {summary_file}")
    
    # Combine per-question results
    if all_per_question:
        combined_per_question_file = Path("logs") / "eval_ragas_per_question_combined.jsonl"
        with open(combined_per_question_file, 'w', encoding='utf-8') as f:
            for result in all_per_question:
                f.write(json.dumps(result) + '\\n')
        print(f"✅ Combined per-question results: {len(all_per_question)} questions")
    
    # Calculate combined summary
    if all_summaries:
        combined_summary = {}
        metrics = ['context_precision', 'context_recall', 'faithfulness', 'table_qa_accuracy', 'answer_correctness']
        
        for metric in metrics:
            values = [s.get(metric) for s in all_summaries if s.get(metric) is not None]
            if values:
                combined_summary[metric] = sum(values) / len(values)
        
        combined_summary_file = Path("logs") / "eval_ragas_summary_combined.json"
        with open(combined_summary_file, 'w', encoding='utf-8') as f:
            json.dump(combined_summary, f, indent=2)
        
        print(f"✅ Combined summary metrics: {list(combined_summary.keys())}")
        
        # Display combined results
        from app.eval_ragas import pretty_metrics, TARGETS
        print("\\n" + "="*60)
        print("📊 COMBINED EVALUATION RESULTS")
        print("="*60)
        print(pretty_metrics(combined_summary))
        
        print("\\n" + "="*60)
        print("🎯 TARGET THRESHOLDS")
        print("="*60)
        for metric, target in TARGETS.items():
            current = combined_summary.get(metric)
            if current is not None:
                status = "✅ PASS" if current >= target else "❌ FAIL"
                print(f"{metric}: {current:.3f} (target: {target:.3f}) {status}")
            else:
                print(f"{metric}: n/a (target: {target:.3f})")

if __name__ == "__main__":
    combine_batch_results()
'''
    
    with open("combine_results.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created combine_results.py script")

def main():
    """Main function to set up batch evaluation"""
    print("🚀 SETTING UP BATCH EVALUATION")
    print("="*60)
    
    # Split questions into batches
    batches_dir = split_questions_into_batches()
    
    # Create evaluation scripts
    create_batch_evaluation_script()
    create_combined_results_script()
    
    print("\n" + "="*60)
    print("✅ BATCH EVALUATION SETUP COMPLETE")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Run batch 1: python batch_eval.py 1")
    print("2. Run batch 2: python batch_eval.py 2")
    print("3. Run batch 3: python batch_eval.py 3")
    print("4. Run batch 4: python batch_eval.py 4")
    print("5. Combine results: python combine_results.py")
    print("\n💡 TIP: Run batches in separate terminals to avoid conflicts")

if __name__ == "__main__":
    main()
