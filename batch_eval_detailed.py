#!/usr/bin/env python3
"""
Detailed batch evaluation script - run with: python batch_eval_detailed.py <batch_number>
Saves results with different JSON files and includes detailed scoring
"""

import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def run_detailed_batch_evaluation(batch_num):
    """Run evaluation on a specific batch with detailed scoring"""
    print(f"🚀 RUNNING DETAILED EVALUATION FOR BATCH {batch_num}")
    print("="*60)
    
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
        from app.pipeline import build_pipeline, _discover_input_paths, _LLM, run_evaluation_with_files
        
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
        
        # Run evaluation and capture results
        results = run_evaluation_with_files(docs, hybrid, llm, qa_file, gt_file)
        
        # Create detailed results structure
        detailed_results = create_detailed_results(batch_num, results)
        
        # Save detailed results
        save_detailed_results(batch_num, detailed_results)
        
        print(f"\n✅ Batch {batch_num} detailed evaluation completed!")
        print(f"📁 Results saved to logs/batch_{batch_num}_detailed_results.json")
        print(f"📊 Summary saved to logs/batch_{batch_num}_summary.json")
        
    except Exception as e:
        print(f"\n❌ Batch {batch_num} evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def create_detailed_results(batch_num, results):
    """Create detailed results structure with scoring breakdown"""
    # Read the per-question results
    per_q_path = Path("logs") / "eval_ragas_per_question.jsonl"
    summary_path = Path("logs") / "eval_ragas_summary.json"
    
    detailed_results = {
        "batch_number": batch_num,
        "timestamp": datetime.now().isoformat(),
        "questions": [],
        "summary": {},
        "total_scores": {}
    }
    
    # Read summary
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            detailed_results["summary"] = json.load(f)
    
    # Read per-question results
    if per_q_path.exists():
        with open(per_q_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "__summary__" not in data:  # Skip summary footer
                            question_data = {
                                "question": data.get("question", ""),
                                "answer": data.get("answer", ""),
                                "ground_truth": data.get("ground_truths", []),
                                "reference": data.get("reference", ""),
                                "scores": {
                                    "faithfulness": data.get("faithfulness"),
                                    "answer_relevancy": data.get("answer_relevancy"),
                                    "context_precision": data.get("context_precision"),
                                    "context_recall": data.get("context_recall"),
                                    "answer_correctness": data.get("answer_correctness"),
                                    "factual_em": data.get("factual_em"),
                                    "factual_token_f1": data.get("factual_token_f1"),
                                    "factual_numeric": data.get("factual_numeric"),
                                    "factual_range": data.get("factual_range"),
                                    "factual_list_f1": data.get("factual_list_f1"),
                                    "factual_score": data.get("factual_score"),
                                    "overlap_precision": data.get("overlap_precision"),
                                    "overlap_recall": data.get("overlap_recall"),
                                    "overlap_f1": data.get("overlap_f1"),
                                    "table_like": data.get("table_like"),
                                    "table_correct": data.get("table_correct")
                                },
                                "reasoning_trace": data.get("reasoning_trace", {})
                            }
                            detailed_results["questions"].append(question_data)
                    except json.JSONDecodeError:
                        continue
    
    # Calculate total scores
    if detailed_results["questions"]:
        total_scores = calculate_total_scores(detailed_results["questions"])
        detailed_results["total_scores"] = total_scores
    
    return detailed_results

def calculate_total_scores(questions):
    """Calculate total scores across all questions"""
    total_scores = {}
    
    # Get all score keys from the first question
    if questions:
        score_keys = list(questions[0]["scores"].keys())
        
        for key in score_keys:
            values = []
            for q in questions:
                value = q["scores"].get(key)
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    values.append(value)
            
            if values:
                total_scores[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "total_questions": len(questions)
                }
            else:
                total_scores[key] = {
                    "mean": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                    "total_questions": len(questions)
                }
    
    return total_scores

def save_detailed_results(batch_num, detailed_results):
    """Save detailed results to separate files"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    detailed_file = logs_dir / f"batch_{batch_num}_detailed_results.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2, allow_nan=False)
    
    # Save summary only
    summary_file = logs_dir / f"batch_{batch_num}_summary.json"
    summary_data = {
        "batch_number": batch_num,
        "timestamp": detailed_results["timestamp"],
        "total_questions": len(detailed_results["questions"]),
        "summary_metrics": detailed_results["summary"],
        "total_scores": detailed_results["total_scores"]
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2, allow_nan=False)
    
    # Print summary to console
    print(f"\n📊 BATCH {batch_num} SUMMARY:")
    print("="*40)
    print(f"Total Questions: {len(detailed_results['questions'])}")
    
    if detailed_results["total_scores"]:
        print("\n📈 SCORE BREAKDOWN:")
        for metric, scores in detailed_results["total_scores"].items():
            if scores["mean"] is not None:
                print(f"  {metric}: {scores['mean']:.3f} (min: {scores['min']:.3f}, max: {scores['max']:.3f})")
            else:
                print(f"  {metric}: No valid scores")
    
    if detailed_results["summary"]:
        print("\n🎯 OVERALL METRICS:")
        for key, value in detailed_results["summary"].items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_eval_detailed.py <batch_number>")
        print("Example: python batch_eval_detailed.py 1")
        sys.exit(1)
    
    try:
        batch_num = int(sys.argv[1])
        if batch_num < 1 or batch_num > 4:
            print("Batch number must be between 1 and 4")
            sys.exit(1)
    except ValueError:
        print("Batch number must be an integer")
        sys.exit(1)
    
    run_detailed_batch_evaluation(batch_num)
