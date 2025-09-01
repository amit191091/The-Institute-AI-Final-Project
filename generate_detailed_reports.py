#!/usr/bin/env python3
"""
Generate detailed reports from existing batch evaluation results
Run with: python generate_detailed_reports.py <batch_number>
"""

import sys
import json
import math
from pathlib import Path
from datetime import datetime

def generate_detailed_report(batch_num):
    """Generate detailed report from existing batch results"""
    print(f"📊 GENERATING DETAILED REPORT FOR BATCH {batch_num}")
    print("="*50)
    
    # Check if results exist
    per_q_path = Path("logs") / "eval_ragas_per_question.jsonl"
    summary_path = Path("logs") / "eval_ragas_summary.json"
    
    if not per_q_path.exists():
        print(f"❌ Per-question results not found: {per_q_path}")
        print("Please run batch evaluation first: python batch_eval.py {batch_num}")
        return
    
    if not summary_path.exists():
        print(f"❌ Summary results not found: {summary_path}")
        print("Please run batch evaluation first: python batch_eval.py {batch_num}")
        return
    
    # Create detailed results structure
    detailed_results = {
        "batch_number": batch_num,
        "timestamp": datetime.now().isoformat(),
        "questions": [],
        "summary": {},
        "total_scores": {}
    }
    
    # Read summary
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            detailed_results["summary"] = json.load(f)
        print("✅ Loaded summary metrics")
    except Exception as e:
        print(f"❌ Error loading summary: {e}")
        return
    
    # Read per-question results
    try:
        with open(per_q_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
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
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: JSON decode error on line {line_num}: {e}")
                        continue
        
        print(f"✅ Loaded {len(detailed_results['questions'])} questions")
    except Exception as e:
        print(f"❌ Error loading per-question results: {e}")
        return
    
    # Calculate total scores
    if detailed_results["questions"]:
        total_scores = calculate_total_scores(detailed_results["questions"])
        detailed_results["total_scores"] = total_scores
        print("✅ Calculated total scores")
    
    # Save detailed results
    save_detailed_results(batch_num, detailed_results)
    
    print(f"\n✅ Detailed report for batch {batch_num} completed!")
    print(f"📁 Results saved to logs/batch_{batch_num}_detailed_results.json")
    print(f"📊 Summary saved to logs/batch_{batch_num}_summary.json")

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
    
    # Print individual question scores
    print(f"\n📝 INDIVIDUAL QUESTION SCORES:")
    print("="*40)
    for i, q in enumerate(detailed_results["questions"], 1):
        print(f"\nQ{i}: {q['question'][:80]}...")
        print(f"Answer: {q['answer'][:100]}...")
        print(f"Ground Truth: {q['ground_truth']}")
        
        # Show key scores
        key_scores = ["faithfulness", "answer_relevancy", "answer_correctness", "factual_score"]
        for score_key in key_scores:
            score_value = q["scores"].get(score_key)
            if score_value is not None and not (isinstance(score_value, float) and math.isnan(score_value)):
                print(f"  {score_key}: {score_value:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_detailed_reports.py <batch_number>")
        print("Example: python generate_detailed_reports.py 3")
        sys.exit(1)
    
    try:
        batch_num = int(sys.argv[1])
        if batch_num < 1 or batch_num > 4:
            print("Batch number must be between 1 and 4")
            sys.exit(1)
    except ValueError:
        print("Batch number must be an integer")
        sys.exit(1)
    
    generate_detailed_report(batch_num)
