#!/usr/bin/env python3
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
                f.write(json.dumps(result) + '\n')
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
        print("\n" + "="*60)
        print("📊 COMBINED EVALUATION RESULTS")
        print("="*60)
        print(pretty_metrics(combined_summary))
        
        print("\n" + "="*60)
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
