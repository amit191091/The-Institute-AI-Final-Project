"""
Evaluation script to test RAG performance with ground truth data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add app to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from typing import Protocol, runtime_checkable, Any
from app.eval_ragas import run_eval, pretty_metrics

@runtime_checkable
class RetrievePipelineProto(Protocol):
    def query(self, question: str) -> Any: ...

def load_ground_truth(path: Path) -> List[Dict]:
    """Load ground truth dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["evaluation_dataset"]

def evaluate_rag_system(ground_truth_path: Path, pipeline: RetrievePipelineProto):
    """Evaluate RAG system against ground truth."""
    
    # Load ground truth
    gt_data = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(gt_data)} ground truth questions")
    
    # Prepare evaluation data
    questions = []
    ground_truths = []
    answers = []
    contexts = []
    
    print("\nEvaluating questions...")
    for i, item in enumerate(gt_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"{i+1}/{len(gt_data)}: {question}")
        
        try:
            # Get answer from RAG system
            result = pipeline.query(question)
            answer = result.get("answer", "")
            context_docs = result.get("contexts", [])
            context_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                           for doc in context_docs]
            
            questions.append(question)
            ground_truths.append(ground_truth)
            answers.append(answer)
            contexts.append(context_texts)
            
            print(f"   Answer: {answer[:100]}...")
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Create RAGAS dataset
    dataset = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "ground_truths": [[gt] for gt in ground_truths],  # RAGAS expects list format
    }
    
    print(f"\nRunning RAGAS evaluation on {len(questions)} questions...")
    
    try:
        metrics = run_eval(dataset)
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(pretty_metrics(metrics))
        print("="*50)
        
        # Save results
        results = {
            "metrics": metrics,
            "questions_evaluated": len(questions),
            "dataset": dataset,
            "individual_results": [
                {
                    "question": q,
                    "ground_truth": gt,
                    "answer": ans,
                    "contexts_count": len(ctx)
                }
                for q, gt, ans, ctx in zip(questions, ground_truths, answers, contexts)
            ]
        }
        
        results_path = Path("data/evaluation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_path}")
        
        return metrics
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print("This might be due to missing dependencies (ragas, datasets)")
        print("Install with: pip install ragas datasets")
        return None

if __name__ == "__main__":
    print("RAG System Evaluation")
    print("="*30)
    
    # Check if ground truth exists
    gt_path = Path("data/quick_ground_truth.json")
    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)
    
    # Initialize pipeline (would need to be configured)
    print("Note: This script requires a configured RetrievePipeline instance.")
    print("For now, it demonstrates the evaluation structure.")
    print(f"\nGround truth questions available: {len(load_ground_truth(gt_path))}")
    
    # Show sample questions
    gt_data = load_ground_truth(gt_path)
    print("\nSample questions:")
    for i, item in enumerate(gt_data[:3]):
        print(f"{i+1}. {item['question']}")
        print(f"   Expected: {item['ground_truth'][:100]}...")
        print(f"   Type: {item['type']}, Difficulty: {item['difficulty']}")
        print()
