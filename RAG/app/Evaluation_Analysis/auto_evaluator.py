"""
AutoEvaluator: Automatic evaluation of RAG system performance.
"""
from typing import Dict, List, Any, Optional
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval, pretty_metrics
from RAG.app.Evaluation_Analysis.ragas_evaluators import run_eval_detailed
from RAG.app.logger import get_logger

logger = get_logger()


class AutoEvaluator:
    """Automatic evaluator for RAG system performance."""
    
    def __init__(self, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings
        self.logger = get_logger()
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a dataset using RAGAS metrics."""
        try:
            if not dataset:
                return {"error": "Empty dataset provided"}
            
            # Run RAGAS evaluation
            results = run_eval_detailed(dataset, self.llm, self.embeddings)
            
            # Format results
            formatted_results = pretty_metrics(results)
            
            return {
                "success": True,
                "metrics": results,
                "formatted": formatted_results
            }
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def evaluate_single_qa(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single Q&A pair."""
        try:
            qa_pair = {
                "question": question,
                "answer": answer,
                "contexts": contexts
            }
            
            if ground_truth:
                qa_pair["ground_truth"] = ground_truth
            
            # Run evaluation
            results = run_eval([qa_pair], self.llm, self.embeddings)
            
            return {
                "success": True,
                "metrics": results
            }
        except Exception as e:
            self.logger.error(f"Single QA evaluation failed: {e}")
            return {"error": str(e)}
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        try:
            if "error" in results:
                return f"Evaluation Error: {results['error']}"
            
            if "formatted" in results:
                return results["formatted"]
            
            # Fallback formatting
            metrics = results.get("metrics", {})
            report_lines = ["# RAG System Evaluation Report", ""]
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"**{metric_name}**: {value:.4f}")
                else:
                    report_lines.append(f"**{metric_name}**: {value}")
            
            return "\n".join(report_lines)
        except Exception as e:
            return f"Report generation failed: {e}"
