"""
Evaluation wrapper functions for Gradio interface integration.
"""
from typing import Dict, List, Any, Optional, Protocol
import os
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval, pretty_metrics
from RAG.app.Evaluation_Analysis.ragas_evaluators import run_eval_detailed
from RAG.app.logger import get_logger

logger = get_logger()

# Check if enhanced evaluation is available
try:
    from RAG.app.Evaluation_Analysis.auto_evaluator import AutoEvaluator
    from RAG.app.Evaluation_Analysis.enhanced_question_analyzer import EnhancedQuestionAnalyzer
    EVAL_AVAILABLE = True
except Exception:
    EVAL_AVAILABLE = False


class PipelineProtocol(Protocol):
    """Protocol for RAG pipeline interface."""
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline."""
        ...


def test_google_api() -> str:
    """Test Google API connectivity."""
    try:
        import google.generativeai as genai
        
        # Check if API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Google API key not found. Set GOOGLE_API_KEY environment variable."
        
        # Test API connection
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Simple test query
        response = model.generate_content("Hello, this is a test.")
        
        if response and response.text:
            return "✅ Google API connection successful!"
        else:
            return "❌ Google API test failed - no response received"
            
    except Exception as e:
        return f"❌ Google API test failed: {str(e)}"


def test_ragas() -> str:
    """Test RAGAS evaluation framework."""
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision
        
        # Test with minimal dataset
        test_data = [{
            "question": "What is gear wear?",
            "answer": "Gear wear is the gradual removal of material from gear surfaces.",
            "contexts": ["Gear wear occurs due to friction and contact between gear teeth."]
        }]
        
        # Try to run evaluation
        result = evaluate(
            test_data,
            metrics=[answer_relevancy, context_precision]
        )
        
        if result:
            return "✅ RAGAS evaluation framework working correctly!"
        else:
            return "❌ RAGAS test failed - no results returned"
            
    except Exception as e:
        return f"❌ RAGAS test failed: {str(e)}"


def generate_ground_truth(pipeline: PipelineProtocol, num_questions: int = 5) -> str:
    """Generate ground truth questions for evaluation."""
    try:
        if not EVAL_AVAILABLE:
            return "Enhanced evaluation modules not available"
        
        # Sample questions for gear wear domain
        sample_questions = [
            "What are the main types of gear wear?",
            "How does lubrication affect gear wear?",
            "What are the common causes of gear failure?",
            "How can gear wear be measured?",
            "What are the preventive measures for gear wear?",
            "What is the relationship between vibration and gear wear?",
            "How does material hardness affect gear wear?",
            "What are the different wear patterns in gears?",
            "How does load affect gear wear rate?",
            "What are the maintenance strategies for worn gears?"
        ]
        
        # Select questions based on requested number
        selected_questions = sample_questions[:min(num_questions, len(sample_questions))]
        
        # Generate answers using pipeline
        ground_truth_data = []
        for question in selected_questions:
            try:
                result = pipeline.query(question)
                ground_truth_data.append({
                    "question": question,
                    "answer": result.get("answer", ""),
                    "contexts": result.get("contexts", [])
                })
            except Exception as e:
                logger.warning(f"Failed to generate answer for '{question}': {e}")
                continue
        
        if ground_truth_data:
            # Save to file
            import json
            from pathlib import Path
            
            output_dir = Path("RAG/logs/evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "ground_truth_dataset.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
            
            return f"✅ Generated {len(ground_truth_data)} ground truth questions. Saved to {output_file}"
        else:
            return "❌ Failed to generate any ground truth data"
            
    except Exception as e:
        return f"❌ Ground truth generation failed: {str(e)}"


def evaluate_rag(pipeline: PipelineProtocol) -> str:
    """Evaluate RAG system performance."""
    try:
        if not EVAL_AVAILABLE:
            return "Enhanced evaluation modules not available"
        
        # Load ground truth dataset
        import json
        from pathlib import Path
        
        dataset_file = Path("RAG/logs/evaluation/ground_truth_dataset.json")
        if not dataset_file.exists():
            return "❌ Ground truth dataset not found. Please generate it first."
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        
        if not ground_truth_data:
            return "❌ Ground truth dataset is empty."
        
        # Run evaluation
        evaluator = AutoEvaluator()
        results = evaluator.evaluate_dataset(ground_truth_data)
        
        if "error" in results:
            return f"❌ Evaluation failed: {results['error']}"
        
        # Generate report
        report = evaluator.generate_evaluation_report(results)
        
        # Save detailed results
        output_dir = Path("RAG/logs/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return f"✅ Evaluation completed successfully!\n\n{report}\n\nDetailed results saved to {results_file}"
        
    except Exception as e:
        return f"❌ RAG evaluation failed: {str(e)}"


def quick_evaluation(question: str, answer: str, contexts: List[str]) -> str:
    """Quick evaluation of a single Q&A pair."""
    try:
        if not EVAL_AVAILABLE:
            return "Enhanced evaluation modules not available"
        
        evaluator = AutoEvaluator()
        results = evaluator.evaluate_single_qa(question, answer, contexts)
        
        if "error" in results:
            return f"❌ Quick evaluation failed: {results['error']}"
        
        # Format results
        metrics = results.get("metrics", {})
        report_lines = ["# Quick Evaluation Results", ""]
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"**{metric_name}**: {value:.4f}")
            else:
                report_lines.append(f"**{metric_name}**: {value}")
        
        return "\n".join(report_lines)
        
    except Exception as e:
        return f"❌ Quick evaluation failed: {str(e)}"


def get_evaluation_status() -> Dict[str, Any]:
    """Get the status of evaluation components."""
    status = {
        "enhanced_eval_available": EVAL_AVAILABLE,
        "google_api_configured": bool(os.getenv("GOOGLE_API_KEY")),
        "ragas_available": False,
        "ground_truth_exists": False
    }
    
    # Check RAGAS availability
    try:
        from ragas import evaluate
        status["ragas_available"] = True
    except Exception:
        pass
    
    # Check if ground truth dataset exists
    try:
        from pathlib import Path
        dataset_file = Path("RAG/logs/evaluation/ground_truth_dataset.json")
        status["ground_truth_exists"] = dataset_file.exists()
    except Exception:
        pass
    
    return status
