"""
Evaluation wrapper for RAG system evaluation.
This module provides a clean interface for evaluation functionality.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

# Global flag for evaluation availability
EVAL_AVAILABLE = True  # We have basic evaluation functionality


def test_google_api() -> str:
    """Test Google API setup."""
    try:
        from app.config import settings
        if settings.GOOGLE_API_KEY:
            return "✅ Google API key is configured"
        else:
            return "❌ Google API key not found"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def test_ragas() -> str:
    """Test RAGAS functionality."""
    try:
        from app.eval_ragas import run_eval, pretty_metrics
        return "✅ RAGAS evaluation functions are available"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def generate_ground_truth(pipeline, num_questions: int) -> str:
    """Generate ground truth dataset."""
    try:
        # Check if ground truth dataset already exists
        gt_path = Path("data/ground_truth_dataset.json")
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                data = json.load(f)
                existing_questions = len(data.get('questions', []))
            return f"✅ Ground truth dataset already exists with {existing_questions} questions"
        else:
            return "❌ Ground truth generation requires additional setup"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def evaluate_rag(pipeline) -> str:
    """Evaluate RAG system."""
    try:
        gt_path = Path("data/ground_truth_dataset.json")
        if not gt_path.exists():
            return "❌ Ground truth dataset not found. Generate it first."
        
        # Load ground truth dataset
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        questions = gt_data.get('questions', [])
        if not questions:
            return "❌ No questions found in ground truth dataset"
        
        # Use the existing evaluation functionality
        from app.eval_ragas import run_eval, pretty_metrics
        
        # Run evaluation on a sample question
        sample_question = questions[0]
        dataset = {
            "question": [sample_question["question"]],
            "answer": ["Sample answer"],  # This would be replaced with actual RAG answer
            "ground_truth": [sample_question["ground_truth"]],
            "contexts": [["Sample context"]]  # This would be replaced with actual contexts
        }
        
        metrics = run_eval(dataset)
        formatted_metrics = pretty_metrics(metrics)
        
        return f"✅ RAG evaluation functions are available\n\nSample evaluation:\n{formatted_metrics}\n\nUse the web UI for full evaluation."
    except Exception as e:
        return f"❌ Error: {str(e)}"
