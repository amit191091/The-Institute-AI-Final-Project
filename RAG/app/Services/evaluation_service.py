#!/usr/bin/env python3
"""
Evaluation Service
=================

Handles RAG system evaluation using RAGAS.
"""

import logging
from typing import Dict, List, Any

from RAG.app.logger import get_logger
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval_detailed, pretty_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for RAG system evaluation."""

    def __init__(self, llm=None):
        """
        Initialize the evaluation service.
        
        Args:
            llm: Language model instance
        """
        self.log = get_logger()
        self.llm = llm

    def evaluate_system(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the RAG system using RAGAS.
        
        Args:
            eval_data: Evaluation dataset
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            logger.info("Running RAGAS evaluation...")
            
            if not self.hybrid_retriever:
                raise ValueError("RAG system not initialized. Run pipeline first.")
            
            results = run_eval_detailed(
                eval_data,
                self.hybrid_retriever,
                self.llm
            )
            
            metrics = pretty_metrics(results)
            
            logger.info("Evaluation completed successfully")
            return {
                "results": results,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            raise

    def set_llm(self, llm):
        """Set the language model for the service."""
        self.llm = llm
