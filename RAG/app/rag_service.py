#!/usr/bin/env python3
"""
RAG Service Layer
=================

Pure business logic functions for RAG operations.
This service layer separates business logic from UI/menu code.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, UTC

from langchain.schema import Document

from RAG.app.config import settings
from RAG.app.logger import get_logger
from RAG.app.loaders import load_many
from RAG.app.chunking import structure_chunks
from RAG.app.Data_Management.metadata import attach_metadata
from RAG.app.Data_Management.indexing import (
    build_dense_index,
    build_sparse_retriever,
    to_documents,
    dump_chroma_snapshot,
)
from RAG.app.Data_Management.normalized_loader import load_normalized_docs
from RAG.app.retrieve import (
    apply_filters,
    build_hybrid_retriever,
    query_analyzer,
    rerank_candidates,
    lexical_overlap,
)
from RAG.app.Agent_Components.agents import (
    answer_needle,
    answer_summary,
    answer_table,
    route_question,
    route_question_ex,
)
from RAG.app.Evaluation_Analysis.validate import validate_min_pages
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval_detailed, pretty_metrics
from RAG.app.Services import RAGOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    """Service class for RAG operations."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the RAG service.
        
        Args:
            project_root: Path to project root directory
        """
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        
        self.log = get_logger()
        
        # Initialize orchestrator
        self.orchestrator = RAGOrchestrator(project_root)

    def run_pipeline(self, use_normalized: bool = False) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            use_normalized: Whether to use normalized documents
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        return self.orchestrator.run_pipeline(use_normalized)

    def query(self, question: str, use_agent: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            use_agent: Whether to use agent routing
            
        Returns:
            Dict[str, Any]: Query results
        """
        return self.orchestrator.query(question, use_agent)

    def evaluate_system(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the RAG system using RAGAS.
        
        Args:
            eval_data: Evaluation dataset
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        return self.orchestrator.evaluate_system(eval_data)

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dict[str, Any]: System status information
        """
        return self.orchestrator.get_system_status()

    # Properties for backward compatibility
    @property
    def docs(self):
        """Get documents from orchestrator."""
        return self.orchestrator.docs

    @property
    def hybrid_retriever(self):
        """Get hybrid retriever from orchestrator."""
        return self.orchestrator.hybrid_retriever

    @property
    def llm(self):
        """Get LLM from orchestrator."""
        return self.orchestrator.llm


# Convenience functions for backward compatibility
def run_rag_pipeline(use_normalized: bool = False) -> Dict[str, Any]:
    """Convenience function to run RAG pipeline."""
    service = RAGService()
    return service.run_pipeline(use_normalized)

def query_rag(question: str, use_agent: bool = True) -> Dict[str, Any]:
    """Convenience function to query RAG system."""
    service = RAGService()
    return service.query(question, use_agent)

def evaluate_rag(eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to evaluate RAG system."""
    service = RAGService()
    return service.evaluate_system(eval_data)

def get_rag_status() -> Dict[str, Any]:
    """Convenience function to get RAG system status."""
    service = RAGService()
    return service.get_system_status()
