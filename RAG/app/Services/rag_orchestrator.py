#!/usr/bin/env python3
"""
RAG Orchestrator
================

Main coordinator service that orchestrates all RAG operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain.schema import Document

from RAG.app.logger import get_logger
from RAG.app.config import settings
from .document_service import DocumentService
from .indexing_service import IndexingService
from .query_service import QueryService
from .evaluation_service import EvaluationService
from .embedding_service import EmbeddingService
from .llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Main orchestrator for RAG operations."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the RAG orchestrator.
        
        Args:
            project_root: Path to project root directory
        """
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        
        # Initialize services
        self.document_service = DocumentService(project_root)
        self.indexing_service = IndexingService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Initialize LLM and create dependent services
        self.llm = self.llm_service._get_llm()
        self.query_service = QueryService(self.llm)
        self.evaluation_service = EvaluationService(self.llm)
        
        # State
        self.docs: Optional[List[Document]] = None
        self.hybrid_retriever = None
        
        # Set attributes for services that need them
        self.query_service.hybrid_retriever = self.hybrid_retriever
        self.evaluation_service.hybrid_retriever = self.hybrid_retriever
        
        self.log = get_logger()

    def run_pipeline(self, use_normalized: bool = False) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            use_normalized: Whether to use normalized documents
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        try:
            logger.info("Starting RAG pipeline...")
            
            # Clean outputs
            self.document_service._clean_run_outputs()
            
            # Load documents
            docs = self.document_service.load_documents(use_normalized)
            
            # Process documents
            processed_docs = self.document_service.process_documents(docs)
            
            # Build indexes
            dense_index, sparse_retriever = self.indexing_service.build_indexes(processed_docs)
            
            # Build hybrid retriever
            hybrid_retriever = self.indexing_service.build_hybrid_retriever(dense_index, sparse_retriever)
            
            # Store for later use
            self.docs = processed_docs
            self.hybrid_retriever = hybrid_retriever
            
            # Update services with the new hybrid_retriever
            self.query_service.hybrid_retriever = hybrid_retriever
            self.query_service.docs = processed_docs
            self.evaluation_service.hybrid_retriever = hybrid_retriever
            
            result = {
                "docs": processed_docs,
                "dense_index": dense_index,
                "sparse_retriever": sparse_retriever,
                "hybrid_retriever": hybrid_retriever,
                "doc_count": len(processed_docs)
            }
            
            logger.info(f"RAG pipeline completed successfully: {len(processed_docs)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error running RAG pipeline: {str(e)}")
            raise

    def query(self, question: str, use_agent: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            use_agent: Whether to use agent routing
            
        Returns:
            Dict[str, Any]: Query results
        """
        return self.query_service.query(question, use_agent)

    def query_with_orchestrator(self, question: str, do_answer: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system using the advanced orchestrator.
        
        Args:
            question: User question
            do_answer: Whether to generate an answer
            
        Returns:
            Dict[str, Any]: Orchestrated query results with detailed trace
        """
        return self.query_service.query_with_orchestrator(question, do_answer)

    def evaluate_system(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the RAG system using RAGAS.
        
        Args:
            eval_data: Evaluation dataset
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        return self.evaluation_service.evaluate_system(eval_data)

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dict[str, Any]: System status information
        """
        try:
            status = {
                "initialized": self.hybrid_retriever is not None,
                "doc_count": len(self.docs) if self.docs else 0,
                "data_dir": str(settings.paths.DATA_DIR),
                "index_dir": str(settings.paths.INDEX_DIR),
                "logs_dir": str(settings.paths.LOGS_DIR),
                "directories": {}
            }
            
            # Check directory status
            for name, path in [
                ("data", settings.paths.DATA_DIR),
                ("index", settings.paths.INDEX_DIR),
                ("logs", settings.paths.LOGS_DIR)
            ]:
                status["directories"][name] = {
                    "exists": path.exists(),
                    "path": str(path),
                    "file_count": len(list(path.rglob("*"))) if path.exists() else 0
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"error": str(e)}
