#!/usr/bin/env python3
"""
Query Service
============

Handles query processing and agent routing.
"""

import logging
from typing import Dict, List, Any

from RAG.app.logger import get_logger
from RAG.app.Agent_Components.agents import (
    answer_needle,
    answer_summary,
    answer_table,
    route_question,
    route_question_ex,
)
from RAG.app.agent_orchestrator import run as run_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:
    """Service for query processing and agent routing."""

    def __init__(self, llm=None):
        """
        Initialize the query service.
        
        Args:
            llm: Language model instance
        """
        self.log = get_logger()
        self.llm = llm
        self.hybrid_retriever = None
        self.docs = None

    def query(self, question: str, use_agent: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            use_agent: Whether to use agent routing
            
        Returns:
            Dict[str, Any]: Query results
        """
        try:
            if not self.hybrid_retriever:
                raise ValueError("RAG system not initialized. Run pipeline first.")
            
            logger.info(f"Processing query: {question}")
            
            if use_agent:
                # Use agent routing
                route, trace = route_question_ex(question)
                
                # Get relevant documents
                docs = self.hybrid_retriever.get_relevant_documents(question)
                
                # Route to appropriate agent
                if route == "summary":
                    answer = answer_summary(self.llm, docs, question)
                elif route == "table":
                    answer = answer_table(self.llm, docs, question)
                else:  # needle
                    answer = answer_needle(self.llm, docs, question)
                
                result = {
                    "answer": answer,
                    "sources": docs,
                    "method": f"agent_routing_{route}",
                    "route": route,
                    "trace": trace
                }
            else:
                # Simple retrieval
                docs = self.hybrid_retriever.get_relevant_documents(question)
                result = {
                    "answer": "Direct retrieval mode - no agent processing",
                    "sources": docs,
                    "method": "direct_retrieval"
                }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def set_llm(self, llm):
        """Set the language model for the service."""
        self.llm = llm

    def query_with_orchestrator(self, question: str, do_answer: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system using the advanced orchestrator.
        
        Args:
            question: User question
            do_answer: Whether to generate an answer
            
        Returns:
            Dict[str, Any]: Orchestrated query results with detailed trace
        """
        try:
            if not self.hybrid_retriever:
                raise ValueError("RAG system not initialized. Run pipeline first.")
            
            logger.info(f"Processing query with orchestrator: {question}")
            
            # Use the orchestrator for advanced processing
            result = run_orchestrator(
                question=question,
                docs=self.docs or [],
                hybrid=self.hybrid_retriever,
                llm_callable=self.llm,
                do_answer=do_answer
            )
            
            # Add metadata about the method used
            result["method"] = "agent_orchestrator"
            result["orchestrator_version"] = "1.0"
            
            logger.info("Orchestrated query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing orchestrated query: {str(e)}")
            raise
