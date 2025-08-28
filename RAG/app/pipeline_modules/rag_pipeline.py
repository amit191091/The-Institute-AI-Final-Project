"""
RAG Pipeline Wrapper
===================

This module provides a backward-compatible wrapper around the ScalableRAGPipeline.
The ScalableRAGPipeline is the main implementation with enhanced features.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document

from RAG.app.interfaces import RAGPipelineInterface
from RAG.app.pipeline_modules.integrated_rag_pipeline import IntegratedRAGPipeline
from RAG.app.config import settings
from RAG.app.logger import get_logger


class RAGPipeline(RAGPipelineInterface):
    """Backward-compatible wrapper around ScalableRAGPipeline."""
    
    def __init__(self, config=None):
        self.config = config or settings
        self.logger = get_logger()
        
        # Use ScalableRAGPipeline as the main implementation
        self._scalable_pipeline = ScalableRAGPipeline(config)
        
        # Pipeline state
        self.documents: List[Document] = []
        self.is_initialized = False
    
    # Property accessors for backward compatibility
    @property
    def document_ingestion(self):
        """Access document ingestion service for backward compatibility."""
        return self._scalable_pipeline.document_ingestion
    
    @property
    def query_processing(self):
        """Access query processing service for backward compatibility."""
        return self._scalable_pipeline.query_processing
    
    @property
    def retrieval(self):
        """Access retrieval service for backward compatibility."""
        return self._scalable_pipeline.retrieval
    
    @property
    def answer_generation(self):
        """Access answer generation service for backward compatibility."""
        return self._scalable_pipeline.answer_generation
    
    @property
    def configuration(self):
        """Access configuration service for backward compatibility."""
        return self._scalable_pipeline.configuration
    
    @property
    def logging(self):
        """Access logging service for backward compatibility."""
        return self._scalable_pipeline.logging
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline."""
        return self._scalable_pipeline.process_query(query)
    
    def add_documents(self, file_paths: List[str]) -> bool:
        """Add new documents to the RAG system."""
        return self._scalable_pipeline.add_documents(file_paths)
    
    def initialize_pipeline(self) -> bool:
        """Initialize the RAG pipeline."""
        return self._scalable_pipeline.initialize_pipeline()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline state."""
        return self._scalable_pipeline.get_pipeline_info()
    
    def clear_cache(self) -> bool:
        """Clear the pipeline cache."""
        return self._scalable_pipeline.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._scalable_pipeline.get_cache_stats()


# Factory functions for backward compatibility
def create_rag_pipeline(config=None) -> RAGPipeline:
    """Create a new RAG pipeline instance."""
    return RAGPipeline(config)


def process_query_with_rag_pipeline(query: str, config=None) -> Dict[str, Any]:
    """Process a query using a RAG pipeline instance."""
    pipeline = create_rag_pipeline(config)
    return pipeline.process_query(query)
