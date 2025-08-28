"""
Tests for the scalable RAG pipeline with caching, batching, and progress tracking.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os

from RAG.app.pipeline_modules.integrated_rag_pipeline import (
    IntegratedRAGPipeline,
    IntegratedRAGPipelineFactory,
    create_integrated_rag_pipeline,
    process_query_with_integrated_pipeline
)
from RAG.app.interfaces import RAGPipelineInterface
from RAG.app.config import settings
from langchain.schema import Document


class TestScalableRAGPipeline(unittest.TestCase):
    """Test cases for the scalable RAG pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the service locator to avoid actual service initialization
        self.mock_services = {
            'document_ingestion': Mock(),
            'query_processing': Mock(),
            'retrieval': Mock(),
            'answer_generation': Mock(),
            'configuration': Mock(),
            'logging': Mock()
        }
        
        # Mock the dependency injection
        with patch('RAG.app.pipeline_modules.integrated_rag_pipeline.RAGServiceLocator') as mock_locator:
            mock_locator.get_document_ingestion.return_value = self.mock_services['document_ingestion']
            mock_locator.get_query_processing.return_value = self.mock_services['query_processing']
            mock_locator.get_retrieval.return_value = self.mock_services['retrieval']
            mock_locator.get_answer_generation.return_value = self.mock_services['answer_generation']
            mock_locator.get_configuration.return_value = self.mock_services['configuration']
            mock_locator.get_logging.return_value = self.mock_services['logging']
            
            self.pipeline = IntegratedRAGPipeline()
    
    def test_pipeline_initialization(self):
        """Test that the pipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, RAGPipelineInterface)
        self.assertIsInstance(self.pipeline, IntegratedRAGPipeline)
        self.assertEqual(len(self.pipeline.documents), 0)
        self.assertFalse(self.pipeline.is_initialized)
    
    def test_process_query_success(self):
        """Test successful query processing."""
        # Mock the query processing steps
        self.mock_services['query_processing'].analyze_query.return_value = {
            'keywords': ['test', 'query'],
            'intent': 'information_retrieval'
        }
        
        mock_docs = [Document(page_content="Test document", metadata={})]
        self.mock_services['retrieval'].retrieve_documents.return_value = mock_docs
        self.mock_services['query_processing'].apply_filters.return_value = mock_docs
        self.mock_services['retrieval'].rerank_documents.return_value = mock_docs
        self.mock_services['answer_generation'].generate_answer.return_value = "Test answer"
        self.mock_services['answer_generation'].format_answer.return_value = {
            "answer": "Test answer",
            "sources": [],
            "num_sources": 0,
            "confidence": "high"
        }
        
        # Process query
        result = self.pipeline.process_query("What is the test query?")
        
        # Verify result
        self.assertIn("answer", result)
        self.assertIn("query", result)
        self.assertIn("query_analysis", result)
        self.assertIn("pipeline_version", result)
        self.assertEqual(result["answer"], "Test answer")
        self.assertEqual(result["query"], "What is the test query?")
        self.assertEqual(result["pipeline_version"], "scalable_v1.0")
    
    def test_process_query_error(self):
        """Test query processing with error handling."""
        # Mock an error in query processing
        self.mock_services['query_processing'].analyze_query.side_effect = Exception("Test error")
        
        # Process query
        result = self.pipeline.process_query("What is the test query?")
        
        # Verify error handling
        self.assertIn("answer", result)
        self.assertIn("error", result)
        self.assertIn("I encountered an error", result["answer"])
        self.assertEqual(result["error"], "Test error")
    
    def test_add_documents_success(self):
        """Test successful document addition."""
        # Mock document ingestion
        mock_docs = [Document(page_content="Test content", metadata={})]
        self.mock_services['document_ingestion'].ingest_documents.return_value = mock_docs
        self.mock_services['document_ingestion'].validate_documents.return_value = True
        self.mock_services['document_ingestion'].preprocess_documents.return_value = mock_docs
        
        # Add documents
        result = self.pipeline.add_documents(["test_file.pdf"])
        
        # Verify success
        self.assertTrue(result)
        self.assertEqual(len(self.pipeline.documents), 1)
        self.assertTrue(self.pipeline.is_initialized)
    
    def test_add_documents_validation_failure(self):
        """Test document addition with validation failure."""
        # Mock document ingestion with validation failure
        mock_docs = [Document(page_content="Test content", metadata={})]
        self.mock_services['document_ingestion'].ingest_documents.return_value = mock_docs
        self.mock_services['document_ingestion'].validate_documents.return_value = False
        
        # Add documents
        result = self.pipeline.add_documents(["test_file.pdf"])
        
        # Verify failure
        self.assertFalse(result)
        self.assertEqual(len(self.pipeline.documents), 0)
        self.assertFalse(self.pipeline.is_initialized)
    
    def test_add_documents_batched(self):
        """Test batched document addition."""
        # Mock document ingestion - return different documents for each batch
        def mock_ingest_documents(file_paths):
            # Return one document per file in the batch
            return [Document(page_content=f"Test content for {path}", metadata={}) for path in file_paths]
        
        self.mock_services['document_ingestion'].ingest_documents.side_effect = mock_ingest_documents
        self.mock_services['document_ingestion'].validate_documents.return_value = True
        self.mock_services['document_ingestion'].preprocess_documents.side_effect = lambda docs: docs  # Return same docs
        
        # Add documents in batches
        file_paths = [f"test_file_{i}.pdf" for i in range(25)]  # 25 files
        result = self.pipeline.add_documents_batched(file_paths, batch_size=10)
        
        # Verify success
        self.assertTrue(result)
        self.assertEqual(len(self.pipeline.documents), 25)  # 25 documents
        self.assertTrue(self.pipeline.is_initialized)
    
    def test_initialize_from_data_directory(self):
        """Test pipeline initialization from data directory."""
        # Mock document discovery and ingestion
        self.mock_services['document_ingestion'].discover_input_paths.return_value = ["test_file.pdf"]
        mock_docs = [Document(page_content="Test content", metadata={})]
        self.mock_services['document_ingestion'].ingest_documents.return_value = mock_docs
        self.mock_services['document_ingestion'].validate_documents.return_value = True
        self.mock_services['document_ingestion'].preprocess_documents.return_value = mock_docs
        
        # Initialize pipeline
        result = self.pipeline.initialize_from_data_directory()
        
        # Verify success
        self.assertTrue(result)
        self.assertEqual(len(self.pipeline.documents), 1)
        self.assertTrue(self.pipeline.is_initialized)
    
    def test_initialize_from_empty_data_directory(self):
        """Test pipeline initialization from empty data directory."""
        # Mock empty document discovery
        self.mock_services['document_ingestion'].discover_input_paths.return_value = []
        
        # Initialize pipeline
        result = self.pipeline.initialize_from_data_directory()
        
        # Verify failure
        self.assertFalse(result)
        self.assertEqual(len(self.pipeline.documents), 0)
        self.assertFalse(self.pipeline.is_initialized)
    
    def test_get_pipeline_status(self):
        """Test getting pipeline status."""
        # Add some documents first
        self.pipeline.documents = [Document(page_content="Test", metadata={})]
        self.pipeline.is_initialized = True
        
        # Get status
        status = self.pipeline.get_pipeline_status()
        
        # Verify status
        self.assertIn("is_initialized", status)
        self.assertIn("num_documents", status)
        self.assertIn("config", status)
        self.assertIn("services", status)
        self.assertIn("scalability", status)
        
        self.assertTrue(status["is_initialized"])
        self.assertEqual(status["num_documents"], 1)
        self.assertIn("embedding_model", status["config"])
        self.assertIn("cache_stats", status["scalability"])
    
    def test_get_document_summary(self):
        """Test getting document summary."""
        # Add documents with metadata
        docs = [
            Document(page_content="Content 1", metadata={"file_name": "test1.pdf", "page": 1, "section": "intro"}),
            Document(page_content="Content 2", metadata={"file_name": "test2.pdf", "page": 2, "section": "main"}),
            Document(page_content="Content 3", metadata={"file_name": "test3.txt", "page": 1, "section": "intro"})
        ]
        self.pipeline.documents = docs
        
        # Get summary
        summary = self.pipeline.get_document_summary()
        
        # Verify summary
        self.assertIn("total_documents", summary)
        self.assertIn("file_types", summary)
        self.assertIn("total_pages", summary)
        self.assertIn("sections", summary)
        
        self.assertEqual(summary["total_documents"], 3)
        self.assertEqual(summary["file_types"]["pdf"], 2)
        self.assertEqual(summary["file_types"]["txt"], 1)
        self.assertEqual(summary["total_pages"], 4)  # 1+2+1
        self.assertEqual(summary["sections"]["intro"], 2)
        self.assertEqual(summary["sections"]["main"], 1)
    
    def test_get_document_summary_empty(self):
        """Test getting document summary for empty pipeline."""
        summary = self.pipeline.get_document_summary()
        
        self.assertIn("message", summary)
        self.assertEqual(summary["message"], "No documents loaded")
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Mock cache manager
        mock_cache_manager = Mock()
        self.pipeline.cache_manager = mock_cache_manager
        
        # Clear cache
        result = self.pipeline.clear_cache()
        
        # Verify cache was cleared
        self.assertTrue(result)
        mock_cache_manager.clear.assert_called_once()
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Add some documents
        self.pipeline.documents = [Document(page_content="Test", metadata={})]
        self.pipeline.is_initialized = True
        
        # Get metrics
        metrics = self.pipeline.get_performance_metrics()
        
        # Verify metrics structure
        self.assertIn("cache_stats", metrics)
        self.assertIn("system_info", metrics)
        self.assertIn("pipeline_status", metrics)
        self.assertIn("document_summary", metrics)


class TestIntegratedRAGPipelineFactory(unittest.TestCase):
    """Test cases for the integrated RAG pipeline factory."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline without documents."""
        pipeline = IntegratedRAGPipelineFactory.create_pipeline()
        
        self.assertIsInstance(pipeline, IntegratedRAGPipeline)
        self.assertEqual(len(pipeline.documents), 0)
        self.assertFalse(pipeline.is_initialized)
    
    def test_create_pipeline_with_documents(self):
        """Test creating a pipeline with documents."""
        # Mock the document ingestion
        with patch('RAG.app.pipeline_modules.integrated_rag_pipeline.RAGServiceLocator') as mock_locator:
            mock_services = {
                'document_ingestion': Mock(),
                'query_processing': Mock(),
                'retrieval': Mock(),
                'answer_generation': Mock(),
                'configuration': Mock(),
                'logging': Mock()
            }
            
            mock_locator.get_document_ingestion.return_value = mock_services['document_ingestion']
            mock_locator.get_query_processing.return_value = mock_services['query_processing']
            mock_locator.get_retrieval.return_value = mock_services['retrieval']
            mock_locator.get_answer_generation.return_value = mock_services['answer_generation']
            mock_locator.get_configuration.return_value = mock_services['configuration']
            mock_locator.get_logging.return_value = mock_services['logging']
            
            # Mock document processing
            mock_docs = [Document(page_content="Test content", metadata={})]
            mock_services['document_ingestion'].ingest_documents.return_value = mock_docs
            mock_services['document_ingestion'].validate_documents.return_value = True
            mock_services['document_ingestion'].preprocess_documents.return_value = mock_docs
            
            # Create pipeline with documents
            pipeline = IntegratedRAGPipelineFactory.create_pipeline_with_documents(["test_file.pdf"])
            
            self.assertIsInstance(pipeline, IntegratedRAGPipeline)
            self.assertEqual(len(pipeline.documents), 1)
            self.assertTrue(pipeline.is_initialized)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_integrated_rag_pipeline(self):
        """Test the convenience function for creating a pipeline."""
        pipeline = create_integrated_rag_pipeline()
        
        self.assertIsInstance(pipeline, IntegratedRAGPipeline)
    
    def test_process_query_with_integrated_pipeline(self):
        """Test the convenience function for processing queries."""
        # Mock the pipeline creation and query processing
        with patch('RAG.app.pipeline_modules.integrated_rag_pipeline.create_integrated_rag_pipeline') as mock_create:
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            # Mock query processing result
            mock_result = {"answer": "Test answer", "query": "test"}
            mock_pipeline.process_query.return_value = mock_result
            
            # Process query
            result = process_query_with_integrated_pipeline("test query")
            
            # Verify result
            self.assertEqual(result, mock_result)
            mock_pipeline.initialize_from_data_directory.assert_called_once()
            mock_pipeline.process_query.assert_called_once_with("test query")


if __name__ == '__main__':
    unittest.main()
