"""
Tests for the new modular RAG architecture.
Verifies that all components work together correctly.
"""

import unittest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from RAG.app.interfaces import (
    DocumentIngestionInterface, QueryProcessingInterface,
    AnswerGenerationInterface, RetrievalInterface
)
# Note: These imports may need to be updated based on actual module structure
# from RAG.app.document_ingestion import DocumentIngestionService
# from RAG.app.Agent_Components.query_processing import QueryProcessingService
# from RAG.app.Agent_Components.answer_generation import AnswerGenerationService
# from RAG.app.rag_pipeline import RAGPipeline
# from RAG.app.Performance_Optimization.dependency_injection import RAGServiceLocator, DependencyContainer
from RAG.app.config import settings


class TestModularArchitecture(unittest.TestCase):
    """Test the modular RAG architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = settings
        self.container = DependencyContainer(self.config)
    
    def test_document_ingestion_service(self):
        """Test the document ingestion service."""
        service = self.container.get_document_ingestion_service()
        
        # Test that it implements the interface
        self.assertIsInstance(service, DocumentIngestionInterface)
        self.assertIsInstance(service, DocumentIngestionService)
        
        # Test configuration injection
        self.assertEqual(service.config, self.config)
    
    def test_query_processing_service(self):
        """Test the query processing service."""
        service = self.container.get_query_processing_service()
        
        # Test that it implements the interface
        self.assertIsInstance(service, QueryProcessingInterface)
        self.assertIsInstance(service, QueryProcessingService)
        
        # Test query analysis
        query = "What is the wear depth for case w1?"
        analysis = service.analyze_query(query)
        
        self.assertIn("question_type", analysis)
        self.assertIn("keywords", analysis)
        self.assertIn("is_table_question", analysis)
        self.assertEqual(analysis["original_query"], query)
    
    def test_answer_generation_service(self):
        """Test the answer generation service."""
        service = self.container.get_answer_generation_service()
        
        # Test that it implements the interface
        self.assertIsInstance(service, AnswerGenerationInterface)
        self.assertIsInstance(service, AnswerGenerationService)
        
        # Test answer formatting
        from langchain.schema import Document
        mock_doc = Document(
            page_content="Test content",
            metadata={"file_name": "test.pdf", "page": 1}
        )
        
        formatted = service.format_answer("Test answer", [mock_doc])
        self.assertIn("answer", formatted)
        self.assertIn("sources", formatted)
        self.assertIn("num_sources", formatted)
        self.assertIn("confidence", formatted)
    
    def test_service_locator(self):
        """Test the service locator pattern."""
        # Test getting services
        ingestion = RAGServiceLocator.get_document_ingestion()
        processing = RAGServiceLocator.get_query_processing()
        generation = RAGServiceLocator.get_answer_generation()
        
        self.assertIsInstance(ingestion, DocumentIngestionInterface)
        self.assertIsInstance(processing, QueryProcessingInterface)
        self.assertIsInstance(generation, AnswerGenerationInterface)
    
    def test_rag_pipeline_creation(self):
        """Test RAG pipeline creation."""
        pipeline = RAGPipeline(self.config)
        
        # Test that all services are injected
        self.assertIsInstance(pipeline.document_ingestion, DocumentIngestionInterface)
        self.assertIsInstance(pipeline.query_processing, QueryProcessingInterface)
        self.assertIsInstance(pipeline.retrieval, RetrievalInterface)
        self.assertIsInstance(pipeline.answer_generation, AnswerGenerationInterface)
    
    def test_pipeline_status(self):
        """Test pipeline status reporting."""
        pipeline = RAGPipeline(self.config)
        status = pipeline.get_pipeline_status()
        
        self.assertIn("is_initialized", status)
        self.assertIn("num_documents", status)
        self.assertIn("config", status)
        self.assertIn("services", status)
        
        # Test initial state
        self.assertFalse(status["is_initialized"])
        self.assertEqual(status["num_documents"], 0)
    
    def test_configuration_usage(self):
        """Test that configuration is properly used throughout the system."""
        # Test that query processing uses config
        service = QueryProcessingService(self.config)
        query = "What is the wear depth for case w1?"
        analysis = service.analyze_query(query)
        
        # Verify that wear case keywords are detected
        self.assertIn("w1", analysis["keywords"])
        
        # Test that technical terms are detected
        query_with_tech = "What is the accelerometer model?"
        analysis_tech = service.analyze_query(query_with_tech)
        self.assertIn("accelerometer", analysis_tech["keywords"])
    
    def test_dependency_injection_benefits(self):
        """Test that dependency injection enables better testability."""
        # Create mock services
        mock_ingestion = Mock(spec=DocumentIngestionInterface)
        mock_processing = Mock(spec=QueryProcessingInterface)
        mock_generation = Mock(spec=AnswerGenerationInterface)
        
        # Register mock services
        self.container.register_service("document_ingestion", mock_ingestion)
        self.container.register_service("query_processing", mock_processing)
        self.container.register_service("answer_generation", mock_generation)
        
        # Verify services are registered
        self.assertEqual(self.container.get_service("document_ingestion"), mock_ingestion)
        self.assertEqual(self.container.get_service("query_processing"), mock_processing)
        self.assertEqual(self.container.get_service("answer_generation"), mock_generation)
    
    def test_interface_contracts(self):
        """Test that interfaces define clear contracts."""
        # Test that all required methods exist
        ingestion = self.container.get_document_ingestion_service()
        processing = self.container.get_query_processing_service()
        generation = self.container.get_answer_generation_service()
        
        # Test DocumentIngestionInterface
        self.assertTrue(hasattr(ingestion, 'ingest_documents'))
        self.assertTrue(hasattr(ingestion, 'validate_documents'))
        
        # Test QueryProcessingInterface
        self.assertTrue(hasattr(processing, 'analyze_query'))
        self.assertTrue(hasattr(processing, 'apply_filters'))
        
        # Test AnswerGenerationInterface
        self.assertTrue(hasattr(generation, 'generate_answer'))
        self.assertTrue(hasattr(generation, 'format_answer'))
    
    def test_configuration_structure(self):
        """Test that configuration is properly structured."""
        # Test path settings
        self.assertTrue(hasattr(self.config.paths, 'PROJECT_ROOT'))
        self.assertTrue(hasattr(self.config.paths, 'DATA_DIR'))
        self.assertTrue(hasattr(self.config.paths, 'INDEX_DIR'))
        
        # Test embedding settings
        self.assertTrue(hasattr(self.config.embedding, 'EMBEDDING_MODEL_OPENAI'))
        self.assertTrue(hasattr(self.config.embedding, 'CONTEXT_TOP_N'))
        
        # Test query analysis settings
        self.assertTrue(hasattr(self.config.query_analysis, 'TECHNICAL_TERMS'))
        self.assertTrue(hasattr(self.config.query_analysis, 'WEAR_CASES'))
        self.assertTrue(hasattr(self.config.query_analysis, 'TABLE_QUESTION_KEYWORDS'))
        
        # Test reranking settings
        self.assertTrue(hasattr(self.config.reranking, 'MIN_SCORE_THRESHOLD_RATIO'))
        self.assertTrue(hasattr(self.config.reranking, 'MAX_DOCS_PER_SECTION'))


class TestConfigurationExtraction(unittest.TestCase):
    """Test that hard-coded values have been properly extracted to configuration."""
    
    def test_query_analysis_keywords_in_config(self):
        """Test that query analysis keywords are in configuration."""
        # Test technical terms
        self.assertIn("mg-5025a", settings.query_analysis.TECHNICAL_TERMS)
        self.assertIn("dytran", settings.query_analysis.TECHNICAL_TERMS)
        self.assertIn("accelerometer", settings.query_analysis.TECHNICAL_TERMS)
        
        # Test wear cases
        self.assertIn("w1", settings.query_analysis.WEAR_CASES)
        self.assertIn("w15", settings.query_analysis.WEAR_CASES)
        self.assertIn("w25", settings.query_analysis.WEAR_CASES)
        
        # Test units
        self.assertIn("rps", settings.query_analysis.UNITS)
        self.assertIn("khz", settings.query_analysis.UNITS)
        self.assertIn("mv/g", settings.query_analysis.UNITS)
    
    def test_reranking_settings_in_config(self):
        """Test that reranking settings are in configuration."""
        self.assertIsInstance(settings.reranking.MIN_SCORE_THRESHOLD_RATIO, float)
        self.assertIsInstance(settings.reranking.MAX_DOCS_PER_SECTION, int)
        self.assertIsInstance(settings.reranking.MAX_DOCS_PER_FILE, int)
        self.assertIsInstance(settings.reranking.MAX_DOCS_PER_PAGE, int)
        
        # Test scoring weights
        self.assertIsInstance(settings.reranking.LEXICAL_OVERLAP_WEIGHT, float)
        self.assertIsInstance(settings.reranking.SECTION_BONUS, float)
        self.assertIsInstance(settings.reranking.WEAR_CASE_BONUS, float)
    
    def test_fallback_settings_in_config(self):
        """Test that fallback settings are in configuration."""
        self.assertIsInstance(settings.fallback.MAX_TABLE_DOCS, int)
        self.assertIsInstance(settings.fallback.MAX_MEASUREMENT_DOCS, int)
        self.assertIsInstance(settings.fallback.MAX_SPEED_DOCS, int)
        
        # Test data patterns
        self.assertIn("w1,40", settings.fallback.WEAR_MEASUREMENTS)
        self.assertIn("15 rps", settings.fallback.SPEED_DATA)
        self.assertIn("dytran", settings.fallback.ACCELEROMETER_DATA)


if __name__ == '__main__':
    unittest.main()
