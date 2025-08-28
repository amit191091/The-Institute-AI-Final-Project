#!/usr/bin/env python3
"""
Comprehensive Unit Tests for RAG Service
========================================

Unit tests with tiny fixtures for RAG service layer functionality.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from RAG.app.rag_service import RAGService
from RAG.app.config import settings
from langchain.schema import Document


class TestRAGService:
    """Test suite for RAGService class."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """Create tiny sample documents for testing."""
        return [
            Document(
                page_content="This is a test document about gear wear analysis.",
                metadata={
                    "file_name": "test_doc1.pdf",
                    "page": 1,
                    "section": "Text",
                    "anchor": "intro"
                }
            ),
            Document(
                page_content="Wear depth measurements: W1=40μm, W15=400μm, W25=608μm.",
                metadata={
                    "file_name": "test_doc2.pdf", 
                    "page": 2,
                    "section": "Table",
                    "anchor": "wear_data"
                }
            ),
            Document(
                page_content="Figure 1: Gear wear progression over time.",
                metadata={
                    "file_name": "test_doc3.pdf",
                    "page": 3,
                    "section": "Figure",
                    "anchor": "figure1"
                }
            )
        ]
    
    @pytest.fixture
    def sample_eval_data(self):
        """Create tiny sample evaluation data."""
        return [
            {
                "question": "What is the wear depth for case W15?",
                "ground_truths": ["400μm", "400 microns"],
                "answer": "The wear depth for case W15 is 400μm."
            },
            {
                "question": "Show me the gear wear figure.",
                "ground_truths": ["Figure 1", "Gear wear progression"],
                "answer": "Figure 1 shows the gear wear progression over time."
            }
        ]
    
    @pytest.fixture
    def mock_embedding_function(self):
        """Mock embedding function for testing."""
        def mock_embed(text):
            return [0.1, 0.2, 0.3, 0.4, 0.5] * 300  # 1500-dim vector
        return mock_embed
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock = Mock()
        mock.return_value = "This is a mock response from the LLM."
        return mock

    def test_initialization(self, temp_project_dir):
        """Test RAGService initialization."""
        service = RAGService(temp_project_dir)
        assert service.project_root == temp_project_dir
        assert service.docs is None
        assert service.hybrid_retriever is None
        assert service.llm is None

    def test_initialization_default_path(self):
        """Test RAGService initialization with default path."""
        service = RAGService()
        assert service.project_root == Path.cwd()
        assert service.docs is None
        assert service.hybrid_retriever is None

    @patch('RAG.app.rag_service.load_many')
    def test_load_documents_success(self, mock_load_many, sample_documents):
        """Test successful document loading."""
        mock_load_many.return_value = sample_documents
        
        service = RAGService()
        docs = service.load_documents(use_normalized=False)
        
        assert len(docs) == 3
        assert docs[0].page_content == "This is a test document about gear wear analysis."
        assert docs[1].metadata["section"] == "Table"
        mock_load_many.assert_called_once()

    @patch('RAG.app.rag_service.load_normalized_docs')
    def test_load_documents_normalized(self, mock_load_normalized, sample_documents):
        """Test loading normalized documents."""
        mock_load_normalized.return_value = sample_documents
        
        service = RAGService()
        docs = service.load_documents(use_normalized=True)
        
        assert len(docs) == 3
        mock_load_normalized.assert_called_once()

    @patch('RAG.app.rag_service.load_many')
    def test_load_documents_error(self, mock_load_many):
        """Test document loading error handling."""
        mock_load_many.side_effect = Exception("Load error")
        
        service = RAGService()
        with pytest.raises(Exception, match="Load error"):
            service.load_documents()

    @patch('RAG.app.rag_service.validate_min_pages')
    @patch('RAG.app.rag_service.structure_chunks')
    @patch('RAG.app.rag_service.attach_metadata')
    def test_process_documents_success(self, mock_attach_metadata, mock_structure_chunks, 
                                     mock_validate_min_pages, sample_documents):
        """Test successful document processing."""
        mock_structure_chunks.return_value = sample_documents
        mock_attach_metadata.return_value = sample_documents
        
        service = RAGService()
        processed_docs = service.process_documents(sample_documents)
        
        assert len(processed_docs) == 3
        mock_validate_min_pages.assert_called_once_with(sample_documents)
        mock_structure_chunks.assert_called_once()
        mock_attach_metadata.assert_called_once()

    @patch('RAG.app.rag_service.validate_min_pages')
    def test_process_documents_error(self, mock_validate_min_pages, sample_documents):
        """Test document processing error handling."""
        mock_validate_min_pages.side_effect = Exception("Validation error")
        
        service = RAGService()
        with pytest.raises(Exception, match="Validation error"):
            service.process_documents(sample_documents)

    @patch('RAG.app.rag_service.build_dense_index')
    @patch('RAG.app.rag_service.build_sparse_retriever')
    def test_build_indexes_success(self, mock_build_sparse, mock_build_dense, 
                                 sample_documents, mock_embedding_function):
        """Test successful index building."""
        mock_dense_index = Mock()
        mock_sparse_retriever = Mock()
        mock_build_dense.return_value = mock_dense_index
        mock_build_sparse.return_value = mock_sparse_retriever
        
        service = RAGService()
        service._get_embedding_function = Mock(return_value=mock_embedding_function)
        
        dense_index, sparse_retriever = service.build_indexes(sample_documents)
        
        assert dense_index == mock_dense_index
        assert sparse_retriever == mock_sparse_retriever
        mock_build_dense.assert_called_once()
        mock_build_sparse.assert_called_once()

    @patch('RAG.app.rag_service.build_dense_index')
    def test_build_indexes_error(self, mock_build_dense, sample_documents):
        """Test index building error handling."""
        mock_build_dense.side_effect = Exception("Index error")
        
        service = RAGService()
        service._get_embedding_function = Mock(return_value=None)
        
        with pytest.raises(Exception, match="Index error"):
            service.build_indexes(sample_documents)

    @patch('RAG.app.rag_service.build_hybrid_retriever')
    def test_build_hybrid_retriever_success(self, mock_build_hybrid):
        """Test successful hybrid retriever building."""
        mock_hybrid = Mock()
        mock_build_hybrid.return_value = mock_hybrid
        
        service = RAGService()
        dense_index = Mock()
        sparse_retriever = Mock()
        
        hybrid = service.build_hybrid_retriever(dense_index, sparse_retriever)
        
        assert hybrid == mock_hybrid
        mock_build_hybrid.assert_called_once()

    @patch('RAG.app.rag_service.build_hybrid_retriever')
    def test_build_hybrid_retriever_error(self, mock_build_hybrid):
        """Test hybrid retriever building error handling."""
        mock_build_hybrid.side_effect = Exception("Hybrid error")
        
        service = RAGService()
        dense_index = Mock()
        sparse_retriever = Mock()
        
        with pytest.raises(Exception, match="Hybrid error"):
            service.build_hybrid_retriever(dense_index, sparse_retriever)

    @patch.object(RAGService, 'load_documents')
    @patch.object(RAGService, 'process_documents')
    @patch.object(RAGService, 'build_indexes')
    @patch.object(RAGService, 'build_hybrid_retriever')
    def test_run_pipeline_success(self, mock_build_hybrid, mock_build_indexes,
                                mock_process_docs, mock_load_docs, sample_documents):
        """Test successful pipeline execution."""
        mock_load_docs.return_value = sample_documents
        mock_process_docs.return_value = sample_documents
        mock_build_indexes.return_value = (Mock(), Mock())
        mock_build_hybrid.return_value = Mock()
        
        service = RAGService()
        result = service.run_pipeline()
        
        assert result["doc_count"] == 3
        assert result["docs"] == sample_documents
        assert result["hybrid_retriever"] is not None
        assert result["dense_index"] is not None
        assert result["sparse_retriever"] is not None
        
        mock_load_docs.assert_called_once()
        mock_process_docs.assert_called_once()
        mock_build_indexes.assert_called_once()
        mock_build_hybrid.assert_called_once()

    @patch.object(RAGService, 'load_documents')
    def test_run_pipeline_error(self, mock_load_docs):
        """Test pipeline execution error handling."""
        mock_load_docs.side_effect = Exception("Pipeline error")
        
        service = RAGService()
        with pytest.raises(Exception, match="Pipeline error"):
            service.run_pipeline()

    @patch('RAG.app.rag_service.route_question_ex')
    def test_query_with_agent_success(self, mock_route_question, sample_documents):
        """Test successful query with agent routing."""
        mock_result = {
            "answer": "The wear depth is 400μm",
            "method": "agent_routing"
        }
        mock_route_question.return_value = mock_result
        
        service = RAGService()
        service.hybrid_retriever = Mock()
        service.llm = Mock()
        
        result = service.query("What is the wear depth for W15?", use_agent=True)
        
        assert result["answer"] == "The wear depth is 400μm"
        assert result["method"] == "agent_routing"
        mock_route_question.assert_called_once()

    def test_query_without_agent_success(self, sample_documents):
        """Test successful query without agent routing."""
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = sample_documents
        
        service = RAGService()
        service.hybrid_retriever = mock_retriever
        
        result = service.query("What is the wear depth for W15?", use_agent=False)
        
        assert result["answer"] == "Direct retrieval mode - no agent processing"
        assert result["method"] == "direct_retrieval"
        assert result["sources"] == sample_documents

    def test_query_not_initialized(self):
        """Test query when system is not initialized."""
        service = RAGService()
        service.hybrid_retriever = None
        
        with pytest.raises(ValueError, match="RAG system not initialized"):
            service.query("Test question")

    @patch('RAG.app.rag_service.run_eval_detailed')
    @patch('RAG.app.rag_service.pretty_metrics')
    def test_evaluate_system_success(self, mock_pretty_metrics, mock_run_eval, 
                                   sample_eval_data):
        """Test successful system evaluation."""
        mock_results = {"accuracy": 0.85, "precision": 0.90}
        mock_metrics = {"overall_score": 0.87}
        mock_run_eval.return_value = mock_results
        mock_pretty_metrics.return_value = mock_metrics
        
        service = RAGService()
        service.hybrid_retriever = Mock()
        service.llm = Mock()
        
        result = service.evaluate_system(sample_eval_data)
        
        assert result["results"] == mock_results
        assert result["metrics"] == mock_metrics
        mock_run_eval.assert_called_once()
        mock_pretty_metrics.assert_called_once()

    def test_evaluate_system_not_initialized(self, sample_eval_data):
        """Test evaluation when system is not initialized."""
        service = RAGService()
        service.hybrid_retriever = None
        
        with pytest.raises(ValueError, match="RAG system not initialized"):
            service.evaluate_system(sample_eval_data)

    def test_get_system_status_initialized(self, sample_documents):
        """Test system status when initialized."""
        service = RAGService()
        service.docs = sample_documents
        service.hybrid_retriever = Mock()
        
        status = service.get_system_status()
        
        assert status["initialized"] is True
        assert status["doc_count"] == 3
        assert "data_dir" in status
        assert "index_dir" in status
        assert "logs_dir" in status
        assert "directories" in status

    def test_get_system_status_not_initialized(self):
        """Test system status when not initialized."""
        service = RAGService()
        
        status = service.get_system_status()
        
        assert status["initialized"] is False
        assert status["doc_count"] == 0
        assert "directories" in status

    def test_clean_run_outputs(self, temp_project_dir):
        """Test cleanup functionality."""
        # Create test directories and files
        data_dir = temp_project_dir / "RAG" / "data"
        logs_dir = temp_project_dir / "RAG" / "logs"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        (data_dir / "images").mkdir(exist_ok=True)
        (data_dir / "elements").mkdir(exist_ok=True)
        (logs_dir / "queries.jsonl").write_text("test data")
        (logs_dir / "elements").mkdir(exist_ok=True)
        
        # Mock settings to use temp directory
        with patch('RAG.app.rag_service.settings') as mock_settings:
            mock_settings.DATA_DIR = data_dir
            mock_settings.LOGS_DIR = logs_dir
            
            service = RAGService(temp_project_dir)
            service._clean_run_outputs()
            
            # Check that cleanup worked
            assert not (data_dir / "images").exists()
            assert not (data_dir / "elements").exists()
            assert not (logs_dir / "queries.jsonl").exists()
            assert not (logs_dir / "elements").exists()

    def test_clean_run_outputs_disabled(self, temp_project_dir):
        """Test cleanup when disabled via environment variable."""
        with patch.dict(os.environ, {'RAG_CLEAN_RUN': '0'}):
            service = RAGService(temp_project_dir)
            # Should not raise any exceptions
            service._clean_run_outputs()


class TestRAGServiceIntegration:
    """Integration tests for RAGService."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data_files(self, temp_project_dir):
        """Create sample data files for integration testing."""
        data_dir = temp_project_dir / "RAG" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple test document
        test_doc = data_dir / "test_document.txt"
        test_doc.write_text("This is a test document for integration testing.")
        
        return data_dir
    
    def test_full_pipeline_integration(self, temp_project_dir, sample_data_files):
        """Test full pipeline integration with mocked components."""
        with patch('RAG.app.rag_service.settings') as mock_settings:
            mock_settings.DATA_DIR = sample_data_files
            mock_settings.INDEX_DIR = temp_project_dir / "RAG" / "index"
            mock_settings.LOGS_DIR = temp_project_dir / "RAG" / "logs"
            mock_settings.SPARSE_K = 10
            mock_settings.DENSE_K = 10
            
            # Mock all the heavy components
            with patch.multiple('RAG.app.rag_service',
                              load_many=Mock(return_value=[Document(page_content="test")]),
                              validate_min_pages=Mock(),
                              structure_chunks=Mock(return_value=[Document(page_content="test")]),
                              attach_metadata=Mock(return_value=[Document(page_content="test")]),
                              build_dense_index=Mock(return_value=Mock()),
                              build_sparse_retriever=Mock(return_value=Mock()),
                              build_hybrid_retriever=Mock(return_value=Mock())):
                
                service = RAGService(temp_project_dir)
                result = service.run_pipeline()
                
                assert result["doc_count"] == 1
                assert result["docs"] is not None
                assert result["hybrid_retriever"] is not None


class TestRAGServiceEdgeCases:
    """Edge case tests for RAGService."""
    
    def test_empty_documents(self):
        """Test handling of empty document list."""
        service = RAGService()
        
        with patch.object(service, 'load_documents', return_value=[]):
            with patch.object(service, 'process_documents', return_value=[]):
                with patch.object(service, 'build_indexes', return_value=(Mock(), Mock())):
                    with patch.object(service, 'build_hybrid_retriever', return_value=Mock()):
                        result = service.run_pipeline()
                        assert result["doc_count"] == 0

    def test_large_document_list(self):
        """Test handling of large document list."""
        large_docs = [Document(page_content=f"Document {i}") for i in range(1000)]
        
        service = RAGService()
        
        with patch.object(service, 'load_documents', return_value=large_docs):
            with patch.object(service, 'process_documents', return_value=large_docs):
                with patch.object(service, 'build_indexes', return_value=(Mock(), Mock())):
                    with patch.object(service, 'build_hybrid_retriever', return_value=Mock()):
                        result = service.run_pipeline()
                        assert result["doc_count"] == 1000

    def test_special_characters_in_question(self):
        """Test handling of special characters in questions."""
        service = RAGService()
        service.hybrid_retriever = Mock()
        service.llm = Mock()
        
        special_questions = [
            "What's the wear depth?",
            "Show me the data (W15)",
            "Is it > 400μm?",
            "Test with 'quotes' and \"double quotes\"",
            "Unicode: αβγδε"
        ]
        
        for question in special_questions:
            with patch('RAG.app.rag_service.route_question_ex') as mock_route:
                mock_route.return_value = {"answer": "Test response"}
                result = service.query(question)
                assert result["answer"] == "Test response"

    def test_concurrent_access(self):
        """Test concurrent access to RAGService."""
        import threading
        import time
        
        service = RAGService()
        service.hybrid_retriever = Mock()
        service.llm = Mock()
        
        results = []
        errors = []
        
        def query_worker():
            try:
                with patch('RAG.app.rag_service.route_question_ex') as mock_route:
                    mock_route.return_value = {"answer": "Thread response"}
                    result = service.query("Test question")
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=query_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert len(errors) == 0
        assert all(r["answer"] == "Thread response" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
