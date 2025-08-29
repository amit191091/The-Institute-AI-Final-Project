#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Core RAG Components
===============================================

Unit tests with tiny fixtures for core RAG functionality.
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

from langchain.schema import Document
from RAG.app.Data_Management.indexing import build_dense_index, build_sparse_retriever, to_documents, _sanitize_docs
from RAG.app.Agent_Components.agents import simplify_question, route_question, route_question_ex
from RAG.app.retrieve import query_analyzer, apply_filters, rerank_candidates, lexical_overlap
from RAG.app.chunking import structure_chunks
from RAG.app.Data_Management.metadata import attach_metadata
from RAG.app.Evaluation_Analysis.validate import validate_min_pages


class TestIndexing:
    """Test suite for indexing functionality."""
    
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
                page_content="Wear depth measurements: W1 = 40 μm, W15 = 400 μm, W25 = 608 μm.",
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
    def mock_embedding_function(self):
        """Mock embedding function for testing."""
        def mock_embed(text):
            return [0.1, 0.2, 0.3, 0.4, 0.5] * 300  # 1500-dim vector
        return mock_embed

    def test_to_documents(self, sample_documents):
        """Test converting records to documents."""
        records = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in sample_documents
        ]
        
        docs = to_documents(records)
        
        assert len(docs) == 3
        assert docs[0].page_content == "This is a test document about gear wear analysis."
        assert docs[1].metadata["section"] == "Table"
        assert docs[2].metadata["anchor"] == "figure1"

    def test_sanitize_docs(self, sample_documents):
        """Test document sanitization."""
        # Add complex metadata to test sanitization
        complex_doc = Document(
            page_content="Test content",
            metadata={
                "simple": "value",
                "list_data": [1, 2, 3],
                "dict_data": {"key": "value"},
                "tuple_data": (1, 2, 3)
            }
        )
        
        docs = [complex_doc] + sample_documents
        sanitized = _sanitize_docs(docs)
        
        assert len(sanitized) == 4
        # Complex types should be converted to strings
        assert isinstance(sanitized[0].metadata["list_data"], str)
        assert isinstance(sanitized[0].metadata["dict_data"], str)
        assert isinstance(sanitized[0].metadata["tuple_data"], str)
        # Simple types should remain unchanged
        assert sanitized[0].metadata["simple"] == "value"

    @patch('langchain_community.vectorstores.Chroma')
    def test_build_dense_index_success(self, mock_chroma, sample_documents, mock_embedding_function):
        """Test successful dense index building."""
        mock_chroma.from_documents.return_value = Mock()
        
        dense_index = build_dense_index(sample_documents, mock_embedding_function)
        
        assert dense_index is not None
        mock_chroma.from_documents.assert_called_once()

    @patch('langchain_community.vectorstores.Chroma')
    def test_build_dense_index_fallback(self, mock_chroma, sample_documents, mock_embedding_function):
        """Test dense index building with fallback."""
        mock_chroma.from_documents.side_effect = Exception("Chroma failed")
        
        # Should fallback to DocArrayInMemorySearch
        with patch('langchain_community.vectorstores.DocArrayInMemorySearch') as mock_docarray:
            mock_docarray.from_documents.return_value = Mock()
            dense_index = build_dense_index(sample_documents, mock_embedding_function)
            
            assert dense_index is not None
            mock_docarray.from_documents.assert_called_once()

    def test_build_sparse_retriever(self, sample_documents):
        """Test sparse retriever building."""
        sparse_retriever = build_sparse_retriever(sample_documents, k=5)
        
        assert sparse_retriever is not None
        assert sparse_retriever.k == 5


class TestAgents:
    """Test suite for agent functionality."""
    
    def test_simplify_question_basic(self):
        """Test basic question simplification."""
        question = "What is the wear depth for case W15?"
        result = simplify_question(question)
        
        assert result["canonical"] != ""
        assert result["wants_value"] is True
        assert result["case_id"] == "W15"
        assert result["target_attr"] == "wear depth"

    def test_simplify_question_table(self):
        """Test table question simplification."""
        question = "Show me table 1 with wear depth data"
        result = simplify_question(question)
        
        assert result["wants_table"] is True
        assert result["table_number"] == "1"
        assert result["target_attr"] == "wear depth"

    def test_simplify_question_figure(self):
        """Test figure question simplification."""
        question = "Display figure 2 showing gear wear"
        result = simplify_question(question)
        
        assert result["wants_figure"] is True
        assert result["figure_number"] == "2"

    def test_simplify_question_summary(self):
        """Test summary question simplification."""
        question = "Give me a summary of the gear failure analysis"
        result = simplify_question(question)
        
        assert result["wants_summary"] is True
        assert "summary" in result["canonical"]

    def test_simplify_question_date(self):
        """Test date question simplification."""
        question = "When did the gear failure occur?"
        result = simplify_question(question)
        
        assert result["wants_date"] is True

    def test_simplify_question_exact(self):
        """Test exact question simplification."""
        question = "What is the exact wear depth for W15?"
        result = simplify_question(question)
        
        assert result["wants_exact"] is True

    def test_simplify_question_special_characters(self):
        """Test question simplification with special characters."""
        questions = [
            "What's the wear depth?",
            "Show me the data (W15)",
            "Is it > 400 μm?",
            "Test with 'quotes' and \"double quotes\"",
            "Unicode: αβγδε"
        ]
        
        for question in questions:
            result = simplify_question(question)
            assert isinstance(result, dict)
            assert "canonical" in result

    @patch('RAG.app.agents.answer_needle')
    @patch('RAG.app.agents.answer_summary')
    @patch('RAG.app.agents.answer_table')
    def test_route_question_needle(self, mock_answer_table, mock_answer_summary, mock_answer_needle):
        """Test question routing to needle agent."""
        mock_answer_needle.return_value = "Needle answer"
        
        result = route_question("What is the gear failure mechanism?")
        
        assert result == "needle"
        mock_answer_needle.assert_not_called()  # route_question only returns the route

    @patch('RAG.app.agents.answer_needle')
    @patch('RAG.app.agents.answer_summary')
    @patch('RAG.app.agents.answer_table')
    def test_route_question_summary(self, mock_answer_table, mock_answer_summary, mock_answer_needle):
        """Test question routing to summary agent."""
        result = route_question("Give me a summary")
        
        assert result == "summary"

    @patch('RAG.app.agents.answer_needle')
    @patch('RAG.app.agents.answer_summary')
    @patch('RAG.app.agents.answer_table')
    def test_route_question_table(self, mock_answer_table, mock_answer_summary, mock_answer_needle):
        """Test question routing to table agent."""
        result = route_question("Show me the table with wear depth")
        
        assert result == "table"


class TestRetrieval:
    """Test suite for retrieval functionality."""
    
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

    def test_query_analyzer_basic(self):
        """Test basic query analysis."""
        question = "What is the wear depth for case W15?"
        result = query_analyzer(question)

        assert "question_type" in result
        assert "keywords" in result
        assert result.get("is_table_question") is True

    def test_query_analyzer_table_question(self):
        """Test table question analysis."""
        question = "Show me table 1 with wear depth data"
        result = query_analyzer(question)
        
        assert result.get("is_table_question", False) is True
        assert "1" in result.get("keywords", [])

    def test_query_analyzer_figure_question(self):
        """Test figure question analysis."""
        question = "Display figure 2 showing gear wear"
        result = query_analyzer(question)
        
        assert result.get("is_figure_question", False) is True
        assert "2" in result.get("keywords", [])

    def test_apply_filters_section(self, sample_documents):
        """Test applying section filters."""
        filters = {"section": "Table"}
        filtered = apply_filters(sample_documents, filters)
        
        assert len(filtered) == 1
        assert filtered[0].metadata["section"] == "Table"

    def test_apply_filters_file_name(self, sample_documents):
        """Test applying file name filters."""
        filters = {"file_name": "test_doc1.pdf"}
        filtered = apply_filters(sample_documents, filters)
        
        assert len(filtered) == 1
        assert filtered[0].metadata["file_name"] == "test_doc1.pdf"

    def test_apply_filters_multiple(self, sample_documents):
        """Test applying multiple filters."""
        filters = {"section": "Text", "page": 1}
        filtered = apply_filters(sample_documents, filters)
        
        assert len(filtered) == 1
        assert filtered[0].metadata["section"] == "Text"
        assert filtered[0].metadata["page"] == 1

    def test_apply_filters_no_match(self, sample_documents):
        """Test applying filters with no matches."""
        filters = {"section": "Nonexistent"}
        filtered = apply_filters(sample_documents, filters)
        
        assert len(filtered) == 0

    def test_rerank_candidates(self, sample_documents):
        """Test candidate reranking."""
        query = "wear depth"
        reranked = rerank_candidates(query, sample_documents, top_n=2)
        
        assert len(reranked) >= 1  # Should return at least one document
        assert len(reranked) <= 2  # Should not return more than requested
        # Document with "wear depth" should be ranked higher if present
        if len(reranked) > 1:
            wear_depth_docs = [doc for doc in reranked if "wear depth" in doc.page_content.lower()]
            if wear_depth_docs:
                assert wear_depth_docs[0] == reranked[0]

    def test_lexical_overlap(self):
        """Test lexical overlap calculation."""
        query = "wear depth measurements"
        text = "The wear depth measurements show W1=40μm, W15=400μm"
        
        overlap = lexical_overlap(query, text)
        
        assert isinstance(overlap, float)
        assert 0 <= overlap <= 1

    def test_lexical_overlap_no_overlap(self):
        """Test lexical overlap with no common words."""
        query = "completely different words"
        text = "entirely separate content here"
        
        overlap = lexical_overlap(query, text)
        
        assert overlap == 0.0


class TestChunking:
    """Test suite for chunking functionality."""
    
    @pytest.fixture
    def sample_elements(self):
        """Create tiny sample elements for testing."""
        return [
            Document(
                page_content="This is a short test element.",
                metadata={"page": 1}
            ),
            Document(
                page_content="This is a longer test element that should be chunked properly according to the token limits and other chunking parameters.",
                metadata={"page": 2}
            ),
            Document(
                page_content="Another element with some content.",
                metadata={"page": 3}
            )
        ]

    def test_structure_chunks_basic(self, sample_elements):
        """Test basic chunk structuring."""
        chunks = structure_chunks(sample_elements, "test_file.pdf")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)
        assert all("file_name" in chunk for chunk in chunks)

    def test_structure_chunks_metadata_preservation(self, sample_elements):
        """Test that metadata is preserved during chunking."""
        chunks = structure_chunks(sample_elements, "test_file.pdf")
        
        for chunk in chunks:
            assert "file_name" in chunk
            assert chunk["file_name"] == "test_file.pdf"


class TestMetadata:
    """Test suite for metadata functionality."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create tiny sample chunks for testing."""
        return [
            {
                "content": "This is a test chunk.",
                "file_name": "test_file.pdf",
                "page": 1
            },
            {
                "content": "Another test chunk with more content.",
                "file_name": "test_file.pdf", 
                "page": 2
            }
        ]

    def test_attach_metadata_basic(self, sample_chunks):
        """Test basic metadata attachment."""
        processed = attach_metadata(sample_chunks[0])
        
        assert isinstance(processed, dict)
        assert "metadata" in processed
        assert "page_content" in processed

    def test_attach_metadata_with_client_id(self, sample_chunks):
        """Test metadata attachment with client ID."""
        processed = attach_metadata(sample_chunks[0], client_id="test_client")
        
        assert processed["metadata"]["client_id"] == "test_client"

    def test_attach_metadata_with_case_id(self, sample_chunks):
        """Test metadata attachment with case ID."""
        processed = attach_metadata(sample_chunks[0], case_id="test_case")
        
        assert processed["metadata"]["case_id"] == "test_case"


class TestValidation:
    """Test suite for validation functionality."""
    
    def test_validate_min_pages_sufficient(self):
        """Test validation with sufficient pages."""
        result, message = validate_min_pages(15, 10)

        assert result is True
        assert "ok" in message.lower()

    def test_validate_min_pages_insufficient(self):
        """Test validation with insufficient pages."""
        result, message = validate_min_pages(5, 10)

        assert result is False
        assert "document has 5 pages" in message.lower()

    def test_validate_min_pages_exact(self):
        """Test validation with exact page count."""
        result, message = validate_min_pages(10, 10)
        
        assert result is True


class TestCoreComponentsIntegration:
    """Integration tests for core components."""
    
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
            )
        ]

    def test_full_retrieval_pipeline(self, sample_documents):
        """Test full retrieval pipeline integration."""
        # Test query analysis
        question = "What is the wear depth for case W15?"
        qa_result = query_analyzer(question)

        assert "question_type" in qa_result
        assert qa_result.get("is_table_question") is True
        
        # Test filtering
        filters = {"section": "Table"}  # Create filters manually since query_analyzer doesn't return them
        filtered = apply_filters(sample_documents, filters)
        
        # Test reranking
        reranked = rerank_candidates(question, filtered, top_n=2)
        
        assert len(reranked) <= 2
        assert all(isinstance(doc, Document) for doc in reranked)

    def test_question_routing_integration(self):
        """Test question routing integration."""
        questions = [
            "What is the wear depth for W15?",
            "Show me table 1",
            "Give me a summary",
            "Display figure 2"
        ]
        
        for question in questions:
            # Test simplification
            simplified = simplify_question(question)
            assert isinstance(simplified, dict)
            
            # Test routing
            route = route_question(question)
            assert route in ["needle", "summary", "table"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
