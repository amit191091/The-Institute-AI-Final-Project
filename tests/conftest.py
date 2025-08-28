#!/usr/bin/env python3
"""
Pytest Configuration for RAG Tests
==================================

Common fixtures and configuration for all RAG system tests.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_project_dir():
    """Create a temporary project directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents():
    """Create tiny sample documents for testing."""
    from langchain.schema import Document
    
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
def sample_eval_data():
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
def mock_embedding_function():
    """Mock embedding function for testing."""
    def mock_embed(text):
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 300  # 1500-dim vector
    return mock_embed


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    from unittest.mock import Mock
    mock = Mock()
    mock.return_value = "This is a mock response from the LLM."
    return mock


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test environment variables
    os.environ["RAG_LOG_LEVEL"] = "ERROR"
    os.environ["RAG_OCR_LANG"] = "eng"
    os.environ["RAG_PDF_HI_RES"] = "false"
    os.environ["RAG_USE_TABULA"] = "false"
    os.environ["RAG_USE_CAMELOT"] = "false"
    os.environ["RAG_USE_SYNTH_TABLES"] = "false"
    os.environ["RAG_EXTRACT_IMAGES"] = "false"
    
    yield
    
    # Clean up environment variables
    for key in [
        "RAG_LOG_LEVEL", "RAG_OCR_LANG", "RAG_PDF_HI_RES",
        "RAG_USE_TABULA", "RAG_USE_CAMELOT", "RAG_USE_SYNTH_TABLES",
        "RAG_EXTRACT_IMAGES"
    ]:
        os.environ.pop(key, None)
