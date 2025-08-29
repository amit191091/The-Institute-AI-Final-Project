#!/usr/bin/env python3
"""
Test script to demonstrate smart source filtering functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from RAG.app.Agent_Components.agents import analyze_source_requirement
from RAG.app.retrieve import filter_documents_by_source
from langchain.schema import Document


def test_source_analysis():
    """Test the source analysis functionality."""
    
    # Test questions
    test_questions = [
        "What is the wear depth at W1?",
        "What are the database statistics for W1 across all cases?",
        "Show me the average wear depth across all cases in the database",
        "What is the wear depth at W15 in this specific case?",
        "Compare wear depths between different cases in the database",
        "What is the transmission ratio for this gearbox?",
        "What are the trends in wear depth across the database?",
        "What is the RMS value for W1 in this case study?"
    ]
    
    print("=== Source Analysis Test ===\n")
    
    for question in test_questions:
        analysis = analyze_source_requirement(question)
        print(f"Question: {question}")
        print(f"Source Type: {analysis['source_type']}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Reasoning: {analysis['reasoning']}")
        print(f"Report Score: {analysis['report_score']}, Database Score: {analysis['database_score']}")
        print("-" * 80)


def test_source_filtering():
    """Test the document filtering functionality."""
    
    # Mock documents using proper Document objects
    mock_documents = [
        Document(page_content="Main report", metadata={"file_name": "Gear wear Failure.pdf"}),
        Document(page_content="Report table", metadata={"file_name": "Gear wear Failure-table-01.csv"}),
        Document(page_content="Database", metadata={"file_name": "Database figures and tables.pdf"}),
        Document(page_content="Database table", metadata={"file_name": "Database figures and tables-table-33.csv"}),
        Document(page_content="Hebrew file", metadata={"file_name": "הנחיות פרויקט גמר.pdf"})
    ]
    
    print("\n=== Document Filtering Test ===\n")
    
    for source_type in ["report", "database", "other"]:
        filtered = filter_documents_by_source(mock_documents, source_type)
        print(f"Source Type: {source_type}")
        print(f"Filtered Documents: {[doc.metadata['file_name'] for doc in filtered]}")
        print("-" * 40)


if __name__ == "__main__":
    test_source_analysis()
    test_source_filtering()
