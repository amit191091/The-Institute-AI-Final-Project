#!/usr/bin/env python3
"""
Focused test for table operations and agent behavior.
Tests the specific improvements made to table parsing and lookup.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Enable verbose logging
os.environ["RAG_TRACE"] = "1"
os.environ["RAG_TRACE_RETRIEVAL"] = "1"
os.environ["LANGCHAIN_VERBOSE"] = "true"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_table_parsing():
    """Test the enhanced table parsing with Table 2 data."""
    
    print("=== Testing Table Parsing Improvements ===")
    
    try:
        from app.table_ops import markdown_to_df, natural_table_lookup, build_tables_from_docs
        from app.logger import get_logger
        
        logger = get_logger()
        
        # Test with the actual Table 2 markdown
        table2_path = Path("data/elements/Gear wear Failure-table-02.md")
        
        if not table2_path.exists():
            print(f"Table 2 file not found: {table2_path}")
            return False
        
        print(f"Testing with Table 2: {table2_path}")
        
        # Parse the markdown table
        df = markdown_to_df(table2_path)
        
        if df is not None:
            print(f"✓ Table parsed successfully: {df.shape}")
            print("Columns:", list(df.columns))
            print("Sample data:")
            print(df.head())
            
            # Test the natural lookup function with our problem questions
            test_questions = [
                "What is the sampling rate of the accelerometer?",
                "What sensor is used for vibration measurement?",
                "What is the model of the tachometer?",
                "How many teeth does the gear have?",
                "What is the accelerometer sensitivity?",
                "What type of accelerometer is used?",
            ]
            
            # Create mock documents for testing
            from langchain.schema import Document
            
            table_content = table2_path.read_text(encoding='utf-8')
            mock_doc = Document(
                page_content=table_content,
                metadata={
                    "section": "Table",
                    "table_number": 2,
                    "file_name": "Gear wear Failure.pdf"
                }
            )
            
            print("\n=== Testing Natural Table Lookup ===")
            
            for i, question in enumerate(test_questions, 1):
                print(f"\nQ{i}: {question}")
                try:
                    answer, source = natural_table_lookup(question, [mock_doc])
                    if answer:
                        print(f"A{i}: {answer}")
                        print(f"Source: {source.metadata.get('file_name', 'unknown')}")
                    else:
                        print(f"A{i}: No answer found")
                except Exception as e:
                    print(f"A{i}: Error - {e}")
                    traceback.print_exc()
            
            return True
        else:
            print("✗ Failed to parse table")
            return False
            
    except Exception as e:
        print(f"Table parsing test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_routing():
    """Test the agent routing improvements."""
    
    print("\n=== Testing Agent Routing ===")
    
    try:
        from app.agents import route_question_ex
        
        test_cases = [
            ("What is the sampling rate of the accelerometer?", "table"),
            ("What sensor is used for vibration measurement?", "table"),  
            ("Summarize the gear wear analysis", "summary"),
            ("What happened on June 13th?", "needle"),
            ("Show me Figure 2", "needle"),
        ]
        
        for question, expected_route in test_cases:
            try:
                route, trace = route_question_ex(question)
                match_symbol = "✓" if route == expected_route else "✗"
                print(f"{match_symbol} Q: {question}")
                print(f"   Route: {route} (expected: {expected_route})")
                print(f"   Trace: {trace.get('matched', [])}")
            except Exception as e:
                print(f"✗ Error routing question: {e}")
        
        return True
        
    except Exception as e:
        print(f"Agent routing test failed: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline_question():
    """Test a complete question through the pipeline."""
    
    print("\n=== Testing Full Pipeline Question ===")
    
    try:
        # Check if we have existing indexes
        from app.config import settings
        
        chroma_path = settings.INDEX_DIR / "chroma"
        if not chroma_path.exists():
            print("No existing index found. Need to run full pipeline first.")
            return False
        
        # Load existing indexes
        from app.indexing import load_dense_index, load_sparse_retriever
        from app.retrieve import build_hybrid_retriever
        from app.pipeline import ask
        from app.loaders import smart_load
        
        print("Loading existing indexes...")
        
        # Load documents and indexes
        docs = smart_load(settings.DATA_DIR)
        print(f"Loaded {len(docs)} documents")
        
        dense_store = load_dense_index()
        sparse_retriever = load_sparse_retriever()
        hybrid = build_hybrid_retriever(dense_store, sparse_retriever)
        
        print("Indexes loaded successfully")
        
        # Test question
        test_question = "What is the sampling rate of the accelerometer?"
        print(f"Testing question: {test_question}")
        
        # Mock LLM for testing
        class MockLLM:
            def invoke(self, prompt, **kwargs):
                return f"Based on the context: 50 kHz"
        
        answer = ask(docs, hybrid, MockLLM(), test_question)
        print(f"Answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"Full pipeline test failed: {e}")
        traceback.print_exc()
        return False

def run_focused_tests():
    """Run all focused tests."""
    
    print("Starting focused tests for table operations and agent behavior...")
    print("=" * 60)
    
    results = []
    
    # Test table parsing
    results.append(("Table Parsing", test_table_parsing()))
    
    # Test agent routing  
    results.append(("Agent Routing", test_agent_routing()))
    
    # Test full pipeline (if possible)
    results.append(("Full Pipeline", test_full_pipeline_question()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    return total_passed == len(results)

if __name__ == "__main__":
    try:
        success = run_focused_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)
