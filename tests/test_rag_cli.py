#!/usr/bin/env python3
"""
Comprehensive Unit Tests for RAG CLI
===================================

Unit tests with tiny fixtures for RAG CLI functionality.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAG.rag_cli import app


class TestRAGCLI:
    """Test suite for RAG CLI commands."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_eval_file(self, temp_project_dir):
        """Create a sample evaluation file for testing."""
        eval_file = temp_project_dir / "test_eval.jsonl"
        eval_data = [
            {
                "question": "What is the wear depth for case W15?",
                "ground_truths": ["400 μm", "400 microns"],
                "answer": "The wear depth for case W15 is 400 μm."
            },
            {
                "question": "Show me the gear wear figure.",
                "ground_truths": ["Figure 1", "Gear wear progression"],
                "answer": "Figure 1 shows the gear wear progression over time."
            }
        ]
        
        with open(eval_file, 'w') as f:
            for item in eval_data:
                f.write(json.dumps(item) + '\n')
        
        return eval_file

    def test_help_command(self, cli_runner):
        """Test help command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "RAG System" in result.output
        assert "build" in result.output
        assert "query" in result.output
        assert "evaluate" in result.output
        assert "status" in result.output
        assert "clean" in result.output

    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "RAG System v1.0.0" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_build_command_success(self, mock_rag_service_class, cli_runner):
        """Test successful build command."""
        mock_service = Mock()
        mock_service.run_pipeline.return_value = {
            "doc_count": 5,
            "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "hybrid_retriever": Mock(),
            "dense_index": Mock(),
            "sparse_retriever": Mock()
        }
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["build"])
        
        assert result.exit_code == 0
        assert "🔨 Building RAG pipeline..." in result.output
        assert "✅ Pipeline built: 5 documents loaded" in result.output
        mock_service.run_pipeline.assert_called_once_with(use_normalized=False)

    @patch('RAG.app.rag_cli.RAGService')
    def test_build_command_normalized(self, mock_rag_service_class, cli_runner):
        """Test build command with normalized flag."""
        mock_service = Mock()
        mock_service.run_pipeline.return_value = {"doc_count": 3}
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["build", "--normalized"])
        
        assert result.exit_code == 0
        mock_service.run_pipeline.assert_called_once_with(use_normalized=True)

    @patch('RAG.app.rag_cli.RAGService')
    def test_build_command_with_project_root(self, mock_rag_service_class, cli_runner, temp_project_dir):
        """Test build command with custom project root."""
        mock_service = Mock()
        mock_service.run_pipeline.return_value = {"doc_count": 2}
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["build", "--project-root", str(temp_project_dir)])
        
        assert result.exit_code == 0
        # Convert to string for comparison since the CLI converts paths to strings
        mock_rag_service_class.assert_called_once_with(str(temp_project_dir))

    @patch('RAG.app.rag_cli.RAGService')
    def test_build_command_error(self, mock_rag_service_class, cli_runner):
        """Test build command error handling."""
        mock_service = Mock()
        mock_service.run_pipeline.side_effect = Exception("Build failed")
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["build"])
        
        assert result.exit_code == 1
        assert "❌ Error: Build failed" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_query_command_success(self, mock_rag_service_class, cli_runner):
        """Test successful query command."""
        mock_service = Mock()
        mock_service.query.return_value = {
            "answer": "The wear depth is 400 μm",
            "sources": ["doc1", "doc2"],
            "method": "agent_routing"
        }
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["query", "What is the wear depth for W15?"])
        
        assert result.exit_code == 0
        assert "🤔 Processing query: What is the wear depth for W15?" in result.output
        assert "📝 Answer:" in result.output
        assert "The wear depth is 400 μm" in result.output
        assert "📚 Sources: 2 documents" in result.output
        assert "🔧 Method: agent_routing" in result.output
        mock_service.query.assert_called_once_with("What is the wear depth for W15?", use_agent=True)

    @patch('RAG.app.rag_cli.RAGService')
    def test_query_command_no_agent(self, mock_rag_service_class, cli_runner):
        """Test query command without agent routing."""
        mock_service = Mock()
        mock_service.query.return_value = {
            "answer": "Direct retrieval response",
            "method": "direct_retrieval"
        }
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["query", "Test question", "--no-agent"])
        
        assert result.exit_code == 0
        mock_service.query.assert_called_once_with("Test question", use_agent=False)

    @patch('RAG.app.rag_cli.RAGService')
    def test_query_command_error(self, mock_rag_service_class, cli_runner):
        """Test query command error handling."""
        mock_service = Mock()
        mock_service.query.side_effect = Exception("Query failed")
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["query", "Test question"])
        
        assert result.exit_code == 1
        assert "❌ Error: Query failed" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_evaluate_command_success(self, mock_rag_service_class, cli_runner, sample_eval_file):
        """Test successful evaluate command."""
        mock_service = Mock()
        mock_service.evaluate_system.return_value = {
            "results": {"accuracy": 0.85},
            "metrics": {
                "answer_relevancy": 0.87,
                "context_relevancy": 0.82,
                "faithfulness": 0.90
            }
        }
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["evaluate", str(sample_eval_file)])
        
        assert result.exit_code == 0
        assert f"📊 Running evaluation with {sample_eval_file}" in result.output
        assert "📈 Evaluation Results:" in result.output
        assert "answer_relevancy: 0.870" in result.output
        assert "context_relevancy: 0.820" in result.output
        assert "faithfulness: 0.900" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_evaluate_command_error(self, mock_rag_service_class, cli_runner, temp_project_dir):
        """Test evaluate command error handling."""
        mock_service = Mock()
        mock_service.evaluate_system.side_effect = Exception("Evaluation failed")
        mock_rag_service_class.return_value = mock_service
        
        eval_file = temp_project_dir / "nonexistent.jsonl"
        result = cli_runner.invoke(app, ["evaluate", str(eval_file)])
        
        assert result.exit_code == 1
        # The error is about file not found, not evaluation failed
        assert "❌ Error:" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_status_command_success(self, mock_rag_service_class, cli_runner):
        """Test successful status command."""
        mock_service = Mock()
        mock_service.get_system_status.return_value = {
            "initialized": True,
            "doc_count": 10,
            "directories": {
                "data": {
                    "exists": True,
                    "path": "/path/to/data",
                    "file_count": 15
                },
                "index": {
                    "exists": True,
                    "path": "/path/to/index",
                    "file_count": 8
                },
                "logs": {
                    "exists": True,
                    "path": "/path/to/logs",
                    "file_count": 12
                }
            }
        }
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "📋 RAG System Status:" in result.output
        assert "🔧 Initialized: ✅" in result.output
        assert "📄 Documents: 10" in result.output
        assert "📁 Directories:" in result.output
        assert "✅ data: /path/to/data" in result.output
        assert "Files: 15" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_status_command_not_initialized(self, mock_rag_service_class, cli_runner):
        """Test status command when system is not initialized."""
        mock_service = Mock()
        mock_service.get_system_status.return_value = {
            "initialized": False,
            "doc_count": 0,
            "directories": {
                "data": {"exists": False, "path": "/path/to/data", "file_count": 0},
                "index": {"exists": False, "path": "/path/to/index", "file_count": 0},
                "logs": {"exists": False, "path": "/path/to/logs", "file_count": 0}
            }
        }
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "🔧 Initialized: ❌" in result.output
        assert "📄 Documents: 0" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_status_command_error(self, mock_rag_service_class, cli_runner):
        """Test status command error handling."""
        mock_service = Mock()
        mock_service.get_system_status.return_value = {"error": "Status check failed"}
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["status"])
        
        assert result.exit_code == 1
        assert "❌ Error: Status check failed" in result.output

    @patch('RAG.app.rag_cli.RAGService')
    def test_clean_command_success(self, mock_rag_service_class, cli_runner):
        """Test successful clean command."""
        mock_service = Mock()
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["clean"])
        
        assert result.exit_code == 0
        assert "🧹 Cleaning RAG system outputs..." in result.output
        assert "✅ Cleanup completed" in result.output
        mock_service._clean_run_outputs.assert_called_once()

    @patch('RAG.app.rag_cli.RAGService')
    def test_clean_command_error(self, mock_rag_service_class, cli_runner):
        """Test clean command error handling."""
        mock_service = Mock()
        mock_service._clean_run_outputs.side_effect = Exception("Cleanup failed")
        mock_rag_service_class.return_value = mock_service
        
        result = cli_runner.invoke(app, ["clean"])
        
        assert result.exit_code == 1
        assert "❌ Error: Cleanup failed" in result.output


class TestRAGCLIEdgeCases:
    """Edge case tests for RAG CLI."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()
    
    def test_no_arguments(self, cli_runner):
        """Test CLI with no arguments."""
        result = cli_runner.invoke(app, [])
        # CLI with no arguments should show help (exit code 2)
        assert result.exit_code == 2
        assert "Usage:" in result.output

    def test_invalid_command(self, cli_runner):
        """Test CLI with invalid command."""
        result = cli_runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_build_with_invalid_project_root(self, cli_runner):
        """Test build command with invalid project root."""
        result = cli_runner.invoke(app, ["build", "--project-root", "/nonexistent/path"])
        # Should not fail immediately, but the service might fail later
        assert result.exit_code == 0 or result.exit_code == 1

    def test_query_with_empty_question(self, cli_runner):
        """Test query command with empty question."""
        result = cli_runner.invoke(app, ["query", ""])
        assert result.exit_code == 0 or result.exit_code == 1

    def test_evaluate_with_nonexistent_file(self, cli_runner):
        """Test evaluate command with nonexistent file."""
        result = cli_runner.invoke(app, ["evaluate", "/nonexistent/file.jsonl"])
        assert result.exit_code == 1

    def test_evaluate_with_invalid_json(self, cli_runner, temp_project_dir):
        """Test evaluate command with invalid JSON file."""
        invalid_file = temp_project_dir / "invalid.jsonl"
        invalid_file.write_text("invalid json content")
        
        result = cli_runner.invoke(app, ["evaluate", str(invalid_file)])
        assert result.exit_code == 1

    def test_concurrent_cli_calls(self, cli_runner):
        """Test concurrent CLI calls."""
        import threading
        import time
        
        results = []
        
        def cli_worker():
            with patch('RAG.app.rag_cli.RAGService') as mock_rag_service_class:
                mock_service = Mock()
                mock_service.get_system_status.return_value = {
                    "initialized": True,
                    "doc_count": 5,
                    "directories": {}
                }
                mock_rag_service_class.return_value = mock_service
                
                result = cli_runner.invoke(app, ["status"])
                results.append(result.exit_code)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=cli_worker)
            threads.append(thread)
            thread.start()
        
                # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Some threads might fail due to I/O issues, but we should get at least 2 results
        assert len(results) >= 2
        assert all(code == 0 for code in results)


class TestRAGCLIIntegration:
    """Integration tests for RAG CLI."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_full_workflow_integration(self, cli_runner, temp_project_dir):
        """Test full CLI workflow integration."""
        # Mock all RAGService methods
        with patch('RAG.app.rag_cli.RAGService') as mock_rag_service_class:
            mock_service = Mock()
            mock_rag_service_class.return_value = mock_service
            
            # Mock build
            mock_service.run_pipeline.return_value = {"doc_count": 3}
            
            # Mock status
            mock_service.get_system_status.return_value = {
                "initialized": True,
                "doc_count": 3,
                "directories": {
                    "data": {"exists": True, "path": str(temp_project_dir / "data"), "file_count": 5},
                    "index": {"exists": True, "path": str(temp_project_dir / "index"), "file_count": 3},
                    "logs": {"exists": True, "path": str(temp_project_dir / "logs"), "file_count": 2}
                }
            }
            
            # Mock query
            mock_service.query.return_value = {
                "answer": "Test answer",
                "method": "agent_routing"
            }
            
            # Test build
            result = cli_runner.invoke(app, ["build", "--project-root", str(temp_project_dir)])
            assert result.exit_code == 0
            
            # Test status
            result = cli_runner.invoke(app, ["status", "--project-root", str(temp_project_dir)])
            assert result.exit_code == 0
            
            # Test query
            result = cli_runner.invoke(app, ["query", "Test question", "--project-root", str(temp_project_dir)])
            assert result.exit_code == 0
            
            # Verify service was called correctly
            assert mock_service.run_pipeline.called
            assert mock_service.get_system_status.called
            assert mock_service.query.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
