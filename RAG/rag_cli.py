#!/usr/bin/env python3
"""
RAG CLI
=======

Command-line interface for RAG operations using Typer.
"""

import typer
import logging
from pathlib import Path
from typing import Optional, List
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from RAG.app.rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="RAG System - Document analysis and question answering",
    no_args_is_help=True
)

def version_callback(value: bool):
    """Version callback for CLI."""
    if value:
        typer.echo("RAG System v1.0.0")
        raise typer.Exit()

@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)
):
    """RAG System - Advanced document analysis and question answering."""
    pass

@app.command()
def build(
    normalized: bool = typer.Option(False, "--normalized", "-n", help="Use normalized documents"),
    project_root: Optional[str] = typer.Option(None, "--project-root", "-p", help="Project root directory")
):
    """
    Build the RAG pipeline.
    
    Examples:
        rag-cli build
        rag-cli build --normalized
        rag-cli build --project-root /path/to/project
    """
    try:
        typer.echo("🔨 Building RAG pipeline...")
        
        service = RAGService(project_root)
        result = service.run_pipeline(use_normalized=normalized)
        
        typer.echo(f"✅ Pipeline built: {result['doc_count']} documents loaded")
        
    except Exception as e:
        logger.error("Error building pipeline: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    no_agent: bool = typer.Option(False, "--no-agent", help="Disable agent routing"),
    project_root: Optional[str] = typer.Option(None, "--project-root", "-p", help="Project root directory")
):
    """
    Query the RAG system.
    
    Examples:
        rag-cli query "What is the wear depth for case W15?"
        rag-cli query "Show me table 1" --no-agent
    """
    try:
        typer.echo(f"🤔 Processing query: {question}")
        
        service = RAGService(project_root)
        result = service.query(question, use_agent=not no_agent)
        
        typer.echo("\n📝 Answer:")
        typer.echo(result.get("answer", "No answer generated"))
        
        if "sources" in result:
            typer.echo(f"\n📚 Sources: {len(result['sources'])} documents")
            
        if "method" in result:
            typer.echo(f"🔧 Method: {result['method']}")
            
    except Exception as e:
        logger.error("Error processing query: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def evaluate(
    eval_file: str = typer.Argument(..., help="Path to evaluation dataset file"),
    project_root: Optional[str] = typer.Option(None, "--project-root", "-p", help="Project root directory")
):
    """
    Evaluate the RAG system using RAGAS.
    
    Examples:
        rag-cli evaluate data/eval_dataset.jsonl
    """
    try:
        typer.echo(f"📊 Running evaluation with {eval_file}")
        
        # Load evaluation data
        import json
        eval_data = []
        with open(eval_file, 'r') as f:
            for line in f:
                eval_data.append(json.loads(line.strip()))
        
        service = RAGService(project_root)
        result = service.evaluate_system(eval_data)
        
        typer.echo("\n📈 Evaluation Results:")
        if "metrics" in result:
            for metric, value in result["metrics"].items():
                typer.echo(f"  {metric}: {value:.3f}")
                
    except Exception as e:
        logger.error("Error running evaluation: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def status(
    project_root: Optional[str] = typer.Option(None, "--project-root", "-p", help="Project root directory")
):
    """
    Show the current status of the RAG system.
    """
    try:
        typer.echo("📋 RAG System Status:")
        
        service = RAGService(project_root)
        status = service.get_system_status()
        
        if "error" in status:
            typer.echo(f"❌ Error: {status['error']}")
            raise typer.Exit(1)
        
        # System status
        typer.echo(f"  🔧 Initialized: {'✅' if status['initialized'] else '❌'}")
        typer.echo(f"  📄 Documents: {status['doc_count']}")
        
        # Directory status
        typer.echo("\n📁 Directories:")
        for name, info in status["directories"].items():
            status_icon = "✅" if info["exists"] else "❌"
            typer.echo(f"  {status_icon} {name}: {info['path']}")
            if info["exists"]:
                typer.echo(f"     Files: {info['file_count']}")
                
    except Exception as e:
        logger.error("Error getting status: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def clean(
    project_root: Optional[str] = typer.Option(None, "--project-root", "-p", help="Project root directory")
):
    """
    Clean RAG system outputs and temporary files.
    """
    try:
        typer.echo("🧹 Cleaning RAG system outputs...")
        
        service = RAGService(project_root)
        service._clean_run_outputs()
        
        typer.echo("✅ Cleanup completed")
        
    except Exception as e:
        logger.error("Error during cleanup: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

def main() -> None:
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
