from __future__ import annotations

"""Thin entrypoint delegating to RAG.app.pipeline for cleaner structure."""

# Set environment variables, before ANY imports
import os
from pathlib import Path

def setup_settings():
    """Set environment variables for startup - called before any heavy imports"""
    """The settings are set to false by default for quick startup - fast startup but poor results"""
    """The settings are set to true by default for detailed analysis - good results but affect startup time"""
    
    # Load environment variables from .env file first
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            print("Environment variables loaded from .env file")
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
    
    # Core RAG System Flags (from the log message)
    os.environ["RAG_HEADLESS"] = "false"  # Run with UI by default, set to "true" for headless mode
    os.environ["RAG_EVAL"] = "false"  # Evaluation mode, set to "true" to run evaluation
    os.environ["RAG_USE_NORMALIZED"] = "true"  # Use normalized documents, set to "true" for normalized processing
    os.environ["RAG_VECTOR_BACKEND"] = "chroma"  # Vector database backend (chroma, pinecone, etc.)
    os.environ["RAG_ENABLE_LLAMAINDEX"] = "true"  # Enable LlamaIndex integration, set to "true" to enable
    os.environ["RAG_USE_LLAMAPARSE"] = "true"  # Enable LlamaParse integration, set to "true" to enable
    
    # Performance and Processing Flags
    os.environ["RAG_PDF_HI_RES"] = "true" # High-res OCR insert true or false
    os.environ["RAG_USE_TABULA"] = "false" # Tabula table extraction - disabled to avoid conflicts
    os.environ["RAG_USE_CAMELOT"] = "false" # Camelot table extraction - disabled to avoid conflicts
    os.environ["RAG_EXTRACT_IMAGES"] = "true" # Image extraction insert true or false
    os.environ["RAG_SYNTH_TABLES"] = "true" # Table synthesis insert true or false
    os.environ["RAG_USE_PDFPLUMBER"] = "true" # PDFPlumber table extraction - keep only this one
    os.environ["RAG_OCR_LANG"] = "eng" # OCR language insert eng for English or other language code
    os.environ["RAG_LOG_LEVEL"] = "INFO" # Log level insert ERROR or INFO or DEBUG
    
    # Additional RAG System Flags
    os.environ["RAG_FAST_EVAL"] = "true"  # Fast evaluation mode (essential metrics only), set to "false" for full evaluation
    os.environ["RAG_BATCH_TIMEOUT"] = "300"  # Batch processing timeout in seconds
    os.environ["RAG_GRAPH_DB"] = "false"  # Enable graph database integration, set to "true" to enable
    os.environ["RAG_USE_NORMALIZED_GRAPH"] = "true"  # Use normalized graph processing, set to "true" to enable
    os.environ["RAG_BUILD_ALT_INDEXES"] = "false"  # Build alternative indexes, set to "true" to enable
    os.environ["RAG_DEEPEVAL"] = "false"  # Enable DeepEval integration, set to "true" to enable

# Set fast mode by default for quick startup
setup_settings()

# Now import lightweight libraries
import typer
from typing import Optional
from pathlib import Path

# Lazy import function to avoid heavy imports during startup
def _import_pipeline():
    """Import pipeline components only when needed"""
    from RAG.app.pipeline import run, run_evaluation, build_pipeline
    from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths, clean_run_outputs
    return run, run_evaluation, build_pipeline, discover_input_paths, clean_run_outputs

app = typer.Typer(help="RAG System for Gear Wear Analysis", no_args_is_help=True)

def version_callback(value: bool):
    if value:
        typer.echo("RAG System v1.0.0")
        raise typer.Exit()

@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)
):
    """RAG System for Gear Wear Analysis - Advanced document processing and Q&A."""
    pass

@app.command()
def start(
    headless: bool = typer.Option(False, "--headless", "-h", help="Run without Gradio UI"),
    port: int = typer.Option(7860, "--port", "-p", help="Port for Gradio interface"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for Gradio interface")
):
    """Start the RAG system with optional Gradio UI."""
    
    # Lazy import to avoid heavy library loading
    run, _, _, _, _ = _import_pipeline()
    
    if headless:
        typer.echo("🚀 Starting RAG system in headless mode...")
        # Set headless environment variable
        os.environ["RAG_HEADLESS"] = "true"
        # Run without UI
        run()
    else:
        typer.echo(f"🚀 Starting RAG system with Gradio UI on {host}:{port}...")
        # Explicitly set headless to false to ensure UI runs
        os.environ["RAG_HEADLESS"] = "false"
        # Set UI environment variables
        os.environ["GRADIO_SERVER_NAME"] = host
        os.environ["GRADIO_PORT"] = str(port)
        # Run with UI
        run()

@app.command()
def build():
    """Build the RAG pipeline without starting the UI."""
    typer.echo("🔨 Building RAG pipeline...")
    
    # Lazy import to avoid heavy library loading
    _, _, build_pipeline, discover_input_paths, clean_run_outputs = _import_pipeline()
    
    # Clean outputs and discover paths
    clean_run_outputs()
    paths = discover_input_paths()
    
    if not paths:
        typer.echo("❌ No input files found. Place PDFs/DOCs under data/ or the root PDF.")
        raise typer.Exit(1)
    
    # Build pipeline
    docs, hybrid, llm = build_pipeline(paths)
    typer.echo(f"✅ Pipeline built: {len(docs)} documents loaded")

@app.command()
def evaluate(
    questions: int = typer.Option(30, "--questions", "-q", help="Number of questions to evaluate"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    fast: bool = typer.Option(True, "--fast/--full", help="Fast evaluation (essential metrics only) or full evaluation (all metrics)")
):
    """Run RAGAS evaluation on the system."""
    typer.echo(f"📊 Running evaluation with {questions} questions...")
    
    # Set evaluation environment
    os.environ["RAG_EVAL"] = "1"
    os.environ["RAG_HEADLESS"] = "1"
    # Enable fast evaluation mode (only essential metrics)
    os.environ["RAG_FAST_EVAL"] = "1" if fast else "0" # 0 for full evaluation, 1 for fast evaluation
    # Increase timeout for evaluation
    os.environ["RAG_BATCH_TIMEOUT"] = "600"  # 10 minutes
    
    # Lazy import to avoid heavy library loading
    _, run_evaluation, build_pipeline, discover_input_paths, clean_run_outputs = _import_pipeline()
    
    # Clean outputs and discover paths
    clean_run_outputs()
    paths = discover_input_paths()
    
    if not paths:
        typer.echo("❌ No input files found. Place PDFs/DOCs under data/ or the root PDF.")
        raise typer.Exit(1)
    
    # Build pipeline and run evaluation
    docs, hybrid, llm = build_pipeline(paths)
    
    run_evaluation(docs, hybrid, llm, questions)
    
    typer.echo("✅ Evaluation completed!")

@app.command()
def status():
    """Show the current status of the RAG system."""
    typer.echo("📋 RAG System Status:")
    
    # Lazy import to avoid heavy library loading
    _, _, _, discover_input_paths, _ = _import_pipeline()
    
    # Check for input files
    paths = discover_input_paths()
    if paths:
        typer.echo(f"  📁 Input files: {len(paths)} found")
        for path in paths:
            typer.echo(f"    - {path.name}")
    else:
        typer.echo("  ❌ No input files found")
    
    # Check for index
    from RAG.app.config import settings
    if settings.paths.INDEX_DIR.exists():
        typer.echo("  ✅ Index directory exists")
    else:
        typer.echo("  ❌ Index directory not found")
    
    # Check for logs
    if settings.paths.LOGS_DIR.exists():
        typer.echo("  ✅ Logs directory exists")
    else:
        typer.echo("  ❌ Logs directory not found")
    
    # Show core RAG system settings
    typer.echo("\n🔧 Core RAG System Settings:")
    core_settings = [
        ("RAG_HEADLESS", "Headless Mode"),
        ("RAG_EVAL", "Evaluation Mode"),
        ("RAG_USE_NORMALIZED", "Normalized Documents"),
        ("RAG_VECTOR_BACKEND", "Vector Backend"),
        ("RAG_ENABLE_LLAMAINDEX", "LlamaIndex Integration"),
        ("RAG_USE_LLAMAPARSE", "LlamaParse Integration"),
        ("RAG_FAST_EVAL", "Fast Evaluation"),
        ("RAG_GRAPH_DB", "Graph Database"),
        ("RAG_DEEPEVAL", "DeepEval Integration")
    ]
    
    for env_var, description in core_settings:
        value = os.getenv(env_var, "auto")
        if env_var == "RAG_VECTOR_BACKEND":
            status = f"🔗 {value}"
        else:
            status = "❌ Disabled" if value.lower() == "false" else "✅ Enabled" if value.lower() == "true" else "⚙️  Auto"
        typer.echo(f"  {description}: {status}")
    
    # Show performance settings
    typer.echo("\n⚡ Performance Settings:")
    settings = [
        ("RAG_PDF_HI_RES", "High-res OCR"),
        ("RAG_USE_TABULA", "Tabula tables"),
        ("RAG_USE_CAMELOT", "Camelot tables"),
        ("RAG_EXTRACT_IMAGES", "Image extraction"),
        ("RAG_SYNTH_TABLES", "Table synthesis"),
        ("RAG_USE_PDFPLUMBER", "PDFPlumber tables")
    ]
    
    for env_var, description in settings:
        value = os.getenv(env_var, "auto")
        status = "❌ Disabled" if value.lower() == "false" else "✅ Enabled" if value.lower() == "true" else "⚙️  Auto"
        typer.echo(f"  {description}: {status}")

def main() -> None:
    """Main entry point for the RAG system."""
    app()

if __name__ == "__main__":
    main()
