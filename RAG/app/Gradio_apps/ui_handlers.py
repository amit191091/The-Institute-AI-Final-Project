"""
Event handlers for the Gradio interface.
"""
import json
import re
from pathlib import Path
import gradio as gr
from RAG.app.logger import get_logger
from RAG.app.Gradio_apps.ui_components import (
    _rows_to_df, _rows_for_df
)
from RAG.app.Gradio_apps.ui_evaluation import evaluate_question_with_ground_truth


def on_shutdown():
    """Gracefully shutdown the server."""
    log = get_logger()
    log.info("Shutdown requested by user")
    try:
        # Close any open files or connections
        import os
        import signal
        
        # Send SIGTERM to current process
        os.kill(os.getpid(), signal.SIGTERM)
        return "🔄 Server shutdown initiated..."
    except Exception as e:
        log.error(f"Error during shutdown: {e}")
        return f"❌ Shutdown error: {e}"

def _on_refresh(docs, fs, qq):
    """Refresh the database explorer table."""
    # Normalize '(All)' to no filter and return a pandas DataFrame
    fsn = None if (fs in (None, "", "(All)")) else fs
    rows = _rows_for_df(docs, fsn, qq)
    return gr.update(value=_rows_to_df(rows))



