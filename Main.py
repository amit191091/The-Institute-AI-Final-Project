from __future__ import annotations

"""Thin entrypoint delegating to app.pipeline for cleaner structure."""

# Disable auto .env parsing by third-party libs before any imports that might load dotenv
import os as _os
_os.environ.setdefault("DOTENV_DISABLE", "1")

# =============================================================================
# RAG SYSTEM CONFIGURATION - Environment Variables Setup
# =============================================================================
# These environment variables configure the RAG system behavior for optimal performance
# during evaluation and production use. Users can override these in their .env file.

import os
# RAG_HEADLESS: Run in headless mode (no UI) by default for CI/tasks
os.environ.setdefault("RAG_HEADLESS", "0") # Set to "0" to enable Gradio UI interface

# RAG_EVAL: Enable evaluation mode by default to run RAGAS metrics
os.environ.setdefault("RAG_EVAL", "0") # Set to "0" to skip evaluation and only run inference

# RAGAS_LLM_PROVIDER: Use Google's LLM for RAGAS evaluation metrics
os.environ.setdefault("RAGAS_LLM_PROVIDER", "google") # Alternative: "openai" for OpenAI models

# RAG_USE_CE_RERANKER: Enable cross-encoder reranking for better retrieval
os.environ.setdefault("RAG_USE_CE_RERANKER", "1") # Uses a more sophisticated model to rerank retrieved documents

# RAG_TRIM_ANSWERS: Trim generated answers to reduce verbosity
os.environ.setdefault("RAG_TRIM_ANSWERS", "0") # Set to "0" to allow more detailed responses

# RAG_EXTRACTIVE_FORCE: Force extractive behavior during evaluation
os.environ.setdefault("RAG_EXTRACTIVE_FORCE", "0") # Set to "0" to allow more natural, detailed responses

# RAG_MIN_CTX_SCORE: Minimum context score threshold (0.05 = 5%)
os.environ.setdefault("RAG_MIN_CTX_SCORE", "0.05") # Prunes low-quality retrieved contexts to improve precision while maintaining recall

# RAG_DEEPEVAL: Enable DeepEval framework for additional evaluation metrics
os.environ.setdefault("RAG_DEEPEVAL", "1") # Safe no-op if DeepEval keys are not configured

# RAG_USE_CLEAN_TABLE: Enable clean table extraction for better table quality
os.environ.setdefault("RAG_USE_CLEAN_TABLE", "1") # Set to "1" to use clean table extraction

# RAG_USE_PDFPLUMBER: Disable old pdfplumber table extraction to avoid conflicts
os.environ.setdefault("RAG_USE_PDFPLUMBER", "0") # Set to "0" to disable old table extraction

from app.pipeline import run
def main() -> None:
    run()


if __name__ == "__main__":
    main()
# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load .env file


# # def Full_pipeline():
# #     print("starting full pipeline")
# #     # Placeholder for the full pipeline logic
# #     #1.file extraction + Parsing+ chunking avg chunk size :250-500, 800 tokens for table\diagram
# #     #2.metadata generation - filename, pagenumber, chunk_summary, keywords, section_type clientID\CaseID etc..
# #     #3.indexing - tables to csv\markdown , tableid, pagenum, anchor saving + small text summarization of table, vector metadate to filter retrival etc
# #     #4.Hybrid retrieval
# #     #5.Multi document support
# #     #6.gradio QA agent
# #     print("pipeline ended")




# def main():
#     print("hello world bitches")
#     return

# if __name__ == "__main__":
#     main()