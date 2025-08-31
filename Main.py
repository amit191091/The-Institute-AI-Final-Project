from __future__ import annotations

"""Thin entrypoint delegating to app.pipeline for cleaner structure."""

# Disable auto .env parsing by third-party libs before any imports that might load dotenv
import os as _os
_os.environ.setdefault("DOTENV_DISABLE", "1")

# Default to headless + eval unless explicitly overridden, to avoid launching UI during CI/tasks
# Users can override in their environment or a .env file.
import os
from pathlib import Path
import sys
import asyncio
os.environ.setdefault("RAG_HEADLESS", "0")
os.environ.setdefault("RAG_EVAL", "1")
os.environ.setdefault("RAGAS_LLM_PROVIDER", "google")
os.environ.setdefault("RAG_USE_CE_RERANKER", "1")
# os.environ.setdefault("RAG_TRIM_ANSWERS", "1")
os.environ.setdefault("RAG_TRIM_ANSWERS", "0") #dont trim
# Prefer highly extractive behavior during eval to boost faithfulness
# os.environ.setdefault("RAG_EXTRACTIVE_FORCE", "1")
os.environ.setdefault("RAG_EXTRACTIVE_FORCE", "0") #dont extractive force
# Prune low-signal contexts to improve precision (keep recall via hybrid retrieval)
os.environ.setdefault("RAG_MIN_CTX_SCORE", "0.05")
# Try DeepEval if keys are present; safe no-op otherwise
os.environ.setdefault("RAG_DEEPEVAL", "1")

# Chroma: disable telemetry noise and set a single default persist directory
# (Prevents scattered .chroma folders and suppresses posthog telemetry errors)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

if not os.environ.get("RAG_CHROMA_DIR"):
    # Persist by default under index/chroma (can be overridden via env)
    os.environ["RAG_CHROMA_DIR"] = str(Path("index") / "chroma")

# Windows fix: avoid "attached to a different loop" errors with grpc.aio/async models
try:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

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