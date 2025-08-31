#!/usr/bin/env python3
"""
Pipeline Utilities Module
========================

Utility functions and helpers for the RAG pipeline.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from RAG.app.config import settings
from RAG.app.logger import get_logger


def get_embeddings():
    """Use config-based provider selection for embeddings."""
    # Use config setting for primary LLM provider
    primary_provider = settings.llm.PRIMARY_LLM_PROVIDER.lower()
    
    # Check for environment override
    force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
    if force_openai:
        primary_provider = "openai"
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception:
        GoogleGenerativeAIEmbeddings = None  # type: ignore
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None  # type: ignore

    # Initialize based on primary provider setting
    if primary_provider == "openai":
        if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
            return OpenAIEmbeddings(model=settings.embedding.EMBEDDING_MODEL_OPENAI)
    elif primary_provider == "google":
        if os.getenv("GOOGLE_API_KEY") and GoogleGenerativeAIEmbeddings is not None:
            return GoogleGenerativeAIEmbeddings(model=settings.embedding.EMBEDDING_MODEL_GOOGLE)
    elif primary_provider == "auto":
        # Auto mode: try Google first, then OpenAI
        if os.getenv("GOOGLE_API_KEY") and GoogleGenerativeAIEmbeddings is not None:
            return GoogleGenerativeAIEmbeddings(model=settings.embedding.EMBEDDING_MODEL_GOOGLE)
    
    # Fallback to OpenAI if primary provider failed or not set
    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model=settings.embedding.EMBEDDING_MODEL_OPENAI)
    
    # Final fallback: FakeEmbeddings
    try:
        from langchain_community.embeddings import FakeEmbeddings  # type: ignore
        print("[Embeddings] Using FakeEmbeddings fallback (no API keys found)")
        return FakeEmbeddings(size=settings.llm.FALLBACK_EMBEDDING_SIZE)
    except Exception:
        pass
    raise RuntimeError(
        "No embedding backend available. Set GOOGLE_API_KEY or OPENAI_API_KEY, or ensure langchain_community FakeEmbeddings is available."
    )


class LLM:
    """Simple callable LLM wrapper preferring Gemini; fallback to OpenAI via LangChain chat models."""

    def __init__(self) -> None:
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        
        self._backend = None
        self._which = None
        
        # Use config setting for primary LLM provider
        primary_provider = settings.llm.PRIMARY_LLM_PROVIDER.lower()
        
        # Check for environment override
        force_openai = os.getenv(settings.llm.FORCE_OPENAI_ENV, "").strip().lower() in ("1", "true", "yes")
        if force_openai:
            primary_provider = "openai"
        
        # Initialize based on primary provider setting
        if primary_provider == "openai":
            # Try OpenAI first
            if os.getenv(settings.llm.OPENAI_API_KEY_ENV):
                try:
                    from langchain_openai import ChatOpenAI
                    model = os.getenv(settings.llm.OPENAI_CHAT_MODEL_ENV, settings.llm.OPENAI_MODEL)
                    try:
                        self._backend = ChatOpenAI(
                            model=model, 
                            temperature=settings.llm.TEMPERATURE
                        )  # type: ignore[call-arg]
                    except Exception:
                        try:
                            self._backend = ChatOpenAI(model_name=model)  # type: ignore[call-arg]
                        except Exception:
                            self._backend = ChatOpenAI()  # type: ignore[call-arg]
                    self._which = "openai"
                except Exception:
                    self._backend = None
                    
        elif primary_provider == "google":
            # Try Google first
            if os.getenv(settings.llm.GOOGLE_API_KEY_ENV):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self._backend = ChatGoogleGenerativeAI(
                        model=settings.llm.GOOGLE_MODEL, 
                        temperature=settings.llm.TEMPERATURE
                    )
                    self._which = "google"
                except Exception:
                    self._backend = None
                    
        elif primary_provider == "auto":
            # Auto mode: try Google first, then OpenAI
            if os.getenv(settings.llm.GOOGLE_API_KEY_ENV):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self._backend = ChatGoogleGenerativeAI(
                        model=settings.llm.GOOGLE_MODEL, 
                        temperature=settings.llm.TEMPERATURE
                    )
                    self._which = "google"
                except Exception:
                    self._backend = None
        
        # Fallback to OpenAI if primary provider failed or not set
        if self._backend is None and os.getenv(settings.llm.OPENAI_API_KEY_ENV):
            try:
                from langchain_openai import ChatOpenAI
                model = os.getenv(settings.llm.OPENAI_CHAT_MODEL_ENV, settings.llm.OPENAI_MODEL)
                try:
                    self._backend = ChatOpenAI(
                        model=model, 
                        temperature=settings.llm.TEMPERATURE
                    )  # type: ignore[call-arg]
                except Exception:
                    try:
                        self._backend = ChatOpenAI(model_name=model)  # type: ignore[call-arg]
                    except Exception:
                        self._backend = ChatOpenAI()  # type: ignore[call-arg]
                self._which = "openai"
            except Exception:
                self._backend = None

    def __call__(self, prompt: str) -> str:
        if self._backend is not None:
            try:
                resp = self._backend.invoke(prompt)
                return getattr(resp, "content", str(resp))
            except Exception as e:  # pragma: no cover
                return f"[LLM error] {e}\n\n{prompt[-settings.llm.ERROR_PROMPT_LENGTH:]}"
        return "[LLM not configured] " + prompt[-settings.llm.ERROR_PROMPT_LENGTH:]


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a .json (list/dict) or .jsonl file into a list of dicts."""
    try:
        if path.suffix.lower() == ".jsonl":
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
            return rows
        else:
            obj = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [{"key": k, "value": v} for k, v in obj.items()]
            return []
    except Exception:
        return []


def discover_eval_files() -> tuple[Path | None, Path | None]:
    """Discover evaluation files."""
    qa_candidates = [
        Path(os.getenv("RAG_QA_FILE", "")) if os.getenv("RAG_QA_FILE") else None,
        Path("gear_wear_qa.json"),
        Path("gear_wear_qa.jsonl"),
        settings.paths.DATA_DIR / "gear_wear_qa.json",
        settings.paths.DATA_DIR / "gear_wear_qa.jsonl",
        Path("RAG") / "data" / "gear_wear_qa.json",
        Path("RAG") / "data" / "gear_wear_qa.jsonl",
    ]
    gt_candidates = [
        Path(os.getenv("RAG_GT_FILE", "")) if os.getenv("RAG_GT_FILE") else None,
        Path("gear_wear_ground_truth.json"),
        Path("gear_wear_ground_truth.json"),
        settings.paths.DATA_DIR / "gear_wear_ground_truth.json",
        Path("RAG") / "data" / "ground_truth_dataset.json",
    ]
    qa = next((p for p in qa_candidates if p and p.exists() and p.is_file()), None)
    gt = next((p for p in gt_candidates if p and p.exists() and p.is_file()), None)
    return qa, gt


def normalize_ground_truth(gt_rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Return dict: question -> list[str] ground truths."""
    mapping = {}
    import re

    def _norm(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = s.strip(".,:;!?-Γאפ\u2013\u2014\"'()[]{}")
        return s

    for r in gt_rows or []:
        if not isinstance(r, dict):
            continue
        q = r.get("question") or r.get("q") or r.get("prompt") or r.get("key")
        if not q:
            continue
        gts = (
            r.get("ground_truths")
            or r.get("ground_truth")
            or r.get("answers")
            or r.get("answer")
            or r.get("value")
        )
        if gts is None:
            mapping[_norm(q)] = []
            continue
        if isinstance(gts, str):
            mapping[_norm(q)] = [gts]
        elif isinstance(gts, list):
            mapping[_norm(q)] = [str(x) for x in gts]
        else:
            mapping[_norm(q)] = [str(gts)]
    return mapping


def nan_to_none(x) -> Optional[Any]:
    """Convert NaN values to None."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, list):
        return [nan_to_none(v) for v in x]
    if isinstance(x, dict):
        return {k: nan_to_none(v) for k, v in x.items()}
    return x


def orchestrate_graph_building(docs: List) -> int:
    """Build graph database from documents if enabled."""
    try:
        if os.getenv("RAG_GRAPH_DB", "").lower() in ("1", "true", "yes"):
            from RAG.app.Gradio_apps.graphdb import build_neo4j_graph
            n = build_neo4j_graph(docs)
            get_logger().debug(f"GraphDB: Upserted ~{n} nodes/edges to Neo4j")
            return n
    except Exception as e:
        print(f"[GraphDB] population failed: {e}")
    return 0


def import_normalized_graph_data() -> int:
    """Import normalized graph data if enabled."""
    try:
        if os.getenv("RAG_IMPORT_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            from RAG.app.Gradio_apps.graphdb_import_normalized import import_normalized_graph
            
            gpath = settings.paths.LOGS_DIR / "normalized" / "graph.json"
            cpath = settings.paths.LOGS_DIR / "normalized" / "chunks.jsonl"
            
            if gpath.exists() and cpath.exists():
                n = import_normalized_graph(gpath, cpath)
                print(f"[GraphDB] normalized import result={n}")
                return n
    except Exception:
        pass
    return 0


def log_normalized_graph_summary() -> None:
    """Log summary of normalized graph if available."""
    try:
        if os.getenv("RAG_USE_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            gpath = settings.paths.LOGS_DIR / "normalized" / "graph.json"
            if gpath.exists():
                data = json.loads(gpath.read_text(encoding="utf-8"))
                print(f"[NormalizedGraph] nodes={len(data.get('nodes', []))}, edges={len(data.get('edges', []))}")
        # Optional: import normalized graph into Neo4j with page/table/figure edges
        if os.getenv("RAG_IMPORT_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            gpath = settings.paths.LOGS_DIR / "normalized" / "graph.json"
            cpath = settings.paths.LOGS_DIR / "normalized" / "chunks.jsonl"
            if gpath.exists() and cpath.exists():
                n2 = import_normalized_graph(gpath, cpath)
                print(f"[GraphDB] normalized import result={n2}")
    except Exception:
        pass



