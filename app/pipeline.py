from __future__ import annotations

"""Centralized pipeline module: ingestion -> chunking -> metadata -> indexing -> retrieval -> UI/eval.

This refactor moves the orchestration logic out of `Main.py` to keep a clean entrypoint and
separate concerns. Behavior is preserved; environment flags still control optional features.
"""

import os
import json
import math
import difflib
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple, Sequence
import warnings
# Suppress the specific FutureWarning from torch
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.")
from app.logger import trace_func, get_logger
from langchain.schema import Document

# Prefer a safe .env loader to avoid parse spam; we update os.environ manually
from dotenv import dotenv_values, find_dotenv

from app.config import settings
import app.loaders as loaders
from app.chunking import structure_chunks
from app.metadata import attach_metadata
from app.indexing import (
    build_dense_index,
    build_sparse_retriever,
    to_documents,
    dump_chroma_snapshot,
    expand_table_kv_docs,
)
from app.normalized_loader import load_normalized_docs  # Optional normalized source
from app.retrieve import (
    apply_filters,
    build_hybrid_retriever,
    query_analyzer,
    rerank_candidates,
    lexical_overlap,
)
from app.agents import (
    answer_needle,
    answer_summary,
    answer_table,
    route_question,
    route_question_ex,
)
from app.ui_gradio import build_ui
# Optional LLM-based router (safe no-op if unavailable)
try:
    from app.router_chain import route_llm  # type: ignore
except Exception:  # pragma: no cover
    def route_llm(question: str) -> str:  # type: ignore
        return "DEFAULT"
# Optional LlamaIndex export
try:
    from app.llamaindex_export import export_llamaindex_for  # type: ignore
except Exception:  # pragma: no cover
    def export_llamaindex_for(paths, out_root=None):
        return 0
try:
    from app.llamaindex_compare import build_alt_indexes  # type: ignore
except Exception:  # pragma: no cover
    def build_alt_indexes(paths, embedding_fn):
        return {}
# Optional graph visualization/database modules. Provide no-op fallbacks if missing.
try:
    from app.graph import build_graph, render_graph_html  # type: ignore
except Exception:  # pragma: no cover
    def build_graph(docs):
        return None
    def render_graph_html(G, out_path: str):
        return None
try:
    from app.graphdb import build_graph_db  # type: ignore
except Exception:  # pragma: no cover
    def build_graph_db(docs):
        return 0
try:
    from app.graphdb_import_normalized import import_normalized_graph  # type: ignore
except Exception:  # pragma: no cover
    def import_normalized_graph(graph_path, chunks_path):
        return 0
from app.validate import validate_min_pages
from app.logger import get_logger, trace_func, trace_here
from app.eval_ragas import run_eval_detailed, pretty_metrics
try:
    # Lightweight metric usable without API keys
    from app.eval_ragas import overlap_prf1  # type: ignore
except Exception:  # pragma: no cover
    overlap_prf1 = None  # type: ignore
try:
    from app.agent_orchestrator import run as run_orchestrator  # type: ignore
except Exception:
    run_orchestrator = None  # type: ignore
try:
    from app.eval_deepeval import run_eval as run_eval_deepeval  # type: ignore
except Exception:  # pragma: no cover
    def run_eval_deepeval(dataset):
        return None, []

@trace_func
def ingest_and_upsert(paths: Sequence[str | Path], dataset_id: str | None = None):
    """Incrementally ingest documents and return updated (docs, hybrid, debug).

    - Uses a single persisted Chroma store when RAG_CHROMA_DIR is set; otherwise in-memory.
    - Attaches dataset_id and source_document_id metadata during ingestion.
    - Respects existing environment flags for chunking/retrieval.
    """
    if dataset_id:
        try:
            os.environ["RAG_DATASET_ID"] = str(dataset_id)
        except Exception:
            pass
    norm: List[Path] = []
    for p in paths or []:
        try:
            norm.append(Path(str(p)))
        except Exception:
            continue
    if not norm:
        raise ValueError("No valid paths provided to ingest_and_upsert")
    return build_pipeline(norm)


@trace_func
def _clean_run_outputs() -> None:
    """Delete prior run artifacts so new extraction overwrites files.
    Cleans: data/images, data/elements, logs/queries.jsonl, logs/elements/*.jsonl
    Controlled by env RAG_CLEAN_RUN (default: true).
    """
    flag = os.getenv("RAG_CLEAN_RUN", "1").lower() not in ("0", "false", "no")
    if not flag:
        return
    import shutil
    # Directories
    for d in (Path("data") / "images", Path("data") / "elements"):
        try:
            if d.exists():
                shutil.rmtree(d)
        except Exception:
            pass
    # Do NOT auto-clean Chroma persist dir; we want a single, growing store for multi-dataset ingestion.
    # Set RAG_CLEAN_CHROMA=1 explicitly to wipe.
    try:
        if os.getenv("RAG_CLEAN_CHROMA", "0").lower() in ("1", "true", "yes"):
            chroma_dir = os.getenv("RAG_CHROMA_DIR")
            if chroma_dir:
                d = Path(chroma_dir)
                if d.exists():
                    shutil.rmtree(d)
    except Exception:
        pass
    # Logs: queries.jsonl and logs/elements dumps
    try:
        q = Path("logs") / "queries.jsonl"
        if q.exists():
            q.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        ed = Path("logs") / "elements"
        if ed.exists():
            shutil.rmtree(ed)
    except Exception:
        pass


@trace_func
def _discover_input_paths() -> List[Path]:
    """Collect input files: root Gear wear Failure.pdf and files under data/."""
    candidates: List[Path] = []
    root_pdf = Path("Gear wear Failure.pdf")
    if root_pdf.exists():
        candidates.append(root_pdf)
    if settings.DATA_DIR.exists():
        for ext in ("*.pdf", "*.docx", "*.doc", "*.txt"):
            candidates.extend(settings.DATA_DIR.glob(ext))
    return candidates


@trace_func
def _get_embeddings():
    """Prefer Google embeddings, fallback to OpenAI, then FakeEmbeddings for local smoke tests."""
    # Debug override to force local embeddings and avoid API calls
    try:
        if os.getenv("RAG_FORCE_FAKE_EMBED", "0").lower() in ("1", "true", "yes"):
            from langchain_community.embeddings import FakeEmbeddings  # type: ignore
            print("[Embeddings] Forcing FakeEmbeddings (RAG_FORCE_FAKE_EMBED=1)")
            return FakeEmbeddings(size=1536)
    except Exception:
        pass
    force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception:
        GoogleGenerativeAIEmbeddings = None  # type: ignore
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None  # type: ignore

    if force_openai and os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_OPENAI)
    if os.getenv("GOOGLE_API_KEY") and GoogleGenerativeAIEmbeddings is not None and not force_openai:
        return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_GOOGLE)
    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_OPENAI)
    # Final fallback: FakeEmbeddings
    try:
        from langchain_community.embeddings import FakeEmbeddings  # type: ignore
        print("[Embeddings] Using FakeEmbeddings fallback (no API keys found)")
        return FakeEmbeddings(size=1536)
    except Exception:
        pass
    raise RuntimeError(
        "No embedding backend available. Set GOOGLE_API_KEY or OPENAI_API_KEY, or ensure langchain_community FakeEmbeddings is available."
    )


class _LLM:
    """Simple callable LLM wrapper preferring Gemini; fallback to OpenAI via LangChain chat models."""
    @trace_func
    def __init__(self) -> None:
        self._backend = None
        self._which = None
        force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
        # Prefer Google Gemini unless forced OpenAI
        if os.getenv("GOOGLE_API_KEY") and not force_openai:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Prefer a more faithful model with deterministic generation
                self._backend = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)
                self._which = "google"
            except Exception:
                self._backend = None
        # Fallback to OpenAI (or forced)
        if (self._backend is None or force_openai) and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-nano")
                try:
                    self._backend = ChatOpenAI(model=model, temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
                except Exception:
                    try:
                        self._backend = ChatOpenAI(model_name=model, temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
                    except Exception:
                        self._backend = ChatOpenAI(model=model, temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
                self._which = "openai"
            except Exception:
                self._backend = None

    @trace_func
    def __call__(self, prompt: str) -> str:
        if self._backend is not None:
            try:
                resp = self._backend.invoke(prompt)
                return getattr(resp, "content", str(resp))
            except Exception as e:  # pragma: no cover
                return f"[LLM error] {e}\n\n{prompt[-400:]}"
        return "[LLM not configured] " + prompt[-400:]

@trace_func
def _count_pdf_pages(path: Path) -> int:
    """Best-effort page count using PyMuPDF or pdfplumber; fallback to 0."""
    try:
        import fitz  # type: ignore
        try:
            with fitz.open(str(path)) as doc:  # type: ignore
                return int(len(doc))
        except Exception:
            pass
    except Exception:
        pass
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path) as pdf:  # type: ignore
            return int(len(pdf.pages))
    except Exception:
        pass
    return 0


@trace_func
def build_pipeline(paths: List[Path]):
    """Ingest documents, build chunks+metadata, and initialize hybrid retriever."""
    log = get_logger()
    # Log high-level knobs once to aid debugging small chunk counts
    try:
        log.info(
            "FLAGS[chunking]: MULTI=%s SEMANTIC=%s TARGET_TOK=%s MAX_TOK=%s OVERLAP_N=%s HEADING_MIN_FONT=%s",
            os.getenv("RAG_TEXT_SPLIT_MULTI", "1"),
            os.getenv("RAG_SEMANTIC_CHUNKING", "1"),
            os.getenv("RAG_TEXT_TARGET_TOKENS", "250"),
            os.getenv("RAG_TEXT_MAX_TOKENS", "500"),
            os.getenv("RAG_TEXT_OVERLAP_SENTENCES", "1"),
            os.getenv("RAG_HEADING_MIN_FONT", "12"),
        )
        log.info(
            "FLAGS[retrieval]: DENSE_K=%d SPARSE_K=%d CONTEXT_TOP_N=%d CE_RERANK=%s GRAPH_DB=%s",
            settings.DENSE_K,
            settings.SPARSE_K,
            settings.CONTEXT_TOP_N,
            os.getenv("RAG_USE_CE_RERANKER", "0"),
            os.getenv("RAG_GRAPH_DB", "1"),
        )
    except Exception:
        pass
    records = []
    # Dataset/document scoping
    dataset_id_env = os.getenv("RAG_DATASET_ID") or os.getenv("DATASET_ID") or None
    # ingest
    for pair in loaders.load_many(paths):
        try:
            path, elements = pair
        except Exception:
            # Fallback in case of unexpected return shapes
            continue
        # basic ingestion validation: min pages
        try:
            page_count = _count_pdf_pages(Path(path))
            ok, msg = validate_min_pages(page_count, settings.MIN_PAGES)
            if not ok:
                print(f"[WARN] {path.name}: {msg}")
            else:
                get_logger().info("%s: page_count=%d", path.name, page_count)
        except Exception:
            pass
        # Structure-aware chunking (semantic+multi by default; see FLAGS above)
        chunks = structure_chunks(elements, str(path))
        # Compute deterministic source_document_id from normalized absolute path
        try:
            apath = str(Path(path).resolve())
        except Exception:
            apath = str(path)
        import hashlib as _hl
        src_id = _hl.sha256(apath.encode("utf-8", errors="ignore")).hexdigest()[:16]
        # Resolve dataset_id: prefer explicit env, else default to per-file stem
        try:
            dataset_id_val = dataset_id_env or Path(path).stem
        except Exception:
            dataset_id_val = dataset_id_env or str(getattr(path, "stem", path))
        for ch in chunks:
            records.append(
                attach_metadata(
                    ch,
                    client_id=os.getenv("CLIENT_ID"),
                    case_id=path.stem,
                    dataset_id=dataset_id_val,
                    source_document_id=src_id,
                    file_path=apath,
                )
            )
    # Section histogram after metadata attachment
    sec_hist = {}
    for r in records:
        sec = (r.get("metadata", {}) or {}).get("section")
        sec_hist[sec] = sec_hist.get(sec, 0) + 1
    if sec_hist:
        log.info("Section histogram: %s", sorted(sec_hist.items(), key=lambda x: (-x[1], str(x[0]))))
    # vectorization
    # Optional: prefer normalized chunks.jsonl if feature flag enabled
    use_normalized = os.getenv("RAG_USE_NORMALIZED", "0").lower() in ("1", "true", "yes")
    if use_normalized and (Path("logs") / "normalized" / "chunks.jsonl").exists():
        docs = load_normalized_docs(Path("logs") / "normalized" / "chunks.jsonl")
        log.info("Using normalized docs for indexing: %d", len(docs))
    else:
        docs = to_documents(records)
    # Normalize file_name for display: prefer basename of file_path
    try:
        for d in docs:
            md = d.metadata or {}
            fp = md.get("file_path") or md.get("file") or md.get("file_name")
            if fp:
                md["file_name"] = Path(str(fp)).name
                d.metadata = md
    except Exception:
        pass
    # Optional: expand table rows into KV mini-docs to improve retrieval of specific values
    try:
        if os.getenv("RAG_EXPAND_TABLE_KV", "1").lower() in ("1", "true", "yes"):
            docs = expand_table_kv_docs(docs)
    except Exception:
        pass
    # Write DB snapshots for debugging
    try:
        Path("logs").mkdir(exist_ok=True)
        snap_path = Path("logs") / "db_snapshot.jsonl"
        full_snap_path = Path("logs") / "db_snapshot_full.jsonl"
        # Append mode so multiple ingests accumulate; dedupe is best-effort and can be improved later
        with open(snap_path, "a", encoding="utf-8") as f, open(full_snap_path, "a", encoding="utf-8") as f_full:
            # per-source snapshot file (helps compare per document ingests)
            try:
                per_src_dir = Path("logs") / "elements"
                per_src_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                per_src_dir = None  # type: ignore
            for d in docs:
                md = d.metadata or {}
                txt = d.page_content or ""
                sec = md.get("section") or md.get("section_type")
                # Build a stable, human-oriented preview
                preview_str = ""
                try:
                    lines = (txt or "").splitlines()
                    if sec == "Figure":
                        # Prefer normalized label (e.g., "Figure N: ...")
                        preview_str = md.get("figure_label") or ""
                        if not preview_str:
                            # Extract CAPTION line
                            cap = None
                            for i, ln in enumerate(lines):
                                if ln.strip().upper() == "CAPTION:" and i + 1 < len(lines):
                                    cap = lines[i + 1].strip()
                                    break
                            preview_str = cap or (lines[0].strip() if lines else "")
                    elif sec == "Table":
                        preview_str = md.get("table_label") or ""
                        if not preview_str:
                            table_no = md.get("table_number")
                            summ = None
                            for i, ln in enumerate(lines):
                                if ln.strip().upper() == "SUMMARY:" and i + 1 < len(lines):
                                    summ = lines[i + 1].strip()
                                    break
                            if summ:
                                preview_str = f"Table {table_no}: {summ}" if table_no is not None else summ
                            else:
                                preview_str = lines[0].strip() if lines else ""
                    else:
                        preview_str = (txt or "")[:200]
                except Exception:
                    preview_str = (txt or "")[:200]
                rec = {
                    "file": md.get("file_name"),
                    "file_path": md.get("file_path"),
                    "page": md.get("page"),
                    "section": md.get("section"),
                    "anchor": md.get("anchor"),
                    # Deterministic IDs for traceability
                    "doc_id": md.get("doc_id"),
                    "chunk_id": md.get("chunk_id"),
                    "content_hash": md.get("content_hash"),
                    # Multi-dataset
                    "dataset_id": md.get("dataset_id"),
                    "source_document_id": md.get("source_document_id"),
                    # Table metadata
                    "table_md_path": md.get("table_md_path"),
                    "table_csv_path": md.get("table_csv_path"),
                    "table_number": md.get("table_number"),
                    "table_label": md.get("table_label"),
                    "table_associated_text_preview": md.get("table_associated_text_preview"),
                    "table_associated_anchor": md.get("table_associated_anchor"),
                    # Figure metadata
                    "image_path": md.get("image_path"),
                    "figure_number": md.get("figure_number"),
                    "figure_order": md.get("figure_order"),
                    "figure_label": md.get("figure_label"),
                    "figure_associated_text_preview": md.get("figure_associated_text_preview"),
                    "figure_associated_anchor": md.get("figure_associated_anchor"),
                    "words": len((d.page_content or "").split()),
                    "preview": preview_str,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Full, non-normalized snapshot record (text + full metadata)
                try:
                    full_rec = {
                        "file": md.get("file_name"),
                        "file_path": md.get("file_path"),
                        "page": md.get("page"),
                        "section": md.get("section") or md.get("section_type"),
                        "anchor": md.get("anchor"),
                        "doc_id": md.get("doc_id"),
                        "chunk_id": md.get("chunk_id"),
                        "content_hash": md.get("content_hash"),
                        "dataset_id": md.get("dataset_id"),
                        "source_document_id": md.get("source_document_id"),
                        "metadata": md,  # entire metadata blob for offline inspection
                        "text": txt,     # full chunk text (untruncated)
                        "words": len((txt or "").split()),
                    }
                    f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")
                    # Write per-source file snapshot as well
                    try:
                        if per_src_dir is not None:
                            ds_id = (md.get("dataset_id") or "unknown_ds").strip()
                            src_id = (md.get("source_document_id") or "unknown_src").strip()
                            per_src = per_src_dir / f"snap_{ds_id}__{src_id}.jsonl"
                            with open(per_src, "a", encoding="utf-8") as _ps:
                                _ps.write(json.dumps(full_rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                except Exception:
                    # Best-effort: if serialization fails, write a minimal fallback
                    try:
                        f_full.write(json.dumps({
                            "file": md.get("file_name"),
                            "page": md.get("page"),
                            "section": md.get("section") or md.get("section_type"),
                            "anchor": md.get("anchor"),
                            "doc_id": md.get("doc_id"),
                            "chunk_id": md.get("chunk_id"),
                            "content_hash": md.get("content_hash"),
                            "text": txt,
                        }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
    except Exception:
        pass
    tbl_cnt = sum(1 for d in docs if (d.metadata or {}).get("section") == "Table")
    fig_cnt = sum(1 for d in docs if (d.metadata or {}).get("section") == "Figure")
    log.info(
        "Ingestion complete: %d files -> %d chunks -> %d documents (tables=%d, figures=%d)",
        len(paths),
        len(records),
        len(docs),
        tbl_cnt,
        fig_cnt,
    )
    embeddings = _get_embeddings()
    dense = build_dense_index(docs, embeddings)
    sparse = build_sparse_retriever(docs, k=settings.SPARSE_K)
    hybrid = build_hybrid_retriever(dense, sparse, dense_k=settings.DENSE_K)
    # Optional DB tracing
    try:
        if os.getenv("RAG_TRACE_DB", "0").lower() in ("1", "true", "yes"):
            log.info("DB_TRACE: backend=%s | dense_k=%d sparse_k=%d | docs=%d", os.getenv("RAG_VECTOR_BACKEND", "chroma"), settings.DENSE_K, settings.SPARSE_K, len(docs))
    except Exception:
        pass
    # If Chroma is persisted, try writing a snapshot
    try:
        if os.getenv("RAG_CHROMA_DIR"):
            dump_chroma_snapshot(dense, Path("logs") / "chroma_snapshot.jsonl")
    except Exception:
        pass
    # expose per-retriever diagnostics
    try:
        dense_ret = dense.as_retriever(search_kwargs={"k": settings.DENSE_K})
    except Exception:
        dense_ret = None
    debug = {"dense": dense_ret, "sparse": sparse}
    # Populate graph database from current docs by default (opt-out via RAG_GRAPH_DB=0/false)
    try:
        if os.getenv("RAG_GRAPH_DB", "1").lower() not in ("0", "false", "no"):
            n = build_graph_db(docs)
            print(f"[GraphDB] Upserted ~{n} nodes/edges to Neo4j")
    except Exception as e:
        print(f"[GraphDB] population failed: {e}")

    # Optional: if normalized graph.json exists and flag enabled, display a quick summary log
    try:
        if os.getenv("RAG_USE_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            gpath = Path("logs") / "normalized" / "graph.json"
            if gpath.exists():
                data = json.loads(gpath.read_text(encoding="utf-8"))
                print(f"[NormalizedGraph] nodes={len(data.get('nodes', []))}, edges={len(data.get('edges', []))}")
        # Optional: import normalized graph into Neo4j with page/table/figure edges
        if os.getenv("RAG_IMPORT_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            gpath = Path("logs") / "normalized" / "graph.json"
            cpath = Path("logs") / "normalized" / "chunks.jsonl"
            if gpath.exists() and cpath.exists():
                n2 = import_normalized_graph(gpath, cpath)
                print(f"[GraphDB] normalized import result={n2}")
    except Exception:
        pass
    # Build alternate indexes for comparison (LlamaIndex exports and/or LlamaParse)
    alt = {}
    try:
        alt = build_alt_indexes(paths, embeddings)
        # Summarize alt indexes
        for key, obj in (alt or {}).items():
            try:
                get_logger().info("ALT[%s]: docs=%d | dense=%s | sparse=%s", key, len(obj.get("docs", [])), type(obj.get("dense")).__name__, type(obj.get("sparse")).__name__)
            except Exception:
                pass
        # Dump Chroma snapshots for alt dense stores
        try:
            for key, obj in (alt or {}).items():
                vs = obj.get("dense")
                if vs is not None:
                    dump_chroma_snapshot(vs, Path("logs") / f"chroma_snapshot_{key}.jsonl")
        except Exception:
            pass
    except Exception:
        alt = {}
    return docs, hybrid, {**debug, "alt": alt}

@trace_func
def ask(docs, hybrid, llm: _LLM, question: str, ground_truth: str | None = None) -> str:
    """Answer a user question using the hybrid retriever and route to sub-agents."""
    log = get_logger()
    trace_here("ask")
    
    # Generate unique trace ID for this query
    import uuid
    trace_id = str(uuid.uuid4())[:8]
    try:
        qprev_lim = int(os.getenv("RAG_MAX_QUERY_PREVIEW_CHARS", "500"))
    except Exception:
        qprev_lim = 500
    q_prev = (question or "") if qprev_lim <= 0 else (question or "")[:qprev_lim]
    log.info(f"QUERY_START: trace_id={trace_id} question='{q_prev}...'")
    
    qa = query_analyzer(question)
    q_exec = qa.get("canonical") or question
    
    log.debug(f"QUERY_ANALYSIS: trace_id={trace_id} canonical='{q_exec}' keywords={qa['keywords']} filters={qa['filters']}")
    
    try:
        candidates = hybrid.invoke(q_exec)
    except Exception:
        candidates = hybrid.invoke(q_exec)
    # Optional: scope by filename via env to remove cross-doc noise during focused evals
    try:
        scope_file = os.getenv("RAG_FILE_SCOPE", "").strip()
        if scope_file:
            strict = os.getenv("RAG_FILE_SCOPE_STRICT", "0").lower() in ("1","true","yes")
            def _ok(d):
                md = getattr(d, "metadata", {}) or {}
                fn = md.get("file_name") or md.get("file_path") or ""
                return (str(fn).lower().endswith(scope_file.lower()) if strict else (scope_file.lower() in str(fn).lower()))
            candidates = [d for d in candidates if _ok(d)]
    except Exception:
        pass
    candidates = candidates[: settings.K_TOP_K]  # rerank TOP K
    
    log.debug(f"RETRIEVAL: trace_id={trace_id} candidates={len(candidates)}")
    
    filtered = apply_filters(candidates, qa["filters"])  # metadata filters
    try:
        sec = qa["filters"].get("section")
    except Exception:
        sec = None
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
    
    log.debug(f"FILTERING: trace_id={trace_id} filtered={len(filtered)} section_hint={sec}")
    
    top_docs = rerank_candidates(q_exec, filtered, top_n=settings.CONTEXT_TOP_N)
    # Fail-closed when no context for factual/table questions
    if not top_docs and (qa.get("intent", {}).get("needs_facts") or qa.get("filters", {}).get("section") in ("Table","Figure")):
        log.info(f"AGENT_COMPLETE: trace_id={trace_id} agent=guard answer_length=0 (empty context)")
        return "Not found in context."
    
    log.debug(f"RERANKING: trace_id={trace_id} top_docs={len(top_docs)}")
    
    # Prefer LLM router when enabled; fall back to heuristic router
    route = route_llm(question)
    router_source = "llm"
    if route == "DEFAULT":
        route, rtrace = route_question_ex(question)
        router_source = "heuristic"
    else:
        rtrace = {"matched": ["llm_router"], "route": route, "simplified": qa.get("intent", {})}

    log.info(f"ROUTING: trace_id={trace_id} route={route} router={router_source}")

    def _doc_head(d):
        md = getattr(d, "metadata", {}) or {}
        return f"{md.get('file_name')} p{md.get('page')} {md.get('section')}#{md.get('anchor', '')}"

    def _score(d):
        base = lexical_overlap(q_exec, d.page_content)
        meta_text = " ".join(map(str, (getattr(d, "metadata", {}) or {}).values()))
        boost = 0.2 * lexical_overlap(" ".join(qa["keywords"]), meta_text)
        return round(base + boost, 4)

    log.info(
        "Q: %s | route=%s | canonical=%s | keywords=%s | filters=%s | pool=%d | filtered=%d | using=%d",
        q_exec,
        route,
        qa.get("canonical"),
        qa["keywords"],
        qa["filters"],
        len(candidates),
        len(filtered),
        len(top_docs),
    )
    try:
        # Log a compact list of candidate heads before filtering
        heads = []
        for d in candidates[:20]:
            md = getattr(d, "metadata", {}) or {}
            heads.append(f"{md.get('file_name')}#p{md.get('page')} {md.get('section')}#{md.get('anchor')}")
        if heads:
            log.debug("CANDIDATES: %s", "; ".join(heads))
    except Exception:
        pass
    for i, d in enumerate(top_docs, start=1):
        log.info("ctx[%d] score=%.4f | %s", i, _score(d), _doc_head(d))
    
    # Orchestrator: prefer orchestrated answer by default (with trace for transparency)
    reasoning_trace = None
    ans = None
    agent_used = None
    
    try:
        if run_orchestrator is not None and os.getenv("RAG_USE_ORCHESTRATOR", "1").lower() in ("1","true","yes"):
            log.debug(f"AGENT_START: trace_id={trace_id} agent=orchestrator")
            reasoning_trace = run_orchestrator(question, docs, hybrid, _LLM(), do_answer=True)
            try:
                if isinstance(reasoning_trace, dict):
                    ans = reasoning_trace.get("answer") or None
                    agent_used = "orchestrator"
                    # If orchestrator selected a route, prefer it for logging
                    if reasoning_trace.get("route"):
                        route = reasoning_trace.get("route") or route
            except Exception:
                pass
    except Exception:
        reasoning_trace = None
    
    # Fallback to route-based agents if orchestrator didn't answer
    if not ans:
        log.debug(f"AGENT_START: trace_id={trace_id} agent={route}")
        if route == "summary":
            ans = answer_summary(_LLM(), top_docs, question)
            agent_used = "summary"
        elif route == "table" or route == "graph":  # temporary: route graph to table agent until dedicated graph agent is added
            ans = answer_table(_LLM(), top_docs, question)
            agent_used = "table"
        else:
            ans = answer_needle(_LLM(), top_docs, question)
            agent_used = "needle"
    # Guard against non-cited answers on factual routes
    try:
        if agent_used in ("needle","table") and (not ans or ("[" not in ans and "]" not in ans)):
            ans = "Not found in context."
    except Exception:
        pass
    
    log.info(f"AGENT_COMPLETE: trace_id={trace_id} agent={agent_used} answer_length={len(ans or '')}")
    
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "trace_id": trace_id,
            "question": question,
            "route": route,
            "router_source": router_source,
            "agent_used": agent_used,
            "router_trace": rtrace,
            "keywords": qa["keywords"],
            "filters": qa["filters"],
            "contexts": [
                {
                    "file": d.metadata.get("file_name"),
                    "page": d.metadata.get("page"),
                    "section": d.metadata.get("section"),
                    "anchor": d.metadata.get("anchor"),
                    "score": _score(d),
                }
                for d in top_docs
            ],
            "answer_preview": (ans or ""),
            "reasoning_trace": reasoning_trace,
        }
        with open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return ans


@trace_func
def answer_with_contexts(docs, hybrid, llm: _LLM, question: str):
    """Answer a question and also return the contexts used (top_docs) and the reasoning trace when available.

    Returns: tuple(answer, top_docs, reasoning_trace)
    """
    log = get_logger()
    qa = query_analyzer(question)
    q_exec = qa.get("canonical") or question
    try:
        candidates = hybrid.invoke(q_exec)
    except Exception:
        candidates = hybrid.invoke(q_exec)
    # Optional file scoping to contain noise during specific benchmarks
    try:
        scope_file = os.getenv("RAG_FILE_SCOPE", "").strip()
        if scope_file:
            strict = os.getenv("RAG_FILE_SCOPE_STRICT", "0").lower() in ("1","true","yes")
            def _ok(d):
                md = getattr(d, "metadata", {}) or {}
                fn = md.get("file_name") or md.get("file_path") or ""
                return (str(fn).lower().endswith(scope_file.lower()) if strict else (scope_file.lower() in str(fn).lower()))
            candidates = [d for d in candidates if _ok(d)]
    except Exception:
        pass
    candidates = candidates[: settings.K_TOP_K]
    filtered = apply_filters(candidates, qa["filters"])  # type: ignore[index]
    try:
        sec = qa["filters"].get("section")  # type: ignore[index]
    except Exception:
        sec = None
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
    top_docs = rerank_candidates(q_exec, filtered, top_n=settings.CONTEXT_TOP_N)
    if not top_docs:
        top_docs = candidates[: settings.CONTEXT_TOP_N] if candidates else []
    if not top_docs:
        top_docs = docs[: settings.CONTEXT_TOP_N]
    # Hard stop answer when no context for factual/table asks
    try:
        qa = query_analyzer(question)
        if not top_docs and (qa.get("intent", {}).get("needs_facts") or (qa.get("filters", {}) or {}).get("section") in ("Table","Figure")):
            return "Not found in context.", top_docs, None
    except Exception:
        pass
    # Prefer orchestrator for answering when enabled, fallback to route-based agents
    ans = None
    trace = None
    try:
        if run_orchestrator is not None and os.getenv("RAG_USE_ORCHESTRATOR", "1").lower() in ("1","true","yes"):
            tr = run_orchestrator(question, docs, hybrid, llm, do_answer=True)
            if isinstance(tr, dict):
                ans = tr.get("answer") or None
                trace = tr
    except Exception:
        ans = None
        trace = None
    if not ans:
        # Use LLM router with fallback to heuristic
        route = route_llm(question)
        if route == "DEFAULT":
            route = route_question(question)
        if route == "summary":
            ans = answer_summary(llm, top_docs, question)
        elif route == "table" or route == "graph":
            ans = answer_table(llm, top_docs, question)
        else:
            ans = answer_needle(llm, top_docs, question)
    return ans, top_docs, trace


@trace_func
def _load_json_or_jsonl(path: Path):
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


@trace_func
def _discover_eval_files():
    # Only load the two context_free files provided (unless explicit env overrides are valid files)
    qa_env = os.getenv("RAG_QA_FILE", "").strip()
    gt_env = os.getenv("RAG_GT_FILE", "").strip()
    qa_override = Path(qa_env) if qa_env else None
    gt_override = Path(gt_env) if gt_env else None
    # Require files, not directories (avoid Path("") -> ".")
    if qa_override is not None and not qa_override.is_file():
        qa_override = None
    if gt_override is not None and not gt_override.is_file():
        gt_override = None
    qa = qa_override if qa_override is not None else (Path("data") / "gear_wear_qa_context_free.jsonl")
    gt = gt_override if gt_override is not None else (Path("data") / "gear_wear_ground_truth_context_free.json")
    qa = qa if qa.exists() and qa.is_file() else None
    gt = gt if gt.exists() and gt.is_file() else None
    return qa, gt


@trace_func
def _normalize_ground_truth(gt_rows):
    """Return dict: question -> list[str] ground truths."""
    mapping = {}
    import re

    def _norm(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = s.strip(".,:;!?-—\u2013\u2014\"'()[]{}")
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


@trace_func
def _index_ground_truth(gt_rows):
    """Build two maps: by_id and by_question for flexible GT matching."""
    by_id: dict[str, list[str]] = {}
    by_q: dict[str, list[str]] = {}
    import re as _re

    def _norm(s: str) -> str:
        s = str(s).lower().strip()
        s = _re.sub(r"\s+", " ", s)
        s = s.strip(".,:;!?—-\u2013\u2014\"'()[]{}")
        return s

    for r in gt_rows or []:
        if not isinstance(r, dict):
            continue
        rid = r.get("id") or r.get("qid") or r.get("question_id") or r.get("key")
        q = r.get("question") or r.get("q") or r.get("prompt")
        gts = (
            r.get("ground_truths")
            or r.get("ground_truth")
            or r.get("answers")
            or r.get("answer")
            or r.get("value")
        )
        if gts is None:
            vals: list[str] = []
        elif isinstance(gts, str):
            vals = [gts]
        elif isinstance(gts, list):
            vals = [str(x) for x in gts]
        else:
            vals = [str(gts)]
        if rid:
            by_id[str(rid)] = vals
        if q:
            by_q[_norm(q)] = vals
    return by_id, by_q


@trace_func
def run_evaluation(docs, hybrid, llm: _LLM):
    log = get_logger()
    qa_path, gt_path = _discover_eval_files()
    # Diagnostics for file discovery
    try:
        log.info(
            "EVAL files: QA=%s (exists=%s) | GT=%s (exists=%s)",
            str(qa_path) if qa_path else "<none>",
            (qa_path.exists() if qa_path else False),
            str(gt_path) if gt_path else "<none>",
            (gt_path.exists() if gt_path else False),
        )
    except Exception:
        pass
    if not qa_path:
        log.warning("Evaluation requested but QA file not found.")
        return
    # Prefer Google for RAGAS if available; do not force OpenAI provider
    try:
        if os.getenv("GOOGLE_API_KEY") and not os.getenv("RAGAS_LLM_PROVIDER"):
            os.environ.setdefault("RAGAS_LLM_PROVIDER", "google")
    except Exception:
        pass
    qa_rows = _load_json_or_jsonl(qa_path)
    try:
        log.info("QA auto-load: Loaded %d QA items from %s", len(qa_rows or []), str(qa_path))
    except Exception:
        pass
    gt_rows = _load_json_or_jsonl(gt_path) if gt_path else []
    gt_by_id, gt_by_q = _index_ground_truth(gt_rows)
    # Fix for empty questions in ground truth
    if "" in gt_by_q:
        del gt_by_q[""]
    try:
        log.info(
            "GT auto-load: Loaded %d ids and %d questions from %s. Sample ids: %s",
            len(gt_by_id), len(gt_by_q), str(gt_path) if gt_path else "<none>", ", ".join(list(gt_by_id.keys())[:5])
        )
    except Exception:
        pass
    rows_out = []
    any_gt = False
    for i, row in enumerate(qa_rows, start=1):
        if not isinstance(row, dict):
            continue
        q = row.get("question") or row.get("q") or row.get("prompt") or row.get("text")
        qid = row.get("id") or row.get("qid") or row.get("question_id") or row.get("key")
        if not q:
            continue
        try:
            ans, ctx_docs, tr = answer_with_contexts(docs, hybrid, llm, q)
        except Exception:
            continue
        ctxs = [getattr(d, "page_content", "") for d in (ctx_docs or []) if getattr(d, "page_content", None)]
        if not ctxs:
            ctxs = [getattr(docs[0], "page_content", "")] if docs else [""]
        norm_q = str(q).lower().strip()
        norm_q = " ".join(norm_q.split())
        norm_q = norm_q.strip(".,:;!?-—\u2013\u2014\"'()[]{}")
        # Prefer ID match, then question match
        gts = []
        if qid and str(qid) in gt_by_id:
            gts = gt_by_id.get(str(qid), [])
        if (not gts) and norm_q in gt_by_q:
            gts = gt_by_q.get(norm_q, [])
        if not gts and gt_by_q:
            keys = list(gt_by_q.keys())
            best = None
            best_score = 0.0
            for k in keys:
                s = difflib.SequenceMatcher(None, norm_q, k).ratio()
                if s > best_score:
                    best_score = s
                    best = k
            if best is not None and best_score >= 0.75:
                gts = gt_by_q.get(best, [])
        if (not gts) and isinstance(row.get("answer"), (str, int, float)):
            ans_txt = str(row["answer"]).strip()
            if ans_txt:
                gts = [ans_txt]
        if gts:
            any_gt = True
        ref = gts[0] if isinstance(gts, list) and gts else ""
        rec = {
            "question": q,
            "answer": ans or "",
            "contexts": ctxs,
            "ground_truths": gts,
            "reference": ref,
            "reasoning_trace": tr,
        }
        # Heuristic label drift detector: if ref token not dominant in contexts, flag it
        try:
            if ref:
                import re as _re
                tok = max(_re.findall(r"\b([A-Za-z]{3,}[0-9A-Za-z\-]{0,})\b", str(ref)), key=len, default=None)
                if tok:
                    ctx_text = "\n".join(ctxs or [])
                    hits_ref = len(_re.findall(rf"\b{_re.escape(tok)}\b", ctx_text, _re.I))
                    # Look for common alternate tokens present in industrial sensors
                    alt_candidates = ["dytran", "pcb", "352c33", "3053b"]
                    best_alt = None; best_count = 0
                    for alt in alt_candidates:
                        if alt.lower() == str(tok).lower():
                            continue
                        c = len(_re.findall(rf"\b{_re.escape(alt)}\b", ctx_text, _re.I))
                        if c > best_count:
                            best_count = c; best_alt = alt
                    if best_alt and best_count > hits_ref:
                        rec["label_drift"] = True
                        rec["label_drift_note"] = f"Context favors '{best_alt}' over reference token '{tok}'."
        except Exception:
            pass
        rows_out.append(rec)
        try:
            if os.getenv("RAG_TRACE_EVAL", "0").lower() in ("1", "true", "yes"):
                log.info("EVAL_TRACE[%d]: qid=%s | gt_found=%s | gt_count=%d | ctx_len=%d", i, str(qid), bool(gts), len(gts or []), len(ctxs or []))
        except Exception:
            pass
        try:
            log.info("EVAL Q[%d]: %s", i, q)
        except Exception:
            pass
    if not rows_out:
        print("No evaluation rows to process.")
        try:
            log.warning("EVAL skipped: QA rows=%d | QA=%s | GT=%s", len(qa_rows or []), str(qa_path), str(gt_path))
        except Exception:
            pass
        return
    ds = {"question": [], "answer": [], "contexts": [], "reference": [], "ground_truths": [], "reasoning_trace": []}
    for r in rows_out:
        ds["question"].append(r["question"])
        ds["answer"].append(r["answer"])
        ds["contexts"].append(r["contexts"])
        ds["reference"].append(r.get("reference", ""))
        ds["ground_truths"].append(r.get("ground_truths", []))  # type: ignore[index]
        ds["reasoning_trace"].append(r.get("reasoning_trace"))
    # --- Run Evaluations ---
    summary, per_q = {}, []
    ragas_was_run = False
    if os.getenv("RAG_SKIP_RAGAS", "0").lower() not in ("1", "true", "yes"):
        try:
            summary, per_q = run_eval_detailed(ds)
            ragas_was_run = True
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            summary, per_q = {}, []
    else:
        print("Skipping RAGAS evaluation (RAG_SKIP_RAGAS=1)")

    # Optional: run DeepEval side-by-side
    de_sum, de_rows = None, []
    deepeval_was_run = False
    if os.getenv("RAG_DEEPEVAL", "0").lower() in ("1", "true", "yes"):
        try:
            de_sum, de_rows = run_eval_deepeval(ds)
            if de_sum:
                deepeval_was_run = True
                print("\nDeepEval summary:")
                print(json.dumps(de_sum, indent=2))
        except Exception as e:
            log.warning("DeepEval run failed: %s", e)

    # If RAGAS was skipped, use DeepEval results as the main summary
    if not ragas_was_run and deepeval_was_run:
        summary = de_sum
        # Merge DeepEval's per-question details into the main `per_q` structure
        q_map = {row.get("question"): row for row in per_q}
        for de_row in de_rows:
            q = de_row.get("question")
            if q in q_map:
                q_map[q].update(de_row)
            else:
                # This case is unlikely if dataset is the same, but handle it
                per_q.append(de_row)

    # Fallback: if both RAGAS and DeepEval were skipped or failed, still emit useful outputs
    if (not ragas_was_run) and (not deepeval_was_run):
        try:
            # Build per-question rows with lightweight overlap metrics (no API calls)
            ref_list = list(ds.get("reference") or [])
            ctx_list = list(ds.get("contexts") or [])
            q_list = list(ds.get("question") or [])
            a_list = list(ds.get("answer") or [])
            n = max(len(q_list), len(a_list), len(ref_list))
            per_q = []
            for i in range(n):
                q = q_list[i] if i < len(q_list) else None
                a = a_list[i] if i < len(a_list) else None
                ref = ref_list[i] if i < len(ref_list) else None
                ctxs = ctx_list[i] if i < len(ctx_list) else []
                row = {"question": q, "answer": a, "reference": ref, "contexts": ctxs}
                # Compute simple token overlap if helper available
                try:
                    if overlap_prf1 is not None:
                        p, r, f1 = overlap_prf1(str(ref or ""), list(ctxs or []))  # type: ignore[arg-type]
                        row.update({
                            "overlap_precision": p,
                            "overlap_recall": r,
                            "overlap_f1": f1,
                        })
                except Exception:
                    pass
                row["note"] = "No eval backend enabled (RAGAS skipped, DeepEval unavailable). Lightweight overlap-only metrics shown."
                per_q.append(row)
            # Summary: averages over overlap metrics when present
            def _mean_safe(vals):
                nums = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
                return float(sum(nums) / len(nums)) if nums else None
            summary = {
                "overlap_precision": _mean_safe([r.get("overlap_precision") for r in per_q]),
                "overlap_recall": _mean_safe([r.get("overlap_recall") for r in per_q]),
                "overlap_f1": _mean_safe([r.get("overlap_f1") for r in per_q]),
                "items": len(per_q),
                "note": "Evaluation backends disabled; computed token overlap vs contexts only.",
            }
        except Exception:
            # keep empty summary/per_q but still persist files below
            pass

    # --- Save and Print Final Results ---
    def _nan_to_none(x):
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, list):
            return [_nan_to_none(v) for v in x]
        if isinstance(x, dict):
            return {k: _nan_to_none(v) for k, v in x.items()}
        return x

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    
    summary_path = out_dir / "eval_summary.json"
    per_q_path = out_dir / "eval_per_question.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(summary), f, ensure_ascii=False, indent=2, allow_nan=False)

    with open(per_q_path, "w", encoding="utf-8") as f:
        if per_q:
            for rec in per_q:
                f.write(json.dumps(_nan_to_none(rec), ensure_ascii=False, allow_nan=False) + "\n")
            
            s = _nan_to_none(summary)
            if s:
                if isinstance(s, dict):
                    footer = {"__summary__": True, **s}
                else:
                    footer = {"__summary__": True, "summary": s}
                f.write(json.dumps(footer, ensure_ascii=False, allow_nan=False) + "\n")

    log = get_logger()
    print("\n--- Evaluation Summary ---")
    if summary:
        summ_str = pretty_metrics(summary)
        print(summ_str)
        log.info("Evaluation summary (averaged over %d items):\n%s", len(rows_out), summ_str)
    else:
        print("No evaluation metrics calculated.")
        log.info("No evaluation metrics calculated.")

    log.info("Saved summary to %s and per-question to %s", str(summary_path), str(per_q_path))
    print("\nPer-question results:")
    try:
        # Allow very long answers in console; clamp only if explicitly configured
        max_chars_env = os.getenv("RAG_MAX_PRINT_ANSWER_CHARS", "5000")
        try:
            max_chars = int(max_chars_env)
        except Exception:
            max_chars = 5000
        for rec in per_q:
            q = rec.get("question", "")
            ans = rec.get("answer", "")
            mets = {k: v for k, v in rec.items() if k not in ("question", "answer", "contexts", "ground_truths")}
            # Sanitize any NaN values for console printing as well
            def _nan_to_none_local(x):
                import math as _m
                if isinstance(x, float) and _m.isnan(x):
                    return None
                if isinstance(x, list):
                    return [_nan_to_none_local(v) for v in x]
                if isinstance(x, dict):
                    return {k: _nan_to_none_local(v) for k, v in x.items()}
                return x
            print("- Q:", q)
            if isinstance(max_chars, int) and max_chars > 0:
                print("  A:", (ans or "")[:max_chars])
            else:
                print("  A:", (ans or ""))
            print("  metrics:", json.dumps(_nan_to_none_local(mets), ensure_ascii=False))
    except Exception:
        pass


@trace_func
def run() -> None:
    """Entry point that mirrors the prior Main.main() behavior."""
    print("stating program")
    # Prevent third-party libraries from auto-loading and parsing .env (which causes noisy parse warnings)
    try:
        os.environ.setdefault("DOTENV_DISABLE", "1")
    except Exception:
        pass
    # Load .env safely (and only if not explicitly disabled) to avoid parse spam
    try:
        if os.getenv("DOTENV_DISABLE", "0").lower() not in ("1", "true", "yes"):
            env_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
            if env_path:
                values = dotenv_values(env_path) or {}
                for k, v in values.items():
                    if v is None:
                        continue
                    # Respect process env precedence by default
                    precedence = os.getenv("RAG_ENV_PRECEDENCE", "process").lower()
                    if precedence in ("process", "runtime"):
                        os.environ.setdefault(k, v)
                    else:
                        os.environ[k] = v
    except Exception:
        pass
    # Fast path: UI-only mode to reuse existing artifacts (skip ingestion/indexing)
    try:
        ui_only = os.getenv("RAG_UI_ONLY", "0").lower() in ("1", "true", "yes")
    except Exception:
        ui_only = False
    if ui_only:
        # Don't clean artifacts; assume vector DB and snapshots are already present
        try:
            os.environ["RAG_CLEAN_RUN"] = "0"
        except Exception:
            pass
        try:
            # Load docs from full snapshot for UI context/debug (if present)
            docs: List[Document] = []
            snap_full = Path("logs") / "db_snapshot_full.jsonl"
            if snap_full.exists():
                with open(snap_full, "r", encoding="utf-8") as f:
                    for ln in f:
                        try:
                            rec = json.loads(ln)
                            txt = rec.get("text") or ""
                            md = rec.get("metadata") or {}
                            # Fallback to minimal fields if metadata missing
                            if not md:
                                md = {
                                    "file_name": rec.get("file"),
                                    "page": rec.get("page"),
                                    "section": rec.get("section"),
                                    "anchor": rec.get("anchor"),
                                }
                            docs.append(Document(page_content=txt, metadata=md))
                        except Exception:
                            continue
            else:
                print("[UI-ONLY] Missing logs/db_snapshot_full.jsonl; cannot load docs for UI.")
                docs = []
            # Reuse persisted Chroma if configured, else build a minimal in-memory hybrid from docs
            emb = _get_embeddings()
            dense_store = None
            try:
                from langchain_chroma import Chroma  # type: ignore
                persist_dir = os.getenv("RAG_CHROMA_DIR")
                collection = os.getenv("RAG_CHROMA_COLLECTION")
                if persist_dir:
                    Path(persist_dir).mkdir(parents=True, exist_ok=True)
                    dense_store = Chroma(embedding_function=emb, persist_directory=persist_dir, collection_name=collection) if collection else Chroma(embedding_function=emb, persist_directory=persist_dir)
                else:
                    print("[UI-ONLY] RAG_CHROMA_DIR not set; dense retrieval will be in-memory from docs.")
            except Exception as e:
                print(f"[UI-ONLY] Failed to open persisted Chroma: {e}")
                dense_store = None
            # Sparse retriever from current docs (used both for hybrid and UI previews)
            sparse = build_sparse_retriever(docs, k=settings.SPARSE_K)
            # If no persisted dense store, fall back to creating an in-memory vectorstore for docs
            if dense_store is None:
                dense_store = build_dense_index(docs, emb)
            hybrid = build_hybrid_retriever(dense_store, sparse, dense_k=settings.DENSE_K)
            # Basic graph render (best-effort; optional)
            try:
                G = build_graph(docs)
                Path("logs").mkdir(exist_ok=True)
                render_graph_html(G, str(Path("logs") / "graph.html"))
            except Exception:
                pass
            llm = _LLM()

            # Optional: evaluation mode even in UI-only reuse flow (skip ingestion/indexing)
            try:
                if os.getenv("RAG_EVAL", "").lower() in ("1", "true", "yes"):
                    run_evaluation(docs, hybrid, llm)
                    # In headless mode, don't launch UI afterward
                    if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
                        return
            except Exception as e:
                print(f"[UI-ONLY] Evaluation failed: {e}")

            # Launch UI directly (with resilient port fallback)
            try:
                ui = build_ui(docs, hybrid, llm, debug=None)
                share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
                base_server = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
                base_port = int(os.getenv("GRADIO_PORT", "7860"))
                launched = False
                last_err = None
                for i in range(0, 5):  # try up to 5 consecutive ports
                    try:
                        port = base_port + i
                        print(f"[UI-ONLY] Launching UI on http://{base_server}:{port} (try {i+1})")
                        ui.launch(share=share, server_name=base_server, server_port=port, show_error=True)
                        launched = True
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if not launched:
                    print(f"UI failed to launch after retries (UI-ONLY): {last_err}")
            except Exception as e:
                print(f"UI failed to build/launch (UI-ONLY): {e}")
            return
        except Exception as e:
            print(f"[UI-ONLY] Fallback to full pipeline due to error: {e}")
    _clean_run_outputs()
    # Default-enable DeepEval when API key is present unless explicitly disabled
    try:
        if (os.getenv("CONFIDENT_API_KEY") or os.getenv("DEEPEVAL_API_KEY") or os.getenv("OPENAI_API_KEY")) and os.getenv("RAG_DEEPEVAL") is None:
            os.environ.setdefault("RAG_DEEPEVAL", "1")
        # If DeepEval is enabled and not explicitly retaining RAGAS, skip RAGAS to avoid async/provider conflicts
        if os.getenv("RAG_DEEPEVAL", "0").lower() in ("1","true","yes") and os.getenv("RAG_KEEP_RAGAS") not in ("1","true","yes"):
            os.environ.setdefault("RAG_SKIP_RAGAS", "1")
    except Exception:
        pass
    paths = _discover_input_paths()
    if not paths:
        # Fallback: reuse existing artifacts and still launch the UI (do not bail out)
        print("No new input files found. Falling back to UI with existing artifacts (if any).")
        try:
            os.environ.setdefault("RAG_CLEAN_RUN", "0")
        except Exception:
            pass
        try:
            # Load docs from full snapshot for UI context/debug (if present)
            docs: List[Document] = []
            snap_full = Path("logs") / "db_snapshot_full.jsonl"
            if snap_full.exists():
                with open(snap_full, "r", encoding="utf-8") as f:
                    for ln in f:
                        try:
                            rec = json.loads(ln)
                            txt = rec.get("text") or ""
                            md = rec.get("metadata") or {}
                            if not md:
                                md = {
                                    "file_name": rec.get("file"),
                                    "page": rec.get("page"),
                                    "section": rec.get("section"),
                                    "anchor": rec.get("anchor"),
                                }
                            docs.append(Document(page_content=txt, metadata=md))
                        except Exception:
                            continue
            else:
                print("[FALLBACK] Missing logs/db_snapshot_full.jsonl; proceeding with empty docs list.")

            emb = _get_embeddings()
            dense_store = None
            try:
                from langchain_chroma import Chroma  # type: ignore
                persist_dir = os.getenv("RAG_CHROMA_DIR")
                collection = os.getenv("RAG_CHROMA_COLLECTION")
                if persist_dir:
                    Path(persist_dir).mkdir(parents=True, exist_ok=True)
                    dense_store = Chroma(embedding_function=emb, persist_directory=persist_dir, collection_name=collection) if collection else Chroma(embedding_function=emb, persist_directory=persist_dir)
                else:
                    print("[FALLBACK] RAG_CHROMA_DIR not set; dense retrieval will be in-memory from docs.")
            except Exception as e:
                print(f"[FALLBACK] Failed to open persisted Chroma: {e}")
                dense_store = None

            sparse = build_sparse_retriever(docs, k=settings.SPARSE_K)
            if dense_store is None:
                dense_store = build_dense_index(docs, emb)
            hybrid = build_hybrid_retriever(dense_store, sparse, dense_k=settings.DENSE_K)

            # Optional quick graph render
            try:
                G = build_graph(docs)
                Path("logs").mkdir(exist_ok=True)
                render_graph_html(G, str(Path("logs") / "graph.html"))
            except Exception:
                pass
            llm = _LLM()

            # Optional evaluation in fallback mode
            try:
                if os.getenv("RAG_EVAL", "").lower() in ("1", "true", "yes"):
                    run_evaluation(docs, hybrid, llm)
                    if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
                        return
            except Exception as e:
                print(f"[FALLBACK] Evaluation failed: {e}")

            # Launch UI unless headless (with resilient port fallback)
            if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
                print("[HEADLESS] Fallback complete. Skipping UI launch.")
                return
            try:
                ui = build_ui(docs, hybrid, llm, debug=None)
                share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
                server = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
                base_port = int(os.getenv("GRADIO_PORT", "7860"))
                launched = False
                last_err = None
                for i in range(0, 5):
                    try:
                        port = base_port + i
                        get_logger().info("Launching UI on http://%s:%s (share=%s) [try=%d]", server, port, share, i+1)
                        ui.launch(share=share, server_name=server, server_port=port, show_error=True)
                        launched = True
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if not launched:
                    print(f"UI failed to launch (fallback) after retries: {last_err}")
            except Exception as e:
                print(f"UI failed to launch (fallback): {e}")
            return
        except Exception as e:
            print(f"[FALLBACK] Error preparing UI: {e}")
            return
    # Log core toggles once
    try:
        log = get_logger()
        log.info("FLAGS: HEADLESS=%s EVAL=%s USE_NORMALIZED=%s VEC_BACKEND=%s LLM_INDEX=%s LLAMAPARSE=%s", os.getenv("RAG_HEADLESS"), os.getenv("RAG_EVAL"), os.getenv("RAG_USE_NORMALIZED"), os.getenv("RAG_VECTOR_BACKEND", "chroma"), os.getenv("RAG_ENABLE_LLAMAINDEX"), os.getenv("RAG_USE_LLAMAPARSE"))
    except Exception:
        pass
    docs, hybrid, debug = build_pipeline(paths)
    # Log discovered image assets (help explain figure counts)
    try:
        log = get_logger()
        figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure"]
        if figs:
            for d in figs:
                md = d.metadata or {}
                if md.get("image_path"):
                    asset_tag = " (asset)" if md.get("is_asset") else ""
                    log.info(
                        "Image%s: %s p%s fig#%s -> %s",
                        asset_tag,
                        md.get("file_name"), md.get("page"), md.get("figure_number"), md.get("image_path"),
                    )
    except Exception:
        pass
    # Optional: export LlamaIndex artifacts and mirror pipeline tables/images
    try:
        enable_llx = os.getenv("RAG_ENABLE_LLAMAINDEX", "0").lower() in ("1", "true", "yes")
        if enable_llx:
            n = export_llamaindex_for(paths)
            if n:
                print(f"[LlamaIndex] Exported artifacts for {n} document(s) under data/elements/llamaindex")
            else:
                print("[LlamaIndex] No export performed (missing dependency or no PDFs)")
    except Exception:
        pass
    # Build a lightweight graph and render it for UI
    try:
        G = build_graph(docs)
        Path("logs").mkdir(exist_ok=True)
        graph_html = str(Path("logs") / "graph.html")
        render_graph_html(G, graph_html)
    except Exception:
        graph_html = None
    llm = _LLM()
    # Optional: evaluation mode
    if os.getenv("RAG_EVAL", "").lower() in ("1", "true", "yes"):
        run_evaluation(docs, hybrid, llm)
        if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
            return
    # Launch Gradio UI (skip in headless mode). Try a few alternative ports if busy.
    if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
        print("[HEADLESS] Ingestion complete. Skipping UI launch.")
        return
    try:
        ui = build_ui(docs, hybrid, llm, debug)
        share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
        server = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
        base_port = int(os.getenv("GRADIO_PORT", "7860"))
        launched = False
        last_err = None
        for i in range(0, 5):
            try:
                port = base_port + i
                get_logger().info("Launching UI on http://%s:%s (share=%s) [try=%d]", server, port, share, i+1)
                ui.launch(share=share, server_name=server, server_port=port, show_error=True)
                launched = True
                break
            except Exception as e:
                last_err = e
                continue
        if not launched:
            print(f"UI failed to launch after retries: {last_err}")
            # Last resort: write a minimal static HTML with a message
            try:
                Path("logs").mkdir(exist_ok=True)
                (Path("logs")/"ui_failed.html").write_text("<h1>UI failed to launch</h1><p>See console for details.</p>", encoding="utf-8")
                print("Wrote logs/ui_failed.html as a fallback notice.")
            except Exception:
                pass
    except Exception as e:
        print(f"UI failed to launch: {e}")
        print(ask(docs, hybrid, llm, "Summarize the failure modes described."))
