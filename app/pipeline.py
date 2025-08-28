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
from typing import List, Tuple
from app.logger import trace_func

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
    # Optional: clean Chroma persist dir
    try:
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
                self._backend = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                self._which = "google"
            except Exception:
                self._backend = None
        # Fallback to OpenAI (or forced)
        if (self._backend is None or force_openai) and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-nano")
                try:
                    self._backend = ChatOpenAI(model=model, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
                except Exception:
                    try:
                        self._backend = ChatOpenAI(model_name=model, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
                    except Exception:
                        self._backend = ChatOpenAI(model=model, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
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
    records = []
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
        chunks = structure_chunks(elements, str(path))
        for ch in chunks:
            records.append(attach_metadata(ch, client_id=os.getenv("CLIENT_ID"), case_id=path.stem))
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
        with open(snap_path, "w", encoding="utf-8") as f, open(full_snap_path, "w", encoding="utf-8") as f_full:
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
                    "page": md.get("page"),
                    "section": md.get("section"),
                    "anchor": md.get("anchor"),
                    # Deterministic IDs for traceability
                    "doc_id": md.get("doc_id"),
                    "chunk_id": md.get("chunk_id"),
                    "content_hash": md.get("content_hash"),
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
                        "page": md.get("page"),
                        "section": md.get("section") or md.get("section_type"),
                        "anchor": md.get("anchor"),
                        "doc_id": md.get("doc_id"),
                        "chunk_id": md.get("chunk_id"),
                        "content_hash": md.get("content_hash"),
                        "metadata": md,  # entire metadata blob for offline inspection
                        "text": txt,     # full chunk text (untruncated)
                        "words": len((txt or "").split()),
                    }
                    f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")
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
    # Optional: populate graph database from current docs (entity co-mention graph)
    try:
        if os.getenv("RAG_GRAPH_DB", "").lower() in ("1", "true", "yes"):
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
    qa = query_analyzer(question)
    q_exec = qa.get("canonical") or question
    try:
        candidates = hybrid.invoke(q_exec)
    except Exception:
        # Fallback to invoke for older retriever interfaces
        candidates = hybrid.invoke(q_exec)
    candidates = candidates[: settings.K_TOP_K]  # rerank TOP K
    filtered = apply_filters(candidates, qa["filters"])  # metadata filters
    try:
        sec = qa["filters"].get("section")
    except Exception:
        sec = None
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
    top_docs = rerank_candidates(q_exec, filtered, top_n=settings.CONTEXT_TOP_N)
    route, rtrace = route_question_ex(question)

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
    if route == "summary":
        ans = answer_summary(_LLM(), top_docs, question)
    elif route == "table":
        ans = answer_table(_LLM(), top_docs, question)
    else:
        ans = answer_needle(_LLM(), top_docs, question)
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "question": question,
            "route": route,
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
            "answer_preview": (ans or "")[:400],
        }
        with open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return ans


@trace_func
def answer_with_contexts(docs, hybrid, llm: _LLM, question: str):
    """Answer a question and also return the contexts used (top_docs)."""
    log = get_logger()
    qa = query_analyzer(question)
    q_exec = qa.get("canonical") or question
    try:
        candidates = hybrid.invoke(q_exec)
    except Exception:
        candidates = hybrid.invoke(q_exec)
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
    route = route_question(question)
    if route == "summary":
        ans = answer_summary(llm, top_docs, question)
    elif route == "table":
        ans = answer_table(llm, top_docs, question)
    else:
        ans = answer_needle(llm, top_docs, question)
    return ans, top_docs


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
    try:
        if os.getenv("OPENAI_API_KEY"):
            os.environ.setdefault("RAGAS_LLM_PROVIDER", "openai")
    except Exception:
        pass
    qa_rows = _load_json_or_jsonl(qa_path)
    try:
        log.info("QA auto-load: Loaded %d QA items from %s", len(qa_rows or []), str(qa_path))
    except Exception:
        pass
    gt_rows = _load_json_or_jsonl(gt_path) if gt_path else []
    gt_by_id, gt_by_q = _index_ground_truth(gt_rows)
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
            ans, ctx_docs = answer_with_contexts(docs, hybrid, llm, q)
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
        }
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
    ds = {"question": [], "answer": [], "contexts": [], "reference": [], "ground_truths": []}
    for r in rows_out:
        ds["question"].append(r["question"])
        ds["answer"].append(r["answer"])
        ds["contexts"].append(r["contexts"])
        ds["reference"].append(r.get("reference", ""))
        ds["ground_truths"].append(r.get("ground_truths", []))  # type: ignore[index]
    try:
        summary, per_q = run_eval_detailed(ds)
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return
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
    with open(out_dir / "eval_ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(summary), f, ensure_ascii=False, indent=2)
    per_q_path = out_dir / "eval_ragas_per_question.jsonl"
    with open(per_q_path, "w", encoding="utf-8") as f:
        for rec in per_q:
            f.write(json.dumps(_nan_to_none(rec), ensure_ascii=False) + "\n")
        # Footer: averaged metrics across all questions for quick inspection
        try:
            s = _nan_to_none(summary)
            footer = {
                "__summary__": True,
                "faithfulness": (s.get("faithfulness") if isinstance(s, dict) else None),
                "answer_relevancy": (s.get("answer_relevancy") if isinstance(s, dict) else None),
                "context_precision": (s.get("context_precision") if isinstance(s, dict) else None),
                "context_recall": (s.get("context_recall") if isinstance(s, dict) else None),
            }
            f.write(json.dumps(footer, ensure_ascii=False) + "\n")
        except Exception:
            pass
    log = get_logger()
    summ_str = pretty_metrics(summary)
    print("RAGAS summary:\n" + summ_str)
    try:
        log.info("Evaluation summary (averaged over %d items):\n%s", len(rows_out), summ_str)
        log.info("Saved summary to %s and per-question to %s", str(out_dir / "eval_ragas_summary.json"), str(per_q_path))
    except Exception:
        pass
    print("\nPer-question results:")
    try:
        for rec in per_q:
            q = rec.get("question", "")
            ans = rec.get("answer", "")
            mets = {k: v for k, v in rec.items() if k not in ("question", "answer", "contexts", "ground_truths")}
            print("- Q:", q)
            print("  A:", (ans or "")[:400])
            print("  metrics:", json.dumps(mets, ensure_ascii=False))
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
    _clean_run_outputs()
    paths = _discover_input_paths()
    if not paths:
        print("No input files found. Place PDFs/DOCs under data/ or the root PDF.")
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
    # Launch Gradio UI (skip in headless mode)
    if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
        print("[HEADLESS] Ingestion complete. Skipping UI launch.")
        return
    try:
        ui = build_ui(docs, hybrid, llm, debug)
        share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
        port = int(os.getenv("GRADIO_PORT", "7860"))
        ui.launch(share=share, server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=port, show_error=True)
    except Exception as e:
        print(f"UI failed to launch: {e}")
        print(ask(docs, hybrid, llm, "Summarize the failure modes described."))
