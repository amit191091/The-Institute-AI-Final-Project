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

from dotenv import load_dotenv

from app.config import settings
from app.loaders import load_many
from app.chunking import structure_chunks
from app.metadata import attach_metadata
from app.indexing import (
    build_dense_index,
    build_sparse_retriever,
    to_documents,
    dump_chroma_snapshot,
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
from app.logger import get_logger
from app.eval_ragas import run_eval_detailed, pretty_metrics


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


def _get_embeddings():
    """Prefer Google embeddings, fallback to OpenAI, then FakeEmbeddings for local smoke tests."""
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
                    self._backend = ChatOpenAI(model=model, temperature=0.1)  # type: ignore[call-arg]
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
                return f"[LLM error] {e}\n\n{prompt[-400:]}"
        return "[LLM not configured] " + prompt[-400:]


def build_pipeline(paths: List[Path]):
    """Ingest documents, build chunks+metadata, and initialize hybrid retriever."""
    log = get_logger()
    records = []
    # ingest
    for path, elements in load_many(paths):
        # basic ingestion validation: min pages
        try:
            pages_list = [
                int(getattr(getattr(e, "metadata", None), "page_number", 0))
                for e in elements
                if getattr(getattr(e, "metadata", None), "page_number", None) is not None
            ]
            pages = sorted(set(pages_list))
            ok, msg = validate_min_pages(len(set(pages)), settings.MIN_PAGES)
            if not ok:
                print(f"[WARN] {path.name}: {msg}")
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
    # Write a quick DB snapshot for debugging
    try:
        Path("logs").mkdir(exist_ok=True)
        snap_path = Path("logs") / "db_snapshot.jsonl"
        with open(snap_path, "w", encoding="utf-8") as f:
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
    return docs, hybrid, debug


def ask(docs, hybrid, llm: _LLM, question: str, ground_truth: str | None = None) -> str:
    """Answer a user question using the hybrid retriever and route to sub-agents."""
    log = get_logger()
    qa = query_analyzer(question)
    q_exec = qa.get("canonical") or question
    try:
        candidates = hybrid.invoke(q_exec)
    except Exception:
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
        "Q: %s | route=%s | keywords=%s | filters=%s | pool=%d | filtered=%d | using=%d",
        q_exec,
        route,
        qa["keywords"],
        qa["filters"],
        len(candidates),
        len(filtered),
        len(top_docs),
    )
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


def _discover_eval_files():
    qa_candidates = [
        Path(os.getenv("RAG_QA_FILE", "")),
        Path("gear_wear_qa.json"),
        Path("gear_wear_qa.jsonl"),
        Path("data") / "gear_wear_qa.json",
        Path("data") / "gear_wear_qa.jsonl",
    ]
    gt_candidates = [
        Path(os.getenv("RAG_GT_FILE", "")),
        Path("gear_wear_ground_truth.json"),
        Path("gear_wear_ground_truth.json"),
        Path("data") / "gear_wear_ground_truth.json",
    ]
    qa = next((p for p in qa_candidates if str(p) and p.exists()), None)
    gt = next((p for p in gt_candidates if str(p) and p.exists()), None)
    return qa, gt


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


def run_evaluation(docs, hybrid, llm: _LLM):
    log = get_logger()
    qa_path, gt_path = _discover_eval_files()
    if not qa_path:
        log.warning("Evaluation requested but QA file not found.")
        return
    try:
        if os.getenv("OPENAI_API_KEY"):
            os.environ.setdefault("RAGAS_LLM_PROVIDER", "openai")
    except Exception:
        pass
    qa_rows = _load_json_or_jsonl(qa_path)
    gt_rows = _load_json_or_jsonl(gt_path) if gt_path else []
    gt_map = _normalize_ground_truth(gt_rows)
    rows_out = []
    any_gt = False
    for i, row in enumerate(qa_rows, start=1):
        if not isinstance(row, dict):
            continue
        q = row.get("question") or row.get("q") or row.get("prompt") or row.get("text")
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
        gts = gt_map.get(norm_q, [])
        if not gts and gt_map:
            keys = list(gt_map.keys())
            best = None
            best_score = 0.0
            for k in keys:
                s = difflib.SequenceMatcher(None, norm_q, k).ratio()
                if s > best_score:
                    best_score = s
                    best = k
            if best is not None and best_score >= 0.75:
                gts = gt_map.get(best, [])
        if (not gts) and isinstance(row.get("answer"), (str, int, float)):
            ans_txt = str(row["answer"]).strip()
            if ans_txt:
                gts = [ans_txt]
        if gts:
            any_gt = True
        ref = gts[0] if isinstance(gts, list) and gts else ""
        rows_out.append({
            "question": q,
            "answer": ans or "",
            "contexts": ctxs,
            "ground_truths": gts,
            "reference": ref,
        })
        try:
            log.info("EVAL Q[%d]: %s", i, q)
        except Exception:
            pass
    if not rows_out:
        print("No evaluation rows to process.")
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
    with open(out_dir / "eval_ragas_per_question.jsonl", "w", encoding="utf-8") as f:
        for rec in per_q:
            f.write(json.dumps(_nan_to_none(rec), ensure_ascii=False) + "\n")
    print("RAGAS summary:\n" + pretty_metrics(summary))
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


def run() -> None:
    """Entry point that mirrors the prior Main.main() behavior."""
    print("stating program")
    # Load .env and override any pre-set env vars to ensure the correct API keys are used
    load_dotenv(override=True)
    _clean_run_outputs()
    paths = _discover_input_paths()
    if not paths:
        print("No input files found. Place PDFs/DOCs under data/ or the root PDF.")
        return
    docs, hybrid, debug = build_pipeline(paths)
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
