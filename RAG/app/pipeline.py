from __future__ import annotations

"""Refactored pipeline module using modular components.

This module now orchestrates the pipeline using separate modules for:
- pipeline_ingestion: Document loading and processing
- pipeline_query: Query processing and answering  
- pipeline_utils: Utility functions and helpers
- pipeline_core: Main orchestration

The original functionality is preserved while improving modularity and maintainability.
"""

import os
import json
import math
import difflib
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from RAG.app.config import settings
from RAG.app.ui_gradio import build_ui
from RAG.app.logger import get_logger

# Import our modular components directly
from RAG.app.pipeline_modules.pipeline_core import build_pipeline, get_pipeline_components
from RAG.app.pipeline_modules.pipeline_query import answer_question, answer_with_contexts
from RAG.app.pipeline_modules.pipeline_utils import (
    LLM, 
    load_json_or_jsonl, 
    discover_eval_files, 
    normalize_ground_truth, 
    nan_to_none,
    get_embeddings
)
from RAG.app.pipeline_modules.pipeline_ingestion import clean_run_outputs, discover_input_paths

# Optional graph visualization/database modules. Provide no-op fallbacks if missing.
try:
    from RAG.app.pipeline_modules.graph import build_graph, render_graph_html  # type: ignore
    from RAG.app.pipeline_modules.graphdb import build_graph_db, run_cypher  # type: ignore
    from RAG.app.pipeline_modules.graphdb_import_normalized import import_normalized_graph  # type: ignore
except Exception as e:  # pragma: no cover
    def build_graph(docs):
        return None
    def render_graph_html(G, out_path: str):
        return None
    def build_graph_db(docs):
        return 0
    def run_cypher(query, parameters=None):
        return []
    def import_normalized_graph(graph_path, chunks_path):
        return 0

# Optional advanced modules. Provide no-op fallbacks if missing.
try:
    from RAG.app.pipeline_modules.clean_table_extract import extract_tables_clean  # type: ignore
    from RAG.app.pipeline_modules.llamaindex_export import export_llamaindex_for  # type: ignore
    from RAG.app.pipeline_modules.llamaindex_compare import build_alt_indexes  # type: ignore
    from RAG.app.Evaluation_Analysis.deepeval_integration import run_eval as run_eval_deepeval  # type: ignore
except Exception as e:  # pragma: no cover
    def extract_tables_clean(pdf_path):
        return []
    def export_llamaindex_for(paths, out_root=None):
        return 0
    def build_alt_indexes(paths, embeddings):
        return {}
    def run_eval_deepeval(dataset):
        return None, []

from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval_detailed, pretty_metrics


# Legacy function aliases removed - use the actual functions directly


def run_evaluation(docs, hybrid, llm: LLM):
    log = get_logger()
    qa_path, gt_path = discover_eval_files()
    if not qa_path:
        log.warning("Evaluation requested but QA file not found.")
        return
    # RAGAS now uses config-based provider selection
    qa_rows = load_json_or_jsonl(qa_path)
    gt_rows = load_json_or_jsonl(gt_path) if gt_path else []
    gt_map = normalize_ground_truth(gt_rows)
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
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue
        ctxs = [getattr(d, "page_content", "") for d in (ctx_docs or []) if getattr(d, "page_content") or ""]
        if not ctxs:
            ctxs = [getattr(docs[0], "page_content", "")] if docs else [""]
        norm_q = str(q).lower().strip()
        norm_q = " ".join(norm_q.split())
        norm_q = norm_q.strip(".,:;!?-Γ\u2013\u2014\"'()[]{}")
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
        from contextlib import suppress
        with suppress(Exception):
            log.info("EVAL Q[%d]: %s", i, q)
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
    
    # Set up output directory
    out_dir = settings.paths.LOGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # DeepEval integration
    try:
        de_sum, de_rows = run_eval_deepeval(ds)
        if de_sum:
            print("\nDeepEval summary:")
            print(json.dumps(de_sum, indent=2))
            
            # Save DeepEval results
            with open(out_dir / "eval_deepeval_summary.json", "w", encoding="utf-8") as f:
                json.dump(_nan_to_none(de_sum), f, ensure_ascii=False, indent=2)
            with open(out_dir / "eval_deepeval_per_question.jsonl", "w", encoding="utf-8") as f:
                for rec in de_rows:
                    f.write(json.dumps(_nan_to_none(rec), ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"DeepEval run failed: {e}")
    
    def _nan_to_none(x):
        return None if isinstance(x, float) and math.isnan(x) else (
            [_nan_to_none(v) for v in x] if isinstance(x, list) else (
                {k: _nan_to_none(v) for k, v in x.items()} if isinstance(x, dict) else x
            )
        )
    with open(out_dir / "eval_ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(summary), f, ensure_ascii=False, indent=2)
    with open(out_dir / "eval_ragas_per_question.jsonl", "w", encoding="utf-8") as f:
        for rec in per_q:
            f.write(json.dumps(_nan_to_none(rec), ensure_ascii=False) + "\n")
    print("RAGAS summary:\n" + pretty_metrics(summary))
    print("\nPer-question results:")
    from contextlib import suppress
    with suppress(Exception):
        for rec in per_q:
            q = rec.get("question", "")
            ans = rec.get("answer", "")
            mets = {k: v for k, v in rec.items() if k not in ("question", "answer", "contexts", "ground_truths")}
            print("- Q:", q)
            print("  A:", (ans or "")[:400])
            print("  metrics:", json.dumps(mets, ensure_ascii=False))


def run() -> None:
    """Entry point that mirrors the prior Main.main() behavior."""
    # Load environment variables
    load_dotenv(override=True)
    
    # Set logging level to reduce verbose output
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    # Get logger for this module
    log = get_logger()
    
    # Advanced environment configuration
    # Default-enable DeepEval when API key is present
    if (os.getenv("CONFIDENT_API_KEY") or os.getenv("DEEPEVAL_API_KEY")) and os.getenv("RAG_DEEPEVAL") is None:
        os.environ.setdefault("RAG_DEEPEVAL", "1")
    
    # Enhanced logging of core toggles
    log.info("FLAGS: HEADLESS=%s EVAL=%s USE_NORMALIZED=%s VEC_BACKEND=%s LLM_INDEX=%s LLAMAPARSE=%s", 
             os.getenv("RAG_HEADLESS"), os.getenv("RAG_EVAL"), os.getenv("RAG_USE_NORMALIZED"), 
             os.getenv("RAG_VECTOR_BACKEND", "chroma"), os.getenv("RAG_ENABLE_LLAMAINDEX"), os.getenv("RAG_USE_LLAMAPARSE"))
    
    log.info("Starting RAG pipeline...")
    
    # Use RAGService for business logic
    from RAG.app.rag_service import RAGService
    service = RAGService()
    
    try:
        # Run the pipeline using service
        result = service.run_pipeline(use_normalized=False)
        docs = result["docs"]
        hybrid = result["hybrid_retriever"]
        debug = None  # Debug info not needed for UI
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return
    # Build a lightweight graph and render it for UI
    try:
        G = build_graph(docs)
        settings.paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        graph_html = str(settings.paths.LOGS_DIR / "graph.html")
        render_graph_html(G, graph_html)
    except Exception as e:
        graph_html = None
    
    # Graph database integration
    try:
        if os.getenv("RAG_GRAPH_DB", "1").lower() not in ("0", "false", "no"):
            n = build_graph_db(docs)
            log.info(f"[GraphDB] Upserted ~{n} nodes/edges to Neo4j")
    except Exception as e:
        log.warning(f"Graph database integration failed: {e}")
    
    # Normalized graph processing
    try:
        if os.getenv("RAG_USE_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            gpath = settings.paths.LOGS_DIR / "normalized" / "graph.json"
            if gpath.exists():
                data = json.loads(gpath.read_text(encoding="utf-8"))
                log.info(f"[NormalizedGraph] nodes={len(data.get('nodes', []))}, edges={len(data.get('edges', []))}")
        
        if os.getenv("RAG_IMPORT_NORMALIZED_GRAPH", "0").lower() in ("1", "true", "yes"):
            gpath = settings.paths.LOGS_DIR / "normalized" / "graph.json"
            cpath = settings.paths.LOGS_DIR / "normalized" / "chunks.jsonl"
            if gpath.exists() and cpath.exists():
                n2 = import_normalized_graph(gpath, cpath)
                log.info(f"[GraphDB] normalized import result={n2}")
    except Exception as e:
        log.warning(f"Normalized graph processing failed: {e}")
    
    # LlamaIndex export integration
    try:
        enable_llx = os.getenv("RAG_ENABLE_LLAMAINDEX", "0").lower() in ("1", "true", "yes")
        if enable_llx:
            # Get document paths from service
            paths = getattr(service, 'input_paths', []) or discover_input_paths()
            if paths:
                n = export_llamaindex_for(paths)
                if n:
                    log.info(f"[LlamaIndex] Exported artifacts for {n} document(s) under data/elements/llamaindex")
    except Exception as e:
        log.warning(f"LlamaIndex export failed: {e}")
    
    # Alternative indexes building
    try:
        if os.getenv("RAG_BUILD_ALT_INDEXES", "0").lower() in ("1", "true", "yes"):
            # Get document paths and embeddings
            paths = getattr(service, 'input_paths', []) or discover_input_paths()
            if paths:
                embeddings = get_embeddings()
                alt = build_alt_indexes(paths, embeddings)
                for key, obj in (alt or {}).items():
                    log.info(f"ALT[{key}]: docs={len(obj.get('docs', []))} | dense={type(obj.get('dense')).__name__} | sparse={type(obj.get('sparse')).__name__}")
    except Exception as e:
        log.warning(f"Alternative indexes building failed: {e}")
    
    llm = LLM()
    # Optional: evaluation mode
    rag_eval_value = os.getenv("RAG_EVAL", "")
    if rag_eval_value.lower() in ("1", "true", "yes"):
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
        server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
        
        try:
            ui.launch(
                share=share, 
                server_name=server_name, 
                server_port=port, 
                show_error=True, 
                inbrowser=True
            )
        except KeyboardInterrupt:
            print("\n🔄 Keyboard interruption in main thread... closing server.")
            from contextlib import suppress
            with suppress(Exception):
                ui.close()
            print("✅ Server closed gracefully.")
        except SystemExit:
            print("\n🔄 System exit requested... closing server.")
            from contextlib import suppress
            with suppress(Exception):
                ui.close()
            print("✅ Server closed gracefully.")
        except Exception as e:
            print(f"Server error: {e}")
            from contextlib import suppress
            with suppress(Exception):
                ui.close()
    except Exception as e:
        print(f"UI failed to launch: {e}")
        print("Please check the error above and try again.")
