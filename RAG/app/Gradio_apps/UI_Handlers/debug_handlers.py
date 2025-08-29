#!/usr/bin/env python3
"""
Debug Handlers
=============

Debug panel and logging logic functions.
"""

import json
from typing import List, Dict, Any, Optional
import gradio as gr
from RAG.app.logger import get_logger
from RAG.app.Gradio_apps.ui_components import _rows_to_df, _rows_from_docs, _fmt_docs, _render_router_info


def log_query_and_answer(q, out, metrics_txt):
    """Log query, answer, and metrics."""
    log = get_logger()
    log.info("Q: %s", q)
    log.info("Answer: %s", out)
    if metrics_txt:
        log.info("Metrics:\n%s", metrics_txt)


def audit_query_to_file(q, r, rtrace, out, metrics_txt, top_docs):
    """Audit query to JSONL file."""
    try:
        from RAG.app.config import settings
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "question": q,
            "route": r,
            "router_trace": rtrace,
            "answer": out,
            "metrics": metrics_txt,
            "contexts": [
                {
                    "file": d.metadata.get("file_name"),
                    "page": d.metadata.get("page"),
                    "section": d.metadata.get("section"),
                }
                for d in top_docs
            ],
        }
        with open(settings.LOGS_DIR/"queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def build_debug_outputs(qa, r, rtrace, dense_docs, sparse_docs, cands, top_docs, dbg):
    """Build structured debug outputs."""
    _dbg_visible = bool(dbg)
    _dbg_router_md = f"**Route:** {r}  \n**Rules:** {', '.join(rtrace.get('matched', []))}  \n**Canonical:** {qa.get('canonical', '')}"
    _dbg_filters_json = {"filters": qa.get("filters", {}), "keywords": qa.get("keywords", []), "canonical": qa.get("canonical", "")}
    _dbg_dense_md = "Dense (top10):\n\n" + _fmt_docs(dense_docs)
    _dbg_sparse_md = "Sparse (top10):\n\n" + _fmt_docs(sparse_docs)
    _dbg_hybrid_md = "Hybrid candidates (pre-filter):\n\n" + _fmt_docs(cands)
    _dbg_top_df = _rows_to_df(_rows_from_docs(top_docs))
    
    return _dbg_visible, _dbg_router_md, _dbg_filters_json, _dbg_dense_md, _dbg_sparse_md, _dbg_hybrid_md, _dbg_top_df


def create_debug_updates(_dbg_visible, _dbg_router_md, _dbg_filters_json, _dbg_dense_md, _dbg_sparse_md, _dbg_hybrid_md, _dbg_top_df, compare_dict, fig_path):
    """Create Gradio update objects for debug panels."""
    _acc_upd = gr.update(visible=_dbg_visible, open=False)
    _router_upd = gr.update(value=_dbg_router_md, visible=_dbg_visible)
    _filters_upd = gr.update(value=_dbg_filters_json, visible=_dbg_visible)
    _dense_upd = gr.update(value=_dbg_dense_md, visible=_dbg_visible)
    _sparse_upd = gr.update(value=_dbg_sparse_md, visible=_dbg_visible)
    _hybrid_upd = gr.update(value=_dbg_hybrid_md, visible=_dbg_visible)
    _topdocs_upd = gr.update(value=_dbg_top_df, visible=_dbg_visible)
    _compare_upd = gr.update(value=compare_dict, visible=_dbg_visible)
    # Update figure preview slot when available; leave None to avoid clearing external viewers
    fig_update = gr.update(value=fig_path) if 'fig_path' in locals() and fig_path else None
    
    return _acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, fig_update
