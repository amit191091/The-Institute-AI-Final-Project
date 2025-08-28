#!/usr/bin/env python3
"""
Query Processors
===============

Query analysis and routing logic functions.
"""

import re
from typing import List, Dict, Any, Optional
from RAG.app.retrieve import apply_filters, query_analyzer, rerank_candidates
from RAG.app.Gradio_apps.ui_components import _fig_sort_key


def _fallback_ok(d, qa):
    """Fallback filtering logic when no documents match initial filters."""
    md = d.metadata or {}
    if (md.get("section") or md.get("section_type")) != qa.get("filters", {}).get("section"):
        return False
    # Keep number filters when present
    fv = qa.get("filters", {}).get("figure_number") if qa else None
    tv = qa.get("filters", {}).get("table_number") if qa else None
    import re as _re
    if fv is not None:
        fn = md.get("figure_number")
        if str(fn) == str(fv):
            return True
        lab = str(md.get("figure_label") or md.get("caption") or "")
        return bool(_re.match(rf"^\s*figure\s*{int(str(fv))}\b", lab, _re.I))
    if tv is not None:
        tn = md.get("table_number")
        if str(tn) == str(tv):
            return True
        lab = str(md.get("table_label") or "")
        return bool(_re.match(rf"^\s*table\s*{int(str(tv))}\b", lab, _re.I))
    return True


def process_query_and_candidates(q, docs, hybrid, debug=None):
    """Process query and retrieve candidates with filtering and fallbacks."""
    qa = query_analyzer(q)
    cands = hybrid.invoke(q)
    dense_docs = []
    sparse_docs = []
    
    if debug and debug.get("dense") is not None:
        try:
            dense_docs = debug["dense"].invoke(q)
        except Exception:
            pass
    if debug and debug.get("sparse") is not None:
        try:
            sparse_docs = debug["sparse"].invoke(q)
        except Exception:
            pass
    
    filtered = apply_filters(cands, qa.get("filters", {}))
    
    # section/number fallback if nothing after filtering
    sec = qa.get("filters", {}).get("section") if qa else None
    if sec and not filtered:
        filtered = [d for d in docs if _fallback_ok(d, qa)]
    
    top_docs = rerank_candidates(q, filtered, top_n=8)
    
    # Fallbacks to avoid empty contexts for metrics/answering
    if not top_docs:
        # Use unfiltered candidates
        top_docs = (cands or [])[:8]
    if not top_docs:
        # Last resort use first few indexed docs
        top_docs = docs[:8]
    
    # If the query is about figures, present them in ascending order by number/order for consistency
    try:
        if (qa.get("filters") or {}).get("section") == "Figure":
            top_docs = sorted(top_docs, key=_fig_sort_key)
    except Exception:
        pass
    
    return qa, cands, filtered, top_docs, dense_docs, sparse_docs


def process_figure_list_request(q, qa, docs, r, rtrace):
    """Process special case: list all figures request."""
    if (qa.get("filters") or {}).get("section") == "Figure" and re.search(r"\b(list|all|show)\b.*\bfigures\b", q, re.I):
        # Gather all figure docs and sort by number/order/page
        all_figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure"]
        all_figs = sorted(all_figs, key=_fig_sort_key)
        # Build a clean list from normalized labels
        lines = []
        for d in all_figs:
            md = d.metadata or {}
            label = md.get("figure_label") or d.page_content.splitlines()[0]
            lines.append(f"{label} [{md.get('file_name')} p{md.get('page')} Figure]")
        ans_text = "\n".join(lines) if lines else "(no figures found)"
        router_info = f"Route: {r} | Top contexts: [all figures]"
        trace = f"Keywords: {qa.get('keywords', [])} | Filters: {qa.get('filters', {})}"
        return True, ans_text, router_info, trace, all_figs
    return False, None, None, None, None
