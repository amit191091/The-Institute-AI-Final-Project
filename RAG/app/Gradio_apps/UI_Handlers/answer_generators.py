#!/usr/bin/env python3
"""
Answer Generators
================

Answer generation for different routes and types.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from RAG.app.Agent_Components.agents import answer_needle, answer_summary, answer_table
from RAG.app.Gradio_apps.ui_components import _extract_table_figure_context


def _is_fig_match(d, desired_fig):
    """Check if document matches desired figure number."""
    md = d.metadata or {}
    if (md.get("section") or md.get("section_type")) != "Figure":
        return False
    fn = md.get("figure_number")
    if fn is not None and str(fn).strip().isdigit() and int(str(fn)) == desired_fig:
        return True
    import re as _re
    lab = str(md.get("figure_label") or "")
    return bool(_re.match(rf"^\s*figure\s*{desired_fig}\b", lab, _re.I))


def _matches_want(d, want):
    """Check if document matches desired figure number for display."""
    if want is None:
        return False
    md = d.metadata or {}
    fn = md.get("figure_number")
    if fn is not None and str(fn).strip().isdigit() and int(str(fn)) == want:
        return True
    import re as _re
    lab = str(md.get("figure_label") or md.get("caption") or "")
    return bool(_re.match(rf"^\s*figure\s*{want}\b", lab, _re.I))


def generate_summary_answer(llm, top_docs, q, router_info, trace):
    """Generate summary answer."""
    ans_raw = answer_summary(llm, top_docs, q)
    return f"{router_info}\n\n{ans_raw}\n\n(trace: {trace})"


def generate_table_answer(llm, top_docs, q, qa, docs, router_info, trace):
    """Generate table answer with figure matching and context."""
    # If user asked for a specific figure/table number, prioritize matching docs
    try:
        desired_fig = None
        desired_tbl = None
        if qa and qa.get("filters"):
            fv = qa["filters"].get("figure_number")
            if fv is not None:
                try:
                    desired_fig = int(str(fv))
                except Exception:
                    desired_fig = None
            tv = qa["filters"].get("table_number")
            if tv is not None:
                try:
                    desired_tbl = int(str(tv))
                except Exception:
                    desired_tbl = None
        if desired_fig is not None:
            matches = [d for d in top_docs if _is_fig_match(d, desired_fig)]
            if matches:
                top_docs = matches + [d for d in top_docs if d not in matches]
    except Exception:
        pass
    
    table_ctx = _extract_table_figure_context(top_docs)
    ans_raw = answer_table(llm, top_docs, q)
    
    # If a relevant figure was retrieved, prefer displaying via an Image component
    fig_path = None
    try:
        # Prefer the first doc that matches desired fig number (by metadata or label), else the first figure doc
        _fig_docs = [d for d in top_docs if (d.metadata or {}).get("section") == "Figure" and (d.metadata or {}).get("image_path")]
        want = None
        if qa and qa.get("filters") and qa["filters"].get("figure_number"):
            try:
                want = int(str(qa["filters"]["figure_number"]).strip())
            except Exception:
                want = None
        if want is not None and _fig_docs:
            pref = [d for d in _fig_docs if _matches_want(d, want)]
            fig_doc = pref[0] if pref else (_fig_docs[0] if _fig_docs else None)
        else:
            fig_doc = _fig_docs[0] if _fig_docs else None
        # If still nothing (e.g., not in top docs), try a best-effort lookup across all docs
        if fig_doc is None and want is not None:
            _all_figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure" and (d.metadata or {}).get("image_path")]
            _pref = [d for d in _all_figs if _matches_want(d, want)]
            fig_doc = _pref[0] if _pref else ( _all_figs[0] if _all_figs else None )
        if fig_doc is not None:
            p = Path(str(fig_doc.metadata.get("image_path")))
            if p.exists():
                fig_path = str(p)
    except Exception:
        pass
    
    # We keep markdown preview for table context, and the actual image will show in a Gallery component
    out = f"{router_info}\n\nTable/Figure context preview:\n{table_ctx}\n\n---\n\n{ans_raw}\n\n(trace: {trace})"
    return out, fig_path


def generate_needle_answer(llm, top_docs, q, router_info, trace):
    """Generate needle answer."""
    ans_raw = answer_needle(llm, top_docs, q)
    return f"{router_info}\n\n{ans_raw}\n\n(trace: {trace})"
