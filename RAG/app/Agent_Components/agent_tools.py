from __future__ import annotations
from typing import Any, Dict, List, Tuple

from langchain.schema import Document

# Thin wrappers around existing pipeline functions to expose as agent tools
from RAG.app.retrieve import query_analyzer, apply_filters, rerank_candidates
from RAG.app.Agent_Components.prompts import PLANNER_SYSTEM, PLANNER_PROMPT


def _doc_brief(d: Document) -> Dict[str, Any]:
    md = d.metadata or {}
    return {
        "file": md.get("file_name"),
        "page": md.get("page"),
        "section": md.get("section") or md.get("section_type"),
        "figure_number": md.get("figure_number"),
        "table_number": md.get("table_number"),
        "anchor": md.get("anchor"),
        "image_path": md.get("image_path"),
        "preview": (d.page_content or "")[:180],
    }


def tool_analyze_query(question: str) -> Dict[str, Any]:
    qa = query_analyzer(question)
    return {
        "canonical": qa.get("canonical"),
        "filters": qa.get("filters"),
        "keywords": qa.get("keywords"),
    }


def tool_retrieve_candidates(question: str, hybrid) -> List[Dict[str, Any]]:
    try:
        cands = hybrid.invoke(question) or []
    except Exception:
        cands = []
    return [_doc_brief(d) for d in cands[:20]]


def tool_rerank(question: str, candidates: List[Document], top_n: int = 8) -> List[Dict[str, Any]]:
    top = rerank_candidates(question, candidates, top_n=top_n)
    return [_doc_brief(d) for d in top]


def tool_retrieve_filtered(question: str, docs: List[Document], hybrid) -> Dict[str, Any]:
    qa = query_analyzer(question)
    try:
        cands = hybrid.invoke(qa.get("canonical") or question) or []
    except Exception:
        cands = []
    filtered = apply_filters(cands, qa.get("filters") or {})
    top = rerank_candidates(qa.get("canonical") or question, filtered, top_n=8)
    return {
        "qa": qa,
        "top_docs": [_doc_brief(d) for d in top],
    }


def tool_list_figures(docs: List[Document]) -> List[Dict[str, Any]]:
    figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure"]
    def _key(d):
        md = d.metadata or {}
        fn = md.get("figure_number")
        fo = md.get("figure_order")
        pg = md.get("page")
        try:
            fnv = int(fn) if fn is not None and str(fn).strip().isdigit() else 10**9
        except Exception:
            fnv = 10**9
        try:
            fov = int(fo) if fo is not None and str(fo).strip().isdigit() else 10**9
        except Exception:
            fov = 10**9
        try:
            pgv = int(pg) if pg is not None and str(pg).strip().isdigit() else 10**9
        except Exception:
            pgv = 10**9
        return (fnv, fov, pgv, str(md.get("anchor") or ""))
    figs_sorted = sorted(figs, key=_key)
    out = []
    for d in figs_sorted:
        md = d.metadata or {}
        out.append({
            "figure_number": md.get("figure_number"),
            "label": md.get("figure_label") or (d.page_content or "").splitlines()[0],
            "image_path": md.get("image_path"),
            "file": md.get("file_name"),
            "page": md.get("page"),
            "anchor": md.get("anchor"),
        })
    return out


def tool_audit_and_fill_figures(docs: List[Document]) -> Dict[str, Any]:
    """Ensure each Figure has a numeric figure_number and sequential figure_order per file.
    - figure_number: keep existing numeric; if missing/non-numeric, infer by order encountered per file
    - figure_order: set as 1..N per file (stable by page, anchor)
    Returns a summary with counts and a sample of changes.
    Note: This mutates in-memory docs.metadata only for the running session/UI; persistence is up to the caller.
    """
    # Group figures by file
    figures: Dict[str, List[Tuple[int, str, Document]]] = {}
    for d in docs:
        md = d.metadata or {}
        if (md.get("section") or md.get("section_type")) != "Figure":
            continue
        file = str(md.get("file_name") or "")
        page = md.get("page")
        try:
            page_i = int(page) if page is not None and str(page).strip().isdigit() else 10**9
        except Exception:
            page_i = 10**9
        anchor = str(md.get("anchor") or "")
        figures.setdefault(file, []).append((page_i, anchor, d))

    total = 0
    filled_number = 0
    fixed_order = 0
    changes: List[Dict[str, Any]] = []
    # For each file, sort and assign
    for file, lst in figures.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        # Determine starting idx; also collect existing valid numbers to keep when present
        idx = 1
        for i, (_, _, d) in enumerate(lst, start=1):
            md = d.metadata or {}
            total += 1
            # figure_order: always set sequentially per file
            prev_order = md.get("figure_order")
            md["figure_order"] = i
            if prev_order != i:
                fixed_order += 1
            # figure_number: keep numeric if present; else assign current idx
            fn = md.get("figure_number")
            try:
                ok_num = fn is not None and str(fn).strip().isdigit()
            except Exception:
                ok_num = False
            if not ok_num:
                md["figure_number"] = idx
                filled_number += 1
            # Improve label when missing/blank
            label = (md.get("figure_label") or "").strip()
            if not label:
                caption = (md.get("caption") or "").strip()
                if caption:
                    md["figure_label"] = f"Figure {md['figure_number']}: {caption}"
                else:
                    md["figure_label"] = f"Figure {md['figure_number']}"
            # record a sample change
            if len(changes) < 12:
                changes.append({
                    "file": file,
                    "page": md.get("page"),
                    "anchor": md.get("anchor"),
                    "figure_number": md.get("figure_number"),
                    "figure_order": md.get("figure_order"),
                    "label": md.get("figure_label"),
                })
            idx += 1
    return {
        "figures_seen": total,
        "figure_numbers_filled": filled_number,
        "figure_orders_set": fixed_order,
        "sample": changes,
    }


def tool_plan(observations: str, llm_callable) -> str:
    """Run a short planner prompt against the configured LLM."""
    prompt = PLANNER_SYSTEM + "\n\n" + PLANNER_PROMPT.format(observations=observations or "(none)")
    try:
        return llm_callable(prompt)
    except Exception as e:
        return f"(planner failed: {e})\n\n{prompt}"
