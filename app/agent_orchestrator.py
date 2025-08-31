from __future__ import annotations

"""Small ReAct-style orchestrator for visibility.

Emits a structured JSON trace of steps and can optionally produce an answer
by delegating to existing agents (summary/needle/table).

No breaking changes: import and use from UI/pipeline as an optional layer.
"""

from typing import Any, Dict, List, Tuple

from langchain.schema import Document

from app.agent_tools import (
    tool_analyze_query,
    tool_retrieve_candidates,
    tool_retrieve_filtered,
    tool_table_read_kv,
    tool_table_filter,
)
from app.agents import answer_summary, answer_table, answer_needle, route_question_ex
from app.fact_miner import mine_answer_from_context, canonicalize_answer
from app.guards import enforce_domain, filter_to_scope
from app.validators import validate_answer


import re
import os

# ---- cues ----
DELTA_TOKENS = ("by how much", "exceed", "increase", "delta", "Δ", "vs", "versus", "compared to", "baseline")
TABLE_CUES = ("table", "μm", "wear depth", "case w", "odd", "even", "<", "≤", ">", "≥", "module", "ratio", "teeth", "sensitivity", "sampling")
FIGURE_RE = re.compile(r"\b(?:fig(?:ure)?\.?\s*)(\d{1,3})\b", re.I)

def _is_delta_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(tok in ql for tok in DELTA_TOKENS)

def _has_explicit_table_cue(q: str) -> bool:
    ql = (q or "").lower()
    return any(tok in ql for tok in TABLE_CUES)

def _figure_num(q: str):
    m = FIGURE_RE.search(q or "")
    return m.group(1) if m else None

# ---- context checks ----
def _has_table_ctx(ds: List[Document]) -> bool:
    for d in ds or []:
        md = d.metadata or {}
        sec = (md.get("section") or md.get("section_type") or "").lower()
        if sec in ("table", "tablecell") or md.get("table_md_path") or md.get("table_csv_path"):
            return True
    return False

def _has_figure_ctx(ds: List[Document]) -> bool:
    for d in ds or []:
        md = d.metadata or {}
        if (md.get("section") or "").lower() == "figure" and md.get("image"):
            return True
    return False

# ---- verification guards ----
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

def _nums(s: str) -> set:
    return set(_NUM_RE.findall((s or "").lower()))

def _combine_text(ds: List[Document], n: int = 10) -> str:
    return "\n".join((d.page_content or "") for d in (ds or [])[:n])

def number_echo_guard(question: str, context_text: str, answer: str) -> bool:
    """True if answer only reused numbers from the question (not supported by context)."""
    qn, an, cn = _nums(question), _nums(answer), _nums(context_text)
    return bool(an) and an.issubset(qn) and not an.intersection(cn)

def delta_contract_guard(question: str, context_text: str, answer: str) -> bool:
    """For delta questions, require at least one numeric tied to context (not just from query)."""
    if not _is_delta_question(question):
        return False
    an, qn, cn = _nums(answer), _nums(question), _nums(context_text)
    # Must contain a numeric, and at least one should appear in context (not just the query)
    return (not an) or (not an.intersection(cn - qn))


def _doc_head(d: Document) -> str:
    md = d.metadata or {}
    return f"{md.get('file_name')}#p{md.get('page')} {md.get('section')}#{md.get('anchor')}"


# def run(question: str, docs: List[Document], hybrid, llm_callable, do_answer: bool = True) -> Dict[str, Any]:
#     """Run an orchestrated sequence and return a reasoning trace.

#     Keys:
#     - route, rules, signals
#     - steps: [{action, observation}]
#     - top_docs: brief list
#     - answer (optional)
#     """
#     trace: Dict[str, Any] = {"question": question, "route": None, "rules": [], "signals": {}, "steps": []}
#     try:
#         # Router (heuristic+signals)
#         route, rtrace = route_question_ex(question)
#         trace["route"] = route
#         if isinstance(rtrace, dict):
#             trace["rules"] = rtrace.get("matched", [])
#             trace["signals"] = rtrace.get("signals", {})
#         # Analyze
#         if tool_analyze_query:
#             qa = tool_analyze_query(question)
#             trace["steps"].append({"action": "analyze_query", "observation": qa})
#         # Retrieve
#         cands = []
#         try:
#             cands = hybrid.invoke(question) or []
#         except Exception:
#             cands = []
#         trace["steps"].append({"action": "retrieve_candidates", "observation_count": len(cands)})
#         # Filter + rerank
#         top_docs: List[Document] = []
#         fr = {}
#         if tool_retrieve_filtered:
#             fr = tool_retrieve_filtered(question, docs, hybrid)
#             trace["steps"].append({"action": "filter+rerank", "observation": {"top_count": len(fr.get("top_docs", []))}})
#         # Collect detailed top docs heads for transparency
#         try:
#             # Prefer the actual Document objects from the last rerank when available
#             if fr and fr.get("top_docs"):
#                 # agent_tools returns brief dicts; fall back to hybrid invoke slice
#                 pass
#             top_docs = []
#             try:
#                 # Best effort: re-run rerank to capture actual docs for answering
#                 from app.retrieve import rerank_candidates, apply_filters, query_analyzer
#                 qa2 = query_analyzer(question)
#                 cands2 = hybrid.invoke(qa2.get("canonical") or question) or []
#                 filtered2 = apply_filters(cands2, qa2.get("filters") or {})
#                 # Increase coverage for instrumentation/speed style questions
#                 sigs = trace.get("signals") or {}
#                 boost_tables = bool(sigs.get("has_sampling_tokens") or sigs.get("has_sensor_tokens"))
#                 top_n = 16 if (route in ("table", "graph") and boost_tables) else 8
#                 top_docs = rerank_candidates(qa2.get("canonical") or question, filtered2, top_n=top_n)
#                 # Ensure table-like docs appear when requested
#                 if route in ("table", "graph") and boost_tables:
#                     extra_tables = []
#                     for d in filtered2:
#                         md = d.metadata or {}
#                         sec = md.get("section") or md.get("section_type")
#                         if sec in ("Table", "TableCell"):
#                             extra_tables.append(d)
#                     # Prepend unique table docs while preserving order
#                     seen = set(id(x) for x in top_docs)
#                     merged = []
#                     for d in extra_tables + top_docs:
#                         did = id(d)
#                         if did in seen:
#                             continue
#                         seen.add(did)
#                         merged.append(d)
#                     top_docs = merged[:top_n]
#                 # Modality questions: ensure imaging/text pages are present alongside sensors
#                 try:
#                     ql = (question or "").lower()
#                     if ("sensor" in ql or "sensors" in ql or "modality" in ql or "modalities" in ql) and ("document" in ql or "wear" in ql or "progression" in ql):
#                         imaging_terms = ("microscope", "microscopy", "photograph", "photography", "imaging", "image", "camera")
#                         addl = []
#                         for d in filtered2:
#                             try:
#                                 t = (d.page_content or "").lower()
#                                 if any(term in t for term in imaging_terms):
#                                     addl.append(d)
#                             except Exception:
#                                 continue
#                         if addl:
#                             seen = set(id(x) for x in top_docs)
#                             merged = []
#                             for d in addl + top_docs:
#                                 did = id(d)
#                                 if did in seen:
#                                     continue
#                                 seen.add(did)
#                                 merged.append(d)
#                             # modestly expand window to keep both tables and imaging
#                             top_docs = merged[: max(top_n, 16)]
#                 except Exception:
#                     pass
#             except Exception:
#                 top_docs = (cands or [])[:8]
#             trace["top_docs"] = [{
#                 "file": (d.metadata or {}).get("file_name"),
#                 "page": (d.metadata or {}).get("page"),
#                 "section": (d.metadata or {}).get("section") or (d.metadata or {}).get("section_type"),
#                 "anchor": (d.metadata or {}).get("anchor"),
#                 "score": (d.metadata or {}).get("_score"),
#             } for d in top_docs]
#         except Exception:
#             pass
#         # Opportunistic deterministic table read when routed to table
#         try:
#             if route in ("table", "graph"):
#                 # Collect candidate table markdown paths from top docs
#                 md_paths = []
#                 for d in top_docs or []:
#                     try:
#                         p = (d.metadata or {}).get("table_md_path")
#                         if p:
#                             md_paths.append(p)
#                     except Exception:
#                         pass
#                 md_paths = list(dict.fromkeys(md_paths))[:3]
#                 # Heuristic keys from the question
#                 ql = (question or "").lower()
#                 keys: List[str] = []
#                 if any(k in ql for k in ["sensitivity", "mv/g", "mv per g", "mvg"]):
#                     keys.append("sensitivity")
#                 if any(k in ql for k in ["sampling", "sample rate", "sampling rate", "hz", "khz", "ks/s", "ksps"]):
#                     keys.append("sampling rate")
#                 if not keys:
#                     # Default to two common instrumentation fields
#                     keys = ["sensitivity", "sampling rate"]
#                 table_obs = {"keys": keys, "tables": []}
#                 for p in md_paths:
#                     try:
#                         kv = tool_table_read_kv(p, keys) if tool_table_read_kv else {"error": "tool unavailable"}
#                         table_obs["tables"].append({"path": p, "kv": kv})
#                     except Exception as e:  # pragma: no cover
#                         table_obs["tables"].append({"path": p, "error": str(e)})
#                 # Add a light filter run too when helpful
#                 try:
#                     if tool_table_filter and md_paths:
#                         constraints = {"contains": [w for w in ql.split() if len(w) > 2][:3]}
#                         filt = tool_table_filter(md_paths[0], constraints)
#                         table_obs["filter_example"] = {"path": md_paths[0], "constraints": constraints, "result": filt}
#                 except Exception:
#                     pass
#                 trace["steps"].append({"action": "table_det_read", "observation": table_obs})
#         except Exception:
#             pass

#         # Optional answer using existing agents
#         if do_answer:
#             # First: try deterministic fact miner over concatenated top_docs
#             try:
#                 mined, mined_meta = mine_answer_from_context(question, top_docs)
#             except Exception:
#                 mined, mined_meta = (None, {})
#             if mined:
#                 ans = mined
#                 trace["steps"].append({"action": "fact_miner", "observation": mined_meta})
#             else:
#                 if route == "summary":
#                     ans = answer_summary(llm_callable, top_docs, question)
#                 elif route in ("table", "graph"):
#                     ans = answer_table(llm_callable, top_docs, question)
#                 else:
#                     ans = answer_needle(llm_callable, top_docs, question)
#             # Canonicalize phrasing to reduce drift when equivalent
#             try:
#                 if ans:
#                     ans = canonicalize_answer(question, ans)
#             except Exception:
#                 pass
#             trace["answer"] = ans
#         return trace
#     except Exception as e:
#         trace.setdefault("errors", []).append(str(e))
#         return trace


def run(question: str, docs: List[Document], hybrid, llm_callable, do_answer: bool = True) -> Dict[str, Any]:
    """
    Orchestrated sequence with a multi-stage fallback strategy for robustness.
    1. Strict Path: Tries the pipeline with a strict domain filter.
    2. Relaxed Path: If the first fails, it retries with a looser domain filter.
    3. Forced Needle Path: If both fail, it overrides routing and uses the general-purpose agent.
    """
    trace: Dict[str, Any] = {"question": question, "route": None, "rules": [], "signals": {}, "steps": []}
    
    try:
        # -------------------
        # 1) Route and Analyze
        # -------------------
        route, rtrace = route_question_ex(question)
        trace["route"] = route
        if isinstance(rtrace, dict):
            trace["rules"] = rtrace.get("matched", [])
            trace["signals"] = rtrace.get("signals", {})

        if _is_delta_question(question) and not _has_explicit_table_cue(question):
            trace["rules"] = list(set(trace.get("rules", []) + ["delta_override"]))
            route = "needle"
        
        fig_n = _figure_num(question)
        figure_intent = bool(fig_n)

        if tool_analyze_query:
            try:
                qa = tool_analyze_query(question)
                trace["steps"].append({"action": "analyze_query", "observation": qa})
            except Exception:
                pass

        # -------------------
        # 2) Initial Retrieval
        # -------------------
        cands = hybrid.invoke(question) or []
        trace["steps"].append({"action": "retrieve_candidates", "observation_count": len(cands)})

        # Use a consistent reranking and filtering logic
        from app.retrieve import rerank_candidates, apply_filters, query_analyzer
        qa2 = query_analyzer(question)
        cands2 = hybrid.invoke(qa2.get("canonical") or question) or []
        filtered2 = apply_filters(cands2, qa2.get("filters") or {})
        
        sigs = trace.get("signals") or {}
        boost_tables = bool(sigs.get("has_sampling_tokens") or sigs.get("has_sensor_tokens"))
        top_n = 16 if (route in ("table", "graph") and boost_tables) else 8
        
        initial_top_docs = rerank_candidates(qa2.get("canonical") or question, filtered2, top_n=top_n)

        # Store a concise view of initial docs in the trace
        trace["top_docs"] = [{"file": (d.metadata or {}).get("file_name"), "page": (d.metadata or {}).get("page"), "section": (d.metadata or {}).get("section"), "anchor": (d.metadata or {}).get("anchor")} for d in initial_top_docs]

        # ==================================================================
        # ATTEMPT 1: STRICT PATH
        # ==================================================================
        trace["steps"].append({"action": "attempt", "observation": "strict_path"})
        
        # Apply strict domain gate
        strict_docs = enforce_domain(initial_top_docs, min_share=0.75)
        trace["steps"].append({"action": "domain_gate_strict", "observation": {"before": len(initial_top_docs), "after": len(strict_docs), "min_share": 0.75}})

        ans = "Not found in context."
        if strict_docs:
            ans, _ = _run_answer_stage(question, strict_docs, route, llm_callable, trace, do_answer)

        if ans != "Not found in context.":
            trace["answer"] = ans
            return trace

        # ==================================================================
        # ATTEMPT 2: RELAXED DOMAIN PATH
        # ==================================================================
        trace["steps"].append({"action": "attempt", "observation": "relaxed_domain_path"})
        
        # Apply relaxed domain gate
        relaxed_docs = enforce_domain(initial_top_docs, min_share=0.25)
        trace["steps"].append({"action": "domain_gate_relaxed", "observation": {"before": len(initial_top_docs), "after": len(relaxed_docs), "min_share": 0.25}})
        
        if relaxed_docs:
            ans, _ = _run_answer_stage(question, relaxed_docs, route, llm_callable, trace, do_answer)

        if ans != "Not found in context.":
            trace["answer"] = ans
            return trace

        # ==================================================================
        # ATTEMPT 3: FORCED NEEDLE FALLBACK
        # ==================================================================
        trace["steps"].append({"action": "attempt", "observation": "forced_needle_fallback"})
        
        # Force route to needle and re-retrieve, bypassing domain gate
        final_route = "needle"
        trace["steps"].append({"action": "route_override", "observation": f"Forcing route to {final_route}"})
        
        # Re-retrieve with a broader context
        fallback_docs = rerank_candidates(qa2.get("canonical") or question, filtered2, top_n=12)
        
        ans, _ = _run_answer_stage(question, fallback_docs, final_route, llm_callable, trace, do_answer)
        
        trace["answer"] = ans
        return trace

    except Exception as e:
        trace.setdefault("errors", []).append(str(e))
        return trace

def _run_answer_stage(
    question: str,
    top_docs: List[Document],
    route: str,
    llm_callable,
    trace: Dict[str, Any],
    do_answer: bool = True,
) -> Tuple[str, List[Document]]:
    """Runs the answering and validation stages for a given set of documents."""
    ans = None
    if not do_answer:
        return "", top_docs

    # 1) Answering
    mined = None
    if not _is_delta_question(question):
        try:
            mined, mined_meta = mine_answer_from_context(question, top_docs)
        except Exception:
            mined, mined_meta = (None, {})
    
    if mined:
        ans = mined
        trace["steps"].append({"action": "fact_miner", "observation": mined_meta})
    else:
        if route == "summary":
            ans = answer_summary(llm_callable, top_docs, question)
        elif route in ("table", "graph"):
            ans = answer_table(llm_callable, top_docs, question)
        else:
            ans = answer_needle(llm_callable, top_docs, question)

    try:
        if ans:
            ans = canonicalize_answer(question, ans)
    except Exception:
        pass

    # 2) Guards and Validation
    try:
        if route in ("needle", "table") and (not ans or not ans.strip()):
            trace.setdefault("steps", []).append({"action": "guard", "observation": "empty_answer_reject"})
            ans = "Not found in context."
    except Exception:
        pass

    try:
        ok, why = validate_answer(question, ans or "")
        trace.setdefault("steps", []).append({"action": "answer_validate", "observation": {"ok": ok, "why": why}})
        if not ok:
            try:
                mined2, meta2 = mine_answer_from_context(question, top_docs)
                if mined2:
                    ans = mined2
                    trace.setdefault("steps", []).append({"action": "fact_miner_retry", "observation": meta2})
                else:
                    ans = "Not found in context."
            except Exception:
                ans = "Not found in context."
    except Exception:
        pass

    # 3) Post-answer verification
    try:
        ctx_text = _combine_text(top_docs, n=10)
        echo_bad = number_echo_guard(question, ctx_text, ans or "")
        delta_bad = delta_contract_guard(question, ctx_text, ans or "")

        trace["steps"].append({
            "action": "post_verify",
            "observation": {"number_echo_guard": echo_bad, "delta_contract_guard": delta_bad}
        })

        if echo_bad or delta_bad:
            # This is a candidate for a fallback, but for now, we just trace it.
            # The main `run` loop will handle the fallback.
            trace["steps"].append({"action": "post_verify_failed", "observation": "Answer rejected by guards"})
            ans = "Not found in context."

    except Exception:
        pass

    return ans or "Not found in context.", top_docs
