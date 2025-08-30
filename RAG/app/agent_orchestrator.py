from __future__ import annotations

"""Small ReAct-style orchestrator for visibility.

Emits a structured JSON trace of steps and can optionally produce an answer
by delegating to existing agents (summary/needle/table).

No breaking changes: import and use from UI/pipeline as an optional layer.
"""

from typing import Any, Dict, List, Tuple

from langchain.schema import Document

from RAG.app.Agent_Components.agent_tools import (
    tool_analyze_query,
    tool_retrieve_candidates,
    tool_retrieve_filtered,
    tool_table_read_kv,
    tool_table_filter,
)
from RAG.app.Agent_Components.agents import answer_summary, answer_table, answer_needle, route_question_ex
from RAG.app.fact_miner import mine_answer_from_context, canonicalize_answer


def _doc_head(d: Document) -> str:
    md = d.metadata or {}
    return f"{md.get('file_name')}#p{md.get('page')} {md.get('section')}#{md.get('anchor')}"


def run(question: str, docs: List[Document], hybrid, llm_callable, do_answer: bool = True) -> Dict[str, Any]:
    """Run an orchestrated sequence and return a reasoning trace.

    Keys:
    - route, rules, signals
    - steps: [{action, observation}]
    - top_docs: brief list
    - answer (optional)
    """
    trace: Dict[str, Any] = {"question": question, "route": None, "rules": [], "signals": {}, "steps": []}
    try:
        # Router (heuristic+signals)
        route, rtrace = route_question_ex(question)
        trace["route"] = route
        if isinstance(rtrace, dict):
            trace["rules"] = rtrace.get("matched", [])
            trace["signals"] = rtrace.get("signals", {})
        # Analyze
        if tool_analyze_query:
            qa = tool_analyze_query(question)
            trace["steps"].append({"action": "analyze_query", "observation": qa})
        # Retrieve
        cands = []
        try:
            cands = hybrid.invoke(question) or []
        except Exception as e:
            cands = []
        trace["steps"].append({"action": "retrieve_candidates", "observation_count": len(cands)})
        # Filter + rerank
        top_docs: List[Document] = []
        fr = {}
        if tool_retrieve_filtered:
            fr = tool_retrieve_filtered(question, docs, hybrid)
            trace["steps"].append({"action": "filter+rerank", "observation": {"top_count": len(fr.get("top_docs", []))}})
        # Collect detailed top docs heads for transparency
        try:
            # Prefer the actual Document objects from the last rerank when available
            if fr and fr.get("top_docs"):
                # agent_tools returns brief dicts; fall back to hybrid invoke slice
                pass
            top_docs = []
            try:
                # Best effort: re-run rerank to capture actual docs for answering
                from RAG.app.retrieve import rerank_candidates, apply_filters, query_analyzer
                qa2 = query_analyzer(question)
                cands2 = hybrid.invoke(qa2.get("canonical") or question) or []
                filtered2 = apply_filters(cands2, qa2.get("filters") or {})
                # Increase coverage for instrumentation/speed style questions
                sigs = trace.get("signals") or {}
                boost_tables = bool(sigs.get("has_sampling_tokens") or sigs.get("has_sensor_tokens"))
                top_n = 16 if (route in ("table", "graph") and boost_tables) else 8
                top_docs = rerank_candidates(qa2.get("canonical") or question, filtered2, top_n=top_n)
                # Ensure table-like docs appear when requested
                if route in ("table", "graph") and boost_tables:
                    extra_tables = []
                    for d in filtered2:
                        md = d.metadata or {}
                        sec = md.get("section") or md.get("section_type")
                        if sec in ("Table", "TableCell"):
                            extra_tables.append(d)
                    # Prepend unique table docs while preserving order
                    seen = set(id(x) for x in top_docs)
                    merged = []
                    for d in extra_tables + top_docs:
                        did = id(d)
                        if did in seen:
                            continue
                        seen.add(did)
                        merged.append(d)
                    top_docs = merged[:top_n]
                # Modality questions: ensure imaging/text pages are present alongside sensors
                try:
                    ql = (question or "").lower()
                    if ("sensor" in ql or "sensors" in ql or "modality" in ql or "modalities" in ql) and ("document" in ql or "wear" in ql or "progression" in ql):
                        imaging_terms = ("microscope", "microscopy", "photograph", "photography", "imaging", "image", "camera")
                        addl = []
                        for d in filtered2:
                            try:
                                t = (d.page_content or "").lower()
                                if any(term in t for term in imaging_terms):
                                    addl.append(d)
                            except Exception as e:
                                continue
                        if addl:
                            seen = set(id(x) for x in top_docs)
                            merged = []
                            for d in addl + top_docs:
                                did = id(d)
                                if did in seen:
                                    continue
                                seen.add(did)
                                merged.append(d)
                            # modestly expand window to keep both tables and imaging
                            top_docs = merged[: max(top_n, 16)]
                except Exception as e:
                    pass
            except Exception as e:
                top_docs = (cands or [])[:8]
            trace["top_docs"] = [{
                "file": (d.metadata or {}).get("file_name"),
                "page": (d.metadata or {}).get("page"),
                "section": (d.metadata or {}).get("section") or (d.metadata or {}).get("section_type"),
                "anchor": (d.metadata or {}).get("anchor"),
                "score": (d.metadata or {}).get("_score"),
            } for d in top_docs]
        except Exception as e:
            pass
        # Opportunistic deterministic table read when routed to table
        try:
            if route in ("table", "graph"):
                # Collect candidate table markdown paths from top docs
                md_paths = []
                for d in top_docs or []:
                    try:
                        p = (d.metadata or {}).get("table_md_path")
                        if p:
                            md_paths.append(p)
                    except Exception as e:
                        pass
                md_paths = list(dict.fromkeys(md_paths))[:3]
                # Heuristic keys from the question
                ql = (question or "").lower()
                keys: List[str] = []
                if any(k in ql for k in ["sensitivity", "mv/g", "mv per g", "mvg"]):
                    keys.append("sensitivity")
                if any(k in ql for k in ["sampling", "sample rate", "sampling rate", "hz", "khz", "ks/s", "ksps"]):
                    keys.append("sampling rate")
                if not keys:
                    # Default to two common instrumentation fields
                    keys = ["sensitivity", "sampling rate"]
                table_obs = {"keys": keys, "tables": []}
                for p in md_paths:
                    try:
                        kv = tool_table_read_kv(p, keys) if tool_table_read_kv else {"error": "tool unavailable"}
                        table_obs["tables"].append({"path": p, "kv": kv})
                    except Exception as e:  # pragma: no cover
                        table_obs["tables"].append({"path": p, "error": str(e)})
                # Add a light filter run too when helpful
                try:
                    if tool_table_filter and md_paths:
                        constraints = {"contains": [w for w in ql.split() if len(w) > 2][:3]}
                        filt = tool_table_filter(md_paths[0], constraints)
                        table_obs["filter_example"] = {"path": md_paths[0], "constraints": constraints, "result": filt}
                except Exception as e:
                    pass
                trace["steps"].append({"action": "table_det_read", "observation": table_obs})
        except Exception as e:
            pass

        # Optional answer using existing agents
        if do_answer:
            # First: try deterministic fact miner over concatenated top_docs
            try:
                mined, mined_meta = mine_answer_from_context(question, top_docs)
            except Exception as e:
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
            # Canonicalize phrasing to reduce drift when equivalent
            try:
                if ans:
                    ans = canonicalize_answer(question, ans)
            except Exception as e:
                pass
            trace["answer"] = ans
        return trace
    except Exception as e:
        trace.setdefault("errors", []).append(str(e))
        return trace
