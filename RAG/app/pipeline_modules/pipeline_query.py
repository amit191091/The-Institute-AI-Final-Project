#!/usr/bin/env python3
"""
Query Processing Module
======================

Handles query analysis, document retrieval, and answer generation.
"""

import json
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from RAG.app.config import settings
from RAG.app.retrieve import query_analyzer, apply_filters, rerank_candidates, lexical_overlap
from RAG.app.Agent_Components.agents import route_question_ex, answer_summary, answer_table, answer_needle, route_question
from RAG.app.logger import get_logger

# Optional LLM-based router (safe no-op if unavailable)
try:
    from RAG.app.pipeline_modules.router_chain import route_llm  # type: ignore
except Exception:  # pragma: no cover
    def route_llm(question: str) -> str:  # type: ignore
        return "DEFAULT"


def analyze_query(question: str) -> Dict[str, Any]:
    """Analyze a query to extract intent, keywords, and filters."""
    return query_analyzer(question)


def retrieve_candidates(hybrid_retriever, question: str, qa: Dict[str, Any]) -> List:
    """Retrieve candidate documents for a question."""
    q_exec = qa.get("canonical") or question
    
    try:
        candidates = hybrid_retriever.invoke(q_exec)
    except Exception:
        candidates = hybrid_retriever.invoke(q_exec)
        
    candidates = candidates[:settings.embedding.K_TOP_K]  # rerank TOP K
    return candidates


def filter_candidates(candidates: List, qa: Dict[str, Any], docs: List) -> List:
    """Apply filters to candidate documents."""
    filtered = apply_filters(candidates, qa.get("filters", {}))
    
    # Fallback: if no filtered results and section filter exists
    try:
        sec = qa.get("filters", {}).get("section")
    except Exception:
        sec = None
        
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
        
    return filtered


def rerank_documents(question: str, filtered_docs: List) -> List:
    """Rerank filtered documents for relevance."""
    q_exec = question  # Use original question for reranking
    return rerank_candidates(q_exec, filtered_docs, top_n=settings.embedding.CONTEXT_TOP_N)


def calculate_document_score(doc, question: str, qa: Dict[str, Any]) -> float:
    """Calculate relevance score for a document."""
    q_exec = question
    base = lexical_overlap(q_exec, doc.page_content)
    meta_text = " ".join(map(str, (getattr(doc, "metadata", {}) or {}).values()))
    boost = 0.2 * lexical_overlap(" ".join(qa.get("keywords", [])), meta_text)
    return round(base + boost, 4)


def get_document_header(doc) -> str:
    """Get a human-readable header for a document."""
    md = getattr(doc, "metadata", {}) or {}
    return f"{md.get('file_name')} p{md.get('page')} {md.get('section')}#{md.get('anchor', '')}"


def route_question_to_agent(question: str) -> tuple:
    """Route question to appropriate agent."""
    return route_question_ex(question)


def generate_answer(route: str, top_docs: List, question: str, llm) -> str:
    """Generate answer using the appropriate agent."""
    if route == "summary":
        return answer_summary(llm, top_docs, question)
    elif route == "table":
        return answer_table(llm, top_docs, question)
    else:  # needle
        return answer_needle(llm, top_docs, question)


def log_query_results(question: str, route: str, rtrace: Dict, qa: Dict[str, Any], 
                     top_docs: List, answer: str) -> None:
    """Log query results for analysis."""
    try:
        log_dir = settings.paths.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "question": question,
            "route": route,
            "router_trace": rtrace,
            "keywords": qa.get("keywords", []),
            "filters": qa.get("filters", {}),
            "contexts": [
                {
                    "file": d.metadata.get("file_name"),
                    "page": d.metadata.get("page"),
                    "section": d.metadata.get("section"),
                    "anchor": d.metadata.get("anchor"),
                    "score": calculate_document_score(d, question, qa),
                }
                for d in top_docs
            ],
            "answer_preview": (answer or "")[:400],
        }
        
        with open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def answer_question(docs: List, hybrid_retriever, llm, question: str, 
                   ground_truth: Optional[str] = None) -> str:
    """Answer a user question using the hybrid retriever and route to sub-agents."""
    log = get_logger()
    qa = query_analyzer(question)
    q_exec = qa.get("canonical") or question
    try:
        candidates = hybrid_retriever.invoke(q_exec)
    except Exception:
        candidates = hybrid_retriever.invoke(q_exec)
    candidates = candidates[: settings.embedding.K_TOP_K]  # rerank TOP K
    filtered = apply_filters(candidates, qa["filters"])  # metadata filters
    try:
        sec = qa["filters"].get("section")
    except Exception:
        sec = None
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
    top_docs = rerank_candidates(q_exec, filtered, top_n=settings.CONTEXT_TOP_N)
    # Prefer LLM router when enabled; fall back to heuristic router
    route = route_llm(question)
    if route == "DEFAULT":
        route, rtrace = route_question_ex(question)
    else:
        rtrace = {"matched": ["llm_router"], "route": route, "simplified": qa.get("intent", {})}

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
        ans = answer_summary(llm, top_docs, question)
    elif route == "table":
        ans = answer_table(llm, top_docs, question)
    else:
        ans = answer_needle(llm, top_docs, question)
    try:
        log_dir = settings.paths.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
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


def answer_with_contexts(docs: List, hybrid_retriever, llm, question: str) -> tuple[str, List]:
    """Answer a question and also return the contexts used (top_docs)."""
    log = get_logger()
    qa = query_analyzer(question)
    q_exec = question  # Use the original question since we don't have canonical anymore
    try:
        candidates = hybrid_retriever.invoke(q_exec)
    except Exception:
        candidates = hybrid_retriever.invoke(q_exec)
    candidates = candidates[: settings.embedding.K_TOP_K]
    
    # Apply enhanced filtering based on question type
    filtered = candidates
    if qa.get("is_table_question"):
        # For table questions, be more inclusive - include both Table section and Text section that contains table data
        filtered = []
        for d in candidates:
            metadata = d.metadata or {}
            section = metadata.get("section", "")
            content = d.page_content.lower()
            
            # Include if it's marked as Table section
            if section == "Table":
                filtered.append(d)
            # Include if it's Text section but contains table-like data (wear depth, case numbers, etc.)
            elif section == "Text" and any(keyword in content for keyword in settings.query_analysis.WEAR_CASES_MAIN + ["wear depth", "case", "μm", "micron"]):
                filtered.append(d)
            # Include if it contains specific table markers
            elif any(marker in content for marker in ["| case |", "| wear depth |", "healthy,0", "w1,40"]):
                filtered.append(d)
            # ENHANCED: Include if it contains sensor/accelerometer data for sensor-related table questions
            elif any(sensor_term in content for sensor_term in ["accelerometer", "sensor", "dytran", "3053b", "starboard", "port", "shaft", "mv/g", "9.47", "9.35"]):
                filtered.append(d)
            # ENHANCED: Include if it contains transmission/gear data for gear-related table questions
            elif any(gear_term in content for gear_term in ["transmission", "gear", "module", "ratio", "18/35", "3 mm"]):
                filtered.append(d)
    elif qa.get("is_figure_question"):
        filtered = [d for d in candidates if (d.metadata or {}).get("section") == "Figure"]
    
    # If no filtered results, use original candidates
    if not filtered:
        filtered = candidates
    
    top_docs = rerank_candidates(q_exec, filtered, top_n=settings.embedding.CONTEXT_TOP_N)
    if not top_docs:
        top_docs = candidates[: settings.embedding.CONTEXT_TOP_N] if candidates else []
    if not top_docs:
        top_docs = docs[: settings.embedding.CONTEXT_TOP_N]
    # Use LLM router with fallback to heuristic
    route = route_llm(question)
    if route == "DEFAULT":
        route = route_question(question)
    if route == "summary":
        ans = answer_summary(llm, top_docs, question)
    elif route == "table":
        ans = answer_table(llm, top_docs, question)
    else:
        ans = answer_needle(llm, top_docs, question)
    return ans, top_docs


def _apply_enhanced_filtering(candidates: List, qa: Dict[str, Any]) -> List:
    """Apply enhanced filtering based on question type."""
    filtered = candidates
    
    if qa.get("is_table_question"):
        # For table questions, be more inclusive
        filtered = []
        for d in candidates:
            metadata = d.metadata or {}
            section = metadata.get("section", "")
            content = d.page_content.lower()
            
            # Include if it's marked as Table section
            if section == "Table":
                filtered.append(d)
            # Or if it contains table-like content
            elif any(table_indicator in content for table_indicator in [
                "wear depth", "accelerometer", "tachometer"
            ]):
                filtered.append(d)
    
    elif qa.get("is_figure_question"):
        # For figure questions, prioritize Figure sections
        filtered = []
        for d in candidates:
            metadata = d.metadata or {}
            section = metadata.get("section", "")
            content = d.page_content.lower()
            
            # Include if it's marked as Figure section
            if section == "Figure":
                filtered.append(d)
            # Or if it contains figure-related content
            elif any(fig_indicator in content for fig_indicator in [
                "figure", "fig", "plot", "graph", "rms", "fft", "spectrogram"
            ]):
                filtered.append(d)
    
    return filtered if filtered else candidates
