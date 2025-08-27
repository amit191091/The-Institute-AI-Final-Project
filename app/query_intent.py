"""LLM-powered query intent detection (optional).
Falls back to regex-based simplify_question when LLM unavailable or disabled.

Enable via env:
  RAG_USE_LLM_ROUTER=true|1
Optionally configure models with:
  OPENAI_API_KEY / GOOGLE_API_KEY
  OPENAI_CHAT_MODEL / GOOGLE_CHAT_MODEL
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Any

# Local fallback
from app.agents import simplify_question

_CACHE: dict[str, Dict[str, Any]] = {}


def _setup_llm():
    provider = (os.getenv("RAG_INTENT_PROVIDER") or "").lower().strip()
    try_openai_first = provider == "openai" or (provider == "" and os.getenv("OPENAI_API_KEY"))
    try_google_next = provider == "google" or (provider == "" and os.getenv("GOOGLE_API_KEY"))

    # Prefer smallest/cheapest for routing
    if try_openai_first:
        try:
            from langchain_openai import ChatOpenAI
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            return ChatOpenAI(model=model, temperature=0)
        except Exception:
            pass
    if try_google_next:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(model=model, temperature=0, convert_system_message_to_human=True)
        except Exception:
            pass
    return None


_PROMPT = (
    "You classify a user question about a technical PDF report.\n"
    "Return ONLY compact JSON with keys: \n"
    "  wants_summary, wants_table, wants_figure, wants_date, wants_value (booleans),\n"
    "  table_number, figure_number, case_id, target_attr, event (nullable strings),\n"
    "  canonical (string).\n"
    "target_attr can be one of: 'wear depth', 'rms', 'crest factor', 'fme', 'sensors', or null.\n"
    "event examples: 'failure date', 'measurement start date', 'healthy through date'.\n"
    "Examples of table-like intents: sensors list, instrumentation, thresholds, values, wear depth.\n"
    "Question: {q}\n"
    "JSON:"
)


def _safe_json_extract(text: str) -> Dict[str, Any]:
    # Try raw parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def analyze_query_llm(q: str) -> Dict[str, Any] | None:
    if not q:
        return None
    if os.getenv("RAG_USE_LLM_ROUTER", "0").lower() not in ("1", "true", "yes"):  # disabled
        return None
    qn = (q or "").strip().lower()
    if qn in _CACHE:
        return _CACHE[qn]
    llm = _setup_llm()
    if llm is None:
        return None
    try:
        resp = llm.invoke(_PROMPT.format(q=q))  # type: ignore[attr-defined]
        raw = resp.content if hasattr(resp, "content") else resp
        # Normalize to string for JSON extraction
        if isinstance(raw, (list, tuple)):
            txt = "\n".join([str(x) for x in raw])
        else:
            txt = str(raw)
        data = _safe_json_extract(txt)
        if not isinstance(data, dict):
            raise ValueError("no JSON")
        # Coerce fields
        def _b(x):
            return bool(x) if isinstance(x, (bool, int, float)) else str(x).lower() in ("1","true","yes")
        out: Dict[str, Any] = {
            "wants_summary": _b(data.get("wants_summary")),
            "wants_table": _b(data.get("wants_table")),
            "wants_figure": _b(data.get("wants_figure")),
            "wants_date": _b(data.get("wants_date")),
            "wants_value": _b(data.get("wants_value")),
            "table_number": (str(data.get("table_number")).strip() or None) if data.get("table_number") is not None else None,
            "figure_number": (str(data.get("figure_number")).strip() or None) if data.get("figure_number") is not None else None,
            "case_id": (str(data.get("case_id")).strip() or None) if data.get("case_id") else None,
            "target_attr": (str(data.get("target_attr")).strip() or None) if data.get("target_attr") else None,
            "event": (str(data.get("event")).strip() or None) if data.get("event") else None,
            "canonical": (str(data.get("canonical")).strip() or None) if data.get("canonical") else None,
        }
        _CACHE[qn] = out
        return out
    except Exception:
        return None


def get_intent(q: str) -> Dict[str, Any]:
    """Return intent dict. Prefer LLM when enabled, fallback to regex simplify_question."""
    data = analyze_query_llm(q)
    if data is None:
        return simplify_question(q)
    return data
