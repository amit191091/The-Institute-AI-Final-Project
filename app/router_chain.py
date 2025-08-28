"""LLM-based Router using LangChain's LLMRouterChain.

This module provides a lightweight, optional routing layer that classifies
incoming questions into destinations: summary | table | graph | needle.

Defaults:
- Prefers Google Gemini for the router model.
- Falls back cleanly to a heuristic router (returns "DEFAULT") if imports fail.

Enable with env RAG_USE_LC_ROUTER=true (default). Disable by setting to 0/false.
"""
from __future__ import annotations

from app.logger import trace_func

import os
from typing import Optional

_ROUTER = None


@trace_func
def _build_router():
    """Build and return an LLMRouterChain, or None if unsupported/missing deps."""
    try:
        from langchain.chains.router import MultiPromptChain  # noqa: F401
        from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
        from langchain.prompts import PromptTemplate
    except Exception:
        return None

    # Prefer Google Gemini for routing (cheap/fast model is fine)
    llm = None
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=os.getenv("ROUTER_CHAT_MODEL", "gemini-1.5-flash"), temperature=0)
        except Exception:
            llm = None
    if llm is None and os.getenv("OPENAI_API_KEY") and os.getenv("ALLOW_OPENAI_ROUTER", "").lower() in ("1","true","yes"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL","gpt-4.1-nano"), temperature=0)
        except Exception:
            llm = None
    if llm is None:
        return None

    prompt_infos = [
        {
            "name": "summary",
            "description": "Summaries, overviews, conclusions across the whole report or multiple tests.",
        },
        {
            "name": "table",
            "description": "Questions about values, charts, tables, figures, transmission ratios, wear depth, sensors, thresholds.",
        },
        {
            "name": "graph",
            "description": "Queries mentioning specific table numbers + keys or requiring Neo4j/graph lookups.",
        },
        {
            "name": "needle",
            "description": "Precise fact lookup in text (dates, names, exact values) when other routes don't match.",
        },
    ]
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = f"""
Given a raw text input to a language model, select the destination best suited for the input.
Return a markdown JSON with a single key "destination" and value in ["summary","table","graph","needle","DEFAULT"].

<< DESTINATIONS >>
{destinations_str}

<< INPUT >>
{{input}}

<< ROUTING DECISION >>
"""

    try:
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        return LLMRouterChain.from_llm(llm, router_prompt)
    except Exception:
        return None


@trace_func
def get_router():
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    _ROUTER = _build_router()
    return _ROUTER


@trace_func
def route_llm(question: str) -> str:
    """Return one of: summary | table | graph | needle | DEFAULT.
    Falls back to DEFAULT if router not available or disabled.
    """
    if os.getenv("RAG_USE_LC_ROUTER", "1").lower() in ("0","false","no"):
        return "DEFAULT"
    rc = get_router()
    if rc is None:
        return "DEFAULT"
    try:
        out = rc.invoke({"input": question})
        dest = (out or {}).get("destination") or "DEFAULT"
        if dest not in ("summary","table","graph","needle"):
            return "DEFAULT"
        return dest
    except Exception:
        return "DEFAULT"
