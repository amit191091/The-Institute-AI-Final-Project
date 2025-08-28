from __future__ import annotations

"""LLM-based router using LangChain's LLMRouterChain.

Decides between destinations: summary | table | graph | needle | DEFAULT
This module is intentionally self-contained and optional. If LangChain or
provider backends are unavailable, it degrades gracefully to DEFAULT.
"""

import os
from typing import Optional

try:
    from langchain.chains.router import MultiPromptChain  # noqa: F401
    from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain.prompts import PromptTemplate
except Exception:  # pragma: no cover
    LLMRouterChain = None  # type: ignore
    RouterOutputParser = None  # type: ignore
    PromptTemplate = None  # type: ignore

_ROUTER = None  # cached router instance


def _build_router_llm():
    """Prefer Gemini Pro for routing; fallback to OpenAI if forced.
    Returns an LCEL-compatible chat model or None.
    """
    force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
    # Prefer Google Gemini unless forced OpenAI
    if os.getenv("GOOGLE_API_KEY") and not force_openai:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        except Exception:
            pass
    # Fallback to OpenAI (or forced)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            try:
                return ChatOpenAI(model=model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
            except Exception:
                return ChatOpenAI(model_name=model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[call-arg]
        except Exception:
            pass
    return None


def _router_prompt_template():
    destinations = [
        {
            "name": "summary",
            "description": "High-level report synopsis, overall conclusions, executive summary, recommendations.",
        },
        {
            "name": "table",
            "description": "Questions about numeric values, units, thresholds, measurements, figures/tables.",
        },
        {
            "name": "graph",
            "description": "Entity/relationship reasoning and connectivity over a knowledge graph (Neo4j).",
        },
        {
            "name": "needle",
            "description": "Direct factual lookup in context, precise value/date extraction, citations.",
        },
    ]
    dest_lines = [f"{d['name']}: {d['description']}" for d in destinations]
    destinations_str = "\n".join(dest_lines)
    template = f"""
Given a raw text input to a language model, select the model prompt best suited for the input.
You will be given the names of the available prompts and a description of what the prompt is best suited for.
You will be returning a markdown JSON blob with a single key "destination" and a value of a single prompt name.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{
    "destination": string // name of the prompt to use or "DEFAULT"
}}

<< DESTINATIONS >>
{destinations_str}
<< INPUT >>
{{input}}
<< ROUTING DECISION >>
"""
    return template


def get_router():
    """Build or return a cached LLMRouterChain instance."""
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    # Respect existing flag name, but default to enabled (1)
    enabled = os.getenv("RAG_USE_LLM_ROUTER", "1").lower() in ("1", "true", "yes")
    if not enabled:
        return None
    if LLMRouterChain is None or PromptTemplate is None or RouterOutputParser is None:
        return None
    llm = _build_router_llm()
    if llm is None:
        return None
    router_prompt = PromptTemplate(
        template=_router_prompt_template(),
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    try:
        _ROUTER = LLMRouterChain.from_llm(llm, router_prompt)
    except Exception:
        _ROUTER = None
    return _ROUTER


def route_llm(question: str) -> str:
    """Return one of: summary | table | graph | needle | DEFAULT.
    Falls back to DEFAULT on any error or when disabled.
    """
    try:
        router = get_router()
        if router is None:
            return "DEFAULT"
        res = router.invoke({"input": question})
        # LLMRouterChain returns a dict with key 'destination'
        dest = (res or {}).get("destination") if isinstance(res, dict) else None
        if not isinstance(dest, str) or not dest:
            return "DEFAULT"
        dest_norm = dest.strip().lower()
        if dest_norm in ("summary", "table", "graph", "needle"):
            return dest_norm
        return "DEFAULT"
    except Exception:
        return "DEFAULT"
