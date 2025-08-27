from __future__ import annotations

import os
from typing import List, Tuple
from dataclasses import dataclass

from langchain.schema import Document


@dataclass
class RerankConfig:
    provider: str = os.getenv("CE_PROVIDER", "auto")  # auto|openai|google|none
    top_k: int = int(os.getenv("CE_TOP_K", "20"))
    model_openai: str = os.getenv("CE_OPENAI_MODEL", os.getenv("OPENAI_CE_MODEL", "gpt-4.1-mini"))
    model_google: str = os.getenv("CE_GOOGLE_MODEL", os.getenv("GOOGLE_CE_MODEL", "gemini-1.5-flash"))


def _pick_provider() -> str:
    pref = os.getenv("CE_PROVIDER", "auto").lower()
    if pref in ("openai", "google", "none"):
        return pref
    # auto: prefer Google if key present, else OpenAI
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "none"


def _score_pairs_openai(query: str, docs: List[Document], model: str) -> List[Tuple[float, int]]:
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return []
    llm = ChatOpenAI(model=model, temperature=0)
    # Simple pairwise relevance scoring via few-shot rubric
    prompt_head = (
        "You are a precise reranker. Score 0..1 relevance of passage to the query.\n"
        "Return only the number.\n\n"
        f"Query: {query}\n"
    )
    scores: List[Tuple[float, int]] = []
    for i, d in enumerate(docs):
        txt = d.page_content[:1400]
        prompt = prompt_head + "Passage:\n" + txt
        try:
            resp = llm.invoke(prompt)
            s = float(str(getattr(resp, "content", "0")).strip().split()[0])
        except Exception:
            s = 0.0
        scores.append((max(0.0, min(1.0, s)), i))
    return scores


def _score_pairs_google(query: str, docs: List[Document], model: str) -> List[Tuple[float, int]]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        return []
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    prompt_head = (
        "You are a precise reranker. Score 0..1 relevance of passage to the query.\n"
        "Return only the number.\n\n"
        f"Query: {query}\n"
    )
    scores: List[Tuple[float, int]] = []
    for i, d in enumerate(docs):
        txt = d.page_content[:1400]
        prompt = prompt_head + "Passage:\n" + txt
        try:
            resp = llm.invoke(prompt)
            s = float(str(getattr(resp, "content", "0")).strip().split()[0])
        except Exception:
            s = 0.0
        scores.append((max(0.0, min(1.0, s)), i))
    return scores


def rerank_cross_encoder(query: str, candidates: List[Document], top_n: int = 8) -> List[Document]:
    """Cross-encoder reranker using GPT-4.1 mini/nano or Gemini 1.5 flash.
    Falls back to input order if no provider configured. top_n caps final list.
    """
    provider = _pick_provider()
    cfg = RerankConfig()
    if provider == "none" or not candidates:
        return candidates[:top_n]
    pool = candidates[: cfg.top_k]
    if provider == "openai":
        pairs = _score_pairs_openai(query, pool, cfg.model_openai)
    else:
        pairs = _score_pairs_google(query, pool, cfg.model_google)
    if not pairs:
        return candidates[:top_n]
    idx_sorted = [i for _, i in sorted(pairs, key=lambda x: -x[0])]
    seen = set(); out: List[Document] = []
    for i in idx_sorted:
        if 0 <= i < len(pool) and i not in seen:
            out.append(pool[i]); seen.add(i)
        if len(out) >= top_n:
            break
    return out
