"""Optional Cross-Encoder reranker.

Enable with env RAG_USE_CE_RERANKER=true and optionally set RAG_CE_MODEL.
Defaults to a small, fast model.
"""
from __future__ import annotations

from RAG.app.logger import trace_func

import os
from typing import List, Optional

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

from langchain.schema import Document

_MODEL: Optional[object] = None

@trace_func
def get_ce_reranker() -> Optional[object]:
    global _MODEL
    if os.getenv("RAG_USE_CE_RERANKER", "0").lower() not in ("1", "true", "yes"):
        return None
    if CrossEncoder is None:
        return None
    if _MODEL is not None:
        return _MODEL
    model_name = os.getenv("RAG_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        _MODEL = CrossEncoder(model_name)  # lazy load
    except Exception:
        _MODEL = None
    return _MODEL

@trace_func
def rerank(query: str, docs: List[Document], top_n: int = 8, max_pairs: int = 64) -> List[Document]:
    model = get_ce_reranker()
    if model is None or not docs:
        return docs[:top_n]
    # Build pairs; clip content to keep inference fast
    pairs = []
    for d in docs[:max_pairs]:
        text = (d.page_content or "")[:1200]
        pairs.append((query, text))
    try:
        scores = model.predict(pairs)  # type: ignore[attr-defined]
        # Attach scores and sort descending
        scored = list(zip(scores, docs[:len(scores)]))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return [d for _, d in scored[:top_n]]
    except Exception:
        return docs[:top_n]
