#!/usr/bin/env python3
"""
Advanced Evaluation Utilities
============================

Advanced evaluation utilities from eval_ragas.py script.
"""

import re
import math
from typing import List, Tuple, Dict, Any, Optional
from RAG.app.logger import trace_func

# Optional sklearn for overlap F1
try:
    from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore
except Exception:  # pragma: no cover
    precision_score = recall_score = f1_score = None  # type: ignore

# Import factual metrics
from .eval_factual import compute_factual_metrics


@trace_func
def _simple_tokens(text: str) -> List[str]:
    """Extract simple tokens from text for overlap calculation."""
    t = (text or "").lower()
    # keep alphanumerics, collapse whitespace
    t = re.sub(r"[^a-z0-9]+", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return toks


@trace_func
def overlap_prf1(reference: str, contexts: List[str]) -> Tuple[float, float, float]:
    """Compute a simple token-overlap precision/recall/F1 between reference and concatenated contexts.
    If sklearn is available, use f1_score/precision_score/recall_score over token presence vectors.
    Otherwise fall back to set-overlap math.
    """
    ref_tokens = set(_simple_tokens(reference or ""))
    ctx_tokens = set(_simple_tokens("\n".join(contexts or [])))
    if not ref_tokens and not ctx_tokens:
        return float("nan"), float("nan"), float("nan")
    # Guard all three to satisfy type checker (each may be None if sklearn missing)
    if precision_score is not None and recall_score is not None and f1_score is not None:
        vocab = sorted(ref_tokens.union(ctx_tokens))
        y_true = [1 if v in ref_tokens else 0 for v in vocab]
        y_pred = [1 if v in ctx_tokens else 0 for v in vocab]
        # handle edge cases when all zeros
        try:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            # fallback to set math
            inter = len(ref_tokens & ctx_tokens)
            p = inter / max(1, len(ctx_tokens))
            r = inter / max(1, len(ref_tokens))
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return float(p), float(r), float(f1)
    # Set-based fallback
    inter = len(ref_tokens & ctx_tokens)
    p = inter / max(1, len(ctx_tokens))
    r = inter / max(1, len(ref_tokens))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)


@trace_func
def _mean_safe(vals):
    """Safely compute mean of values, handling NaN and invalid values."""
    vals2 = []
    for v in vals:
        try:
            fv = float(v)
            if not (isinstance(fv, float) and (fv != fv)):  # not NaN
                vals2.append(fv)
        except Exception:
            pass
    return float(sum(vals2)/len(vals2)) if vals2 else float("nan")


@trace_func
def _maybe_float(x):
    """Safely convert value to float, returning None if conversion fails."""
    try:
        return float(x)
    except Exception:
        return None


@trace_func
def _pick(series_like, names):
    """Pick value from series-like object by trying multiple possible names."""
    # pandas Series or dict-like getter
    for name in names:
        try:
            v = series_like.get(name)
            if v is not None:
                return v
        except Exception:
            pass
    # try fuzzy by lower contains
    try:
        keys = list(getattr(series_like, 'index', [])) or list(getattr(series_like, 'keys')())
    except Exception:
        keys = []
    lower = {str(k).lower(): k for k in keys}
    for target in names:
        t = str(target).lower()
        for lk, orig in lower.items():
            if t in lk:
                try:
                    return series_like.get(orig)
                except Exception:
                    continue
    return None


@trace_func
def _is_table_like_question(q: str | None) -> bool:
    """Detect if question is asking for table-like data."""
    if not q:
        return False
    ql = str(q).lower()
    pats = [
        r"wear\s*depth",
        r"\bwhich\s+wear\s+case\b",
        r"\btransmission\s+ratio\b|\bgear\s+ratio\b|\bz\s*/\s*z\b",
        r"\bmodule\b",
        r"sampling\s*rate|k\s*s\s*/\s*s|khz|hz",
        r"\bsensitivity\b|m\s*v\s*/\s*g",
        r"\btachometer\b|\baccelerometer\b",
        r"\blubricant\b|viscosity",
    ]
    return any(re.search(p, ql) for p in pats)


@trace_func
def _table_correct(rec: dict) -> bool:
    """Heuristic correctness for table-like Qs using factual metrics."""
    if not _is_table_like_question(rec.get("question")):
        return False
        
    # Strong signals for correctness
    if bool(rec.get("factual_em")):
        return True
    if (rec.get("factual_numeric") or 0.0) >= 0.99:
        return True
    if (rec.get("factual_list_f1") or 0.0) >= 0.99:
        return True
    # Token F1 high threshold
    if (rec.get("factual_token_f1") or 0.0) >= 0.9:
        return True
    return False


@trace_func
def append_eval_footer(per_question_path: str, summary: dict) -> None:
    """Append a summary footer line to the per-question JSONL for quick averages view."""
    try:
        import json as _json
        footer = {
            "__summary__": True,
            "faithfulness": summary.get("faithfulness"),
            "answer_relevancy": summary.get("answer_relevancy"),
            "context_precision": summary.get("context_precision"),
            "context_recall": summary.get("context_recall"),
            "factual_em_rate": summary.get("factual_em_rate"),
            "factual_token_f1": summary.get("factual_token_f1"),
            "factual_numeric": summary.get("factual_numeric"),
            "factual_range": summary.get("factual_range"),
            "factual_list_f1": summary.get("factual_list_f1"),
            "factual_score": summary.get("factual_score"),
        }
        with open(per_question_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(footer, ensure_ascii=False) + "\n")
    except Exception:
        pass
