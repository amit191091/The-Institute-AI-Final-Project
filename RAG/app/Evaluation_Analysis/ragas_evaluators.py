#!/usr/bin/env python3
"""
RAGAS Evaluators
================

RAGAS evaluation functions with advanced features from eval_ragas.py.
"""

import os
import math
from typing import Dict, Any, List, Tuple

# RAGAS imports with robust fallbacks
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    try:
        from ragas.run_config import RunConfig  # optional in some versions
    except Exception:  # pragma: no cover
        RunConfig = None  # type: ignore
except Exception:  # pragma: no cover
    evaluate = None  # type: ignore
    faithfulness = answer_relevancy = context_precision = context_recall = None  # type: ignore
    RunConfig = None  # type: ignore

try:
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore

from RAG.app.logger import trace_func
from .ragas_setup import _setup_ragas_llm, _setup_ragas_embeddings
from .advanced_evaluation_utils import (
    overlap_prf1, _mean_safe, _maybe_float, _pick, 
    _is_table_like_question, _table_correct, compute_factual_metrics
)
from .table_qa_evaluators import calculate_table_qa_accuracy


@trace_func
def run_eval(dataset):
    """Run RAGAS evaluation on dataset with advanced result parsing."""
    if evaluate is None:
        raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")

    print("Using RAGAS with configured LLM and embeddings")

    # Ensure proper Dataset object to avoid API differences
    ds = dataset
    try:
        if Dataset is not None and not hasattr(dataset, "select"):
            ds = Dataset.from_dict(dataset)  # type: ignore[arg-type]
    except Exception:
        ds = dataset

    # Choose metrics based on dataset columns/presence to avoid NaNs
    has_ref = False
    has_gt = False
    try:
        refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
        has_ref = any(bool(r) for r in (refs or []))
    except Exception:
        has_ref = False
    try:
        gts = dataset.get("ground_truths") if isinstance(dataset, dict) else dataset["ground_truths"]
        has_gt = any(isinstance(x, list) and len(x) > 0 for x in (gts or []))
    except Exception:
        has_gt = False

    metrics: list = [m for m in (faithfulness, answer_relevancy) if m is not None]
    if has_ref and context_precision is not None:
        metrics.append(context_precision)
    if has_gt and context_recall is not None:
        metrics.append(context_recall)
    if not metrics:
        raise RuntimeError("No RAGAS metrics available")

    llm = _setup_ragas_llm()
    emb = _setup_ragas_embeddings()
    run_config = None
    try:
        if RunConfig is not None:
            run_config = RunConfig()  # type: ignore[call-arg]
    except Exception:
        run_config = None
    try:
        if run_config is not None:
            result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
        else:
            result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
    except TypeError:
        try:
            if run_config is not None:
                result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
            else:
                result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
        except Exception as e:
            raise RuntimeError(f"RAGAS evaluation failed: {e}") from e

    # Preferred: use to_pandas per-question and average
    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()  # type: ignore
            faith = _mean_safe(df.get("faithfulness", []))
            relev = _mean_safe(df.get("answer_relevancy", []))
            cprec = _mean_safe(df.get("context_precision", []))
            crec = _mean_safe(df.get("context_recall", []))
            out = {
                "faithfulness": faith,
                "answer_relevancy": relev,
                "context_precision": cprec,
                "context_recall": crec,
            }
            # Add heuristic overlap metrics if reference/contexts present
            try:
                refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
                ctxs = dataset.get("contexts") if isinstance(dataset, dict) else dataset["contexts"]
                if refs and ctxs:
                    p_list, r_list, f1_list = [], [], []
                    for ref, ctx in zip(refs, ctxs):
                        p, r, f1v = overlap_prf1(ref or "", list(ctx or []))
                        p_list.append(p); r_list.append(r); f1_list.append(f1v)
                    out["overlap_precision"] = _mean_safe(p_list)
                    out["overlap_recall"] = _mean_safe(r_list)
                    out["overlap_f1"] = _mean_safe(f1_list)
            except Exception:
                pass
            return out
    except Exception:
        pass

    # Next: try dict-like summary
    try:
        raw = dict(result)  # type: ignore
        as_dict = {str(k): raw[k] for k in raw.keys()}
        faith = float(as_dict.get("faithfulness", float("nan")))
        relev = float(as_dict.get("answer_relevancy", float("nan")))
        cprec = float(as_dict.get("context_precision", float("nan")))
        crec = float(as_dict.get("context_recall", float("nan")))
        out = {
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": cprec,
            "context_recall": crec,
        }
        # Add heuristic overlap metrics
        try:
            refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
            ctxs = dataset.get("contexts") if isinstance(dataset, dict) else dataset["contexts"]
            if refs and ctxs:
                p_list, r_list, f1_list = [], [], []
                for ref, ctx in zip(refs, ctxs):
                    p, r, f1v = overlap_prf1(ref or "", list(ctx or []))
                    p_list.append(p); r_list.append(r); f1_list.append(f1v)
                out["overlap_precision"] = _mean_safe(p_list)
                out["overlap_recall"] = _mean_safe(r_list)
                out["overlap_f1"] = _mean_safe(f1_list)
        except Exception:
            pass
        return out
    except Exception:
        pass

    # Finally: handle .results list shape
    try:
        items = list(getattr(result, "results") or [])
        faith = _mean_safe([r.get("faithfulness") for r in items])
        relev = _mean_safe([r.get("answer_relevancy") for r in items])
        cprec = _mean_safe([r.get("context_precision") for r in items])
        crec = _mean_safe([r.get("context_recall") for r in items])
        out = {
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": cprec,
            "context_recall": crec,
        }
        # Add heuristic overlap metrics
        try:
            refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
            ctxs = dataset.get("contexts") if isinstance(dataset, dict) else dataset["contexts"]
            if refs and ctxs:
                p_list, r_list, f1_list = [], [], []
                for ref, ctx in zip(refs, ctxs):
                    p, r, f1v = overlap_prf1(ref or "", list(ctx or []))
                    p_list.append(p); r_list.append(r); f1_list.append(f1v)
                out["overlap_precision"] = _mean_safe(p_list)
                out["overlap_recall"] = _mean_safe(r_list)
                out["overlap_f1"] = _mean_safe(f1_list)
        except Exception:
            pass
        return out
    except Exception:
        return {
            "faithfulness": float("nan"),
            "answer_relevancy": float("nan"),
            "context_precision": float("nan"),
            "context_recall": float("nan"),
            "overlap_precision": float("nan"),
            "overlap_recall": float("nan"),
            "overlap_f1": float("nan"),
        }


@trace_func
def run_eval_detailed(dataset):
    """Run RAGAS and return (summary_metrics, per_question) where per_question is a
    list of dicts including metrics per item when available. Falls back to empty list otherwise.
    """
    if evaluate is None:
        raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")
    # Ensure Dataset object
    ds = dataset
    try:
        if Dataset is not None and not hasattr(dataset, "select"):
            ds = Dataset.from_dict(dataset)  # type: ignore[arg-type]
    except Exception:
        ds = dataset
    # Choose metrics dynamically based on columns/presence
    has_ref = False
    has_gt = False
    try:
        refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
        has_ref = any(bool(r) for r in (refs or []))
    except Exception:
        has_ref = False
    try:
        gts = dataset.get("ground_truths") if isinstance(dataset, dict) else dataset["ground_truths"]
        has_gt = any(isinstance(x, list) and len(x) > 0 for x in (gts or []))
    except Exception:
        has_gt = False
    metrics: list = [m for m in (faithfulness, answer_relevancy) if m is not None]
    if has_ref and context_precision is not None:
        metrics.append(context_precision)
    if has_gt and context_recall is not None:
        metrics.append(context_recall)
    if not metrics:
        raise RuntimeError("No RAGAS metrics available")
    # Evaluate
    llm = _setup_ragas_llm()
    emb = _setup_ragas_embeddings()
    run_config = None
    try:
        if RunConfig is not None:
            run_config = RunConfig()  # type: ignore[call-arg]
    except Exception:
        run_config = None
    try:
        if run_config is not None:
            result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
        else:
            result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
    except TypeError:
        # older versions may not accept named args
        try:
            if run_config is not None:
                result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
            else:
                result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
        except Exception:
            result = evaluate(ds, metrics=metrics)  # type: ignore
    # Per-question extraction aligned with original inputs
    per_q = []

    try:
        # Build a view of original inputs for alignment
        def _get_col(key):
            if isinstance(ds, dict):
                return ds.get(key, [])
            try:
                return ds[key]  # datasets style
            except Exception:
                return []
        q_list = list(_get_col("question"))
        a_list = list(_get_col("answer"))
        r_list = list(_get_col("reference"))
        tr_raw = _get_col("reasoning_trace")
        tr_list = list(tr_raw) if tr_raw else []
        n = max(len(q_list), len(a_list), len(r_list))

        if hasattr(result, "to_pandas"):
            df = result.to_pandas()  # type: ignore
            m = min(n, getattr(df, "shape", [0])[0])
            for i in range(m):
                row = df.iloc[i]
                rec = {
                    "question": q_list[i] if i < len(q_list) else None,
                    "answer": a_list[i] if i < len(a_list) else None,
                    "reference": r_list[i] if i < len(r_list) else None,
                    "reasoning_trace": (tr_list[i] if i < len(tr_list) else None),
                    "faithfulness": _maybe_float(_pick(row, ["faithfulness", "faithfulness_score", "faith"])),
                    "answer_relevancy": _maybe_float(_pick(row, ["answer_relevancy", "relevancy", "answer_rel"])) ,
                    "context_precision": _maybe_float(_pick(row, ["context_precision", "ctx_precision", "precision"])),
                    "context_recall": _maybe_float(_pick(row, ["context_recall", "ctx_recall", "recall"])),
                }
                # Factual metrics against reference
                try:
                    fm = compute_factual_metrics(rec.get("answer") or "", rec.get("reference") or "")
                    rec.update(fm)
                except Exception:
                    pass
                # Add overlap metrics per-question
                try:
                    ref = rec.get("reference") or ""
                    # Use original contexts for this index
                    ctxs = _get_col("contexts")[i] if i < len(_get_col("contexts")) else []
                    p, r, f1v = overlap_prf1(ref, list(ctxs or []))
                    rec["overlap_precision"], rec["overlap_recall"], rec["overlap_f1"] = p, r, f1v
                except Exception:
                    pass
                # mark table QA correctness
                rec["table_like"] = _is_table_like_question(rec.get("question"))
                rec["table_correct"] = _table_correct(rec)
                per_q.append(rec)
        elif hasattr(result, "results"):
            items = list(getattr(result, "results") or [])
            m = min(n, len(items))
            for i in range(m):
                item = items[i]
                rec = {
                    "question": q_list[i] if i < len(q_list) else None,
                    "answer": a_list[i] if i < len(a_list) else None,
                    "reference": r_list[i] if i < len(r_list) else None,
                    "reasoning_trace": (tr_list[i] if i < len(tr_list) else None),
                    "faithfulness": _maybe_float(_pick(item, ["faithfulness", "faithfulness_score", "faith"])),
                    "answer_relevancy": _maybe_float(_pick(item, ["answer_relevancy", "relevancy", "answer_rel"])) ,
                    "context_precision": _maybe_float(_pick(item, ["context_precision", "ctx_precision", "precision"])),
                    "context_recall": _maybe_float(_pick(item, ["context_recall", "ctx_recall", "recall"])),
                }
                try:
                    fm = compute_factual_metrics(rec.get("answer") or "", rec.get("reference") or "")
                    rec.update(fm)
                except Exception:
                    pass
                try:
                    ref = rec.get("reference") or ""
                    ctxs = _get_col("contexts")[i] if i < len(_get_col("contexts")) else []
                    p, r, f1v = overlap_prf1(ref, list(ctxs or []))
                    rec["overlap_precision"], rec["overlap_recall"], rec["overlap_f1"] = p, r, f1v
                except Exception:
                    pass
                # mark table QA correctness
                rec["table_like"] = _is_table_like_question(rec.get("question"))
                rec["table_correct"] = _table_correct(rec)
                per_q.append(rec)
        # If results shorter than dataset, pad remaining with None metrics
        for i in range(len(per_q), n):
            per_q.append({
                "question": q_list[i] if i < len(q_list) else None,
                "answer": a_list[i] if i < len(a_list) else None,
                "reference": r_list[i] if i < len(r_list) else None,
                "reasoning_trace": (tr_list[i] if i < len(tr_list) else None),
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "table_like": _is_table_like_question(q_list[i] if i < len(q_list) else None),
                "table_correct": None,
            })
    except Exception:
        per_q = []

    # Compute summary from per-question metrics as a robust fallback
    @trace_func
    def _mean_safe(values):
        vals = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "faithfulness": _mean_safe([r.get("faithfulness") for r in per_q]),
        "answer_relevancy": _mean_safe([r.get("answer_relevancy") for r in per_q]),
        "context_precision": _mean_safe([r.get("context_precision") for r in per_q]),
        "context_recall": _mean_safe([r.get("context_recall") for r in per_q]),
    }
    # Factual summary
    try:
        summary["factual_em_rate"] = _mean_safe([1.0 if r.get("factual_em") else 0.0 for r in per_q])
        summary["factual_token_f1"] = _mean_safe([r.get("factual_token_f1") for r in per_q])
        summary["factual_numeric"] = _mean_safe([r.get("factual_numeric") for r in per_q])
        summary["factual_range"] = _mean_safe([r.get("factual_range") for r in per_q])
        summary["factual_list_f1"] = _mean_safe([r.get("factual_list_f1") for r in per_q])
        summary["factual_score"] = _mean_safe([r.get("factual_score") for r in per_q])
    except Exception:
        pass
    # Compute overlap metrics summary
    try:
        summary["overlap_precision"] = _mean_safe([r.get("overlap_precision") for r in per_q])
        summary["overlap_recall"] = _mean_safe([r.get("overlap_recall") for r in per_q])
        summary["overlap_f1"] = _mean_safe([r.get("overlap_f1") for r in per_q])
    except Exception:
        pass

    # Table QA accuracy
    try:
        table_items = [r for r in per_q if r.get("table_like")]
        if table_items:
            correct = [1 for r in table_items if r.get("table_correct")]
            summary["table_qa_accuracy"] = float(sum(correct) / len(table_items))
        else:
            summary["table_qa_accuracy"] = None
    except Exception:
        pass

    return summary, per_q


@trace_func
def pretty_metrics(m: dict) -> str:
    """Format metrics for pretty printing with advanced metrics support."""
    def _fmt(x):
        try:
            import math as _m
            if x is None:
                return "n/a"
            if isinstance(x, float) and (_m.isnan(x) or _m.isinf(x)):
                return "n/a"
            if isinstance(x, (int, float)):
                return f"{x:.3f}"
            return str(x)
        except Exception:
            return str(x)
    lines = [
        f"Faithfulness: {_fmt(m.get('faithfulness'))}",
        f"Answer relevancy: {_fmt(m.get('answer_relevancy'))}",
        f"Context precision: {_fmt(m.get('context_precision'))}",
        f"Context recall: {_fmt(m.get('context_recall'))}",
    ]
    if "table_qa_accuracy" in m:
        lines.append(f"Table QA accuracy: {_fmt(m.get('table_qa_accuracy'))}")
    # Factual metrics (context-agnostic)
    if any(k in m for k in ("factual_score", "factual_em_rate", "factual_token_f1", "factual_numeric", "factual_range", "factual_list_f1")):
        lines.append(f"Factual score: {_fmt(m.get('factual_score'))}")
        lines.append(f"Factual EM rate: {_fmt(m.get('factual_em_rate'))}")
        lines.append(f"Factual token F1: {_fmt(m.get('factual_token_f1'))}")
        lines.append(f"Factual numeric: {_fmt(m.get('factual_numeric'))}")
        lines.append(f"Factual range: {_fmt(m.get('factual_range'))}")
        lines.append(f"Factual list F1: {_fmt(m.get('factual_list_f1'))}")
    # Optional heuristic overlap metrics (no LLM/embeddings required)
    if any(k in m for k in ("overlap_precision", "overlap_recall", "overlap_f1")):
        lines.append(f"Overlap precision (heuristic): {_fmt(m.get('overlap_precision'))}")
        lines.append(f"Overlap recall (heuristic): {_fmt(m.get('overlap_recall'))}")
        lines.append(f"Overlap F1 (heuristic): {_fmt(m.get('overlap_f1'))}")
    return "\n".join(lines)
