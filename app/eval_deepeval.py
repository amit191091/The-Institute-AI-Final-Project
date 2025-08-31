from __future__ import annotations

"""Optional DeepEval integration.

Guarded by env flags and availability of CONFIDENT_API_KEY. If enabled, runs
side-by-side with RAGAS and persists a compact summary and per-question details.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from app.logger import trace_func, get_logger


def _has_confident_key() -> bool:
    key = os.getenv("CONFIDENT_API_KEY") or os.getenv("DEEPEVAL_API_KEY")
    return bool(key and key.strip())


@trace_func
def run_eval(dataset: Dict[str, List[Any]]):
    """Run DeepEval if available and enabled.

    Expected dataset keys: question, answer, contexts, ground_truths (optional), reference (optional)
    Writes logs/deepeval_summary.json and logs/deepeval_per_question.jsonl
    """
    if os.getenv("RAG_DEEPEVAL", "0").lower() not in ("1", "true", "yes"):
        return None, []
    if not _has_confident_key():
        get_logger().info("DeepEval skipped: missing CONFIDENT_API_KEY/DEEPEVAL_API_KEY")
        return None, []

    try:
        # Import lazily to avoid hard dependency
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except Exception as e:  # pragma: no cover
        get_logger().warning(f"DeepEval not available: {e}")
        return None, []

    # Respect OpenAI disable switch unless the user explicitly provides a non-OpenAI DEEPEVAL_MODEL
    openai_allowed = os.getenv("RAG_USE_OPENAI", "0").lower() in ("1", "true", "yes")
    model_name = os.getenv("DEEPEVAL_MODEL")
    if not model_name:
        # If no explicit model, only fall back to OpenAI if allowed and key exists
        if openai_allowed and os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        else:
            get_logger().info("DeepEval skipped: no DEEPEVAL_MODEL and OpenAI disabled or missing OPENAI_API_KEY")
            return None, []

    # Build per-question evaluation items
    questions: List[str] = list(dataset.get("question", []))
    answers: List[str] = list(dataset.get("answer", []))
    contexts: List[List[str]] = list(dataset.get("contexts", []))
    gts: List[List[str]] = list(dataset.get("ground_truths", []))

    items = []
    n = max(len(questions), len(answers), len(contexts))
    for i in range(n):
        q = questions[i] if i < len(questions) else ""
        a = answers[i] if i < len(answers) else ""
        ctx = contexts[i] if i < len(contexts) else []
        gt = gts[i] if i < len(gts) else []
        items.append({
            "question": q,
            "answer": a,
            "contexts": ctx,
            "ground_truths": gt,
        })

    # Configure metrics (keep minimal and fast by default)
    # Compute per-question metrics
    per_q = []
    for it in items:
        try:
            tc = LLMTestCase(
                input=it["question"],
                actual_output=it["answer"],
                expected_output=(it["ground_truths"][0] if (it.get("ground_truths") and len(it["ground_truths"])>0) else None),
                retrieval_context=list(it.get("contexts") or []),
            )
            m_faith = FaithfulnessMetric(model=model_name)
            m_rel = AnswerRelevancyMetric(model=model_name)
            # Measure
            m_faith.measure(tc)
            m_rel.measure(tc)
            row = {
                "question": it.get("question"),
                "answer": it.get("answer"),
                "faithfulness": float(getattr(m_faith, "score", None) or 0.0),
                "answer_relevancy": float(getattr(m_rel, "score", None) or 0.0),
            }
            per_q.append(row)
        except Exception as e:  # pragma: no cover
            get_logger().warning(f"DeepEval test case failed: {e}")
            continue

    # Summarize
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _mean(vals):
        nums = [v for v in vals if isinstance(v, (int, float))]
        return (sum(nums) / len(nums)) if nums else None

    summary = {
        "faithfulness": _mean([r.get("faithfulness") for r in per_q]),
        "answer_relevancy": _mean([r.get("answer_relevancy") for r in per_q]),
    }

    # Persist
    out_dir = Path("logs"); out_dir.mkdir(exist_ok=True)
    with open(out_dir / "deepeval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "deepeval_per_question.jsonl", "w", encoding="utf-8") as f:
        for rec in per_q:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        get_logger().info("DeepEval summary: %s", json.dumps(summary))
    except Exception:
        pass

    return summary, per_q
