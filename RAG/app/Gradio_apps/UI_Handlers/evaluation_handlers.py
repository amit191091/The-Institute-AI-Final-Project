#!/usr/bin/env python3
"""
Evaluation Handlers
==================

Evaluation and metrics logic functions.
"""

import re
import difflib
from typing import List, Dict, Any, Optional
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval, pretty_metrics
from RAG.app.Gradio_apps.ui_data_loader import _norm_q


def _tok(s):
    """Helper to tokenize text for comparison."""
    return re.findall(r"\w+", s.lower()) if s else []


def evaluate_with_ground_truth(q, ans_raw, ground_truth, top_docs):
    """Evaluate answer with provided ground truth."""
    try:
        context_texts = [d.page_content for d in top_docs]
        dataset = {
            "question": [q],
            "answer": [ans_raw],
            "ground_truth": [ground_truth.strip()],
            "contexts": [context_texts]  # List of context strings, not list of lists
        }
        m = run_eval(dataset)
        return pretty_metrics(m)
    except Exception as e:
        return f"(evaluation with provided ground truth failed: {e})"


def evaluate_with_enhanced_system(q, ans_raw, top_docs, llm):
    """Evaluate using enhanced evaluation system."""
    try:
        # Enhanced evaluation imports
        from RAG.app.auto_evaluator import AutoEvaluator
        from RAG.app.enhanced_question_analyzer import EnhancedQuestionAnalyzer
        
        # Use enhanced evaluation system
        auto_evaluator = AutoEvaluator(llm)
        question_analyzer = EnhancedQuestionAnalyzer()
        
        # Analyze question type
        question_type = question_analyzer.analyze_question(q)
        
        # Generate synthetic ground truth
        synthetic_gt = auto_evaluator.generate_synthetic_ground_truth(q, top_docs)
        
        # Evaluate with synthetic ground truth
        eval_result = auto_evaluator.evaluate_answer(q, top_docs, synthetic_gt)
        
        metrics_txt = f"Enhanced Evaluation (Question Type: {question_type})\n"
        metrics_txt += f"Synthetic Ground Truth: {synthetic_gt}\n\n"
        metrics_txt += f"Evaluation Metrics:\n"
        for metric, score in eval_result.items():
            metrics_txt += f"{metric}: {score:.3f}\n"
        
        # Calculate overall score
        overall_score = sum(eval_result.values()) / len(eval_result)
        metrics_txt += f"\nOverall Score: {overall_score:.3f}"
        return metrics_txt
    except Exception as e:
        return f"(enhanced evaluation failed: {e})"


def evaluate_with_fallback(q, ans_raw, top_docs, gt_map, qa_map):
    """Evaluate using fallback GT/QA maps."""
    try:
        gts = []
        nq = _norm_q(q)
        # Exact or fuzzy GT lookup
        if gt_map.get("__loaded__") and nq in gt_map.get("norm", {}):
            gts = gt_map["norm"][nq]
        elif gt_map.get("__loaded__") and gt_map.get("norm"):
            keys = list(gt_map["norm"].keys())
            best = None; best_s = 0.0
            for k in keys:
                s = difflib.SequenceMatcher(None, nq, k).ratio()
                if s > best_s:
                    best_s = s; best = k
            if best is not None and best_s >= 0.75:
                gts = gt_map["norm"][best]
        
        # QA fallback
        ref = None
        if not gts and qa_map.get("__loaded__"):
            if nq in qa_map.get("norm", {}):
                ref = qa_map["norm"][nq]
            else:
                keys = list(qa_map["norm"].keys())
                best = None; best_s = 0.0
                for k in keys:
                    s = difflib.SequenceMatcher(None, nq, k).ratio()
                    if s > best_s:
                        best_s = s; best = k
                if best is not None and best_s >= 0.75:
                    ref = qa_map["norm"][best]
        
        if not ref:
            ref = (gts[0] if isinstance(gts, list) and gts else ans_raw or "")
        
        # Build dataset + metrics
        context_texts = [d.page_content for d in top_docs]
        dataset = {
            "question": [q],
            "answer": [ans_raw],
            "contexts": [context_texts],  # List of context strings, not list of lists
            "ground_truths": [gts],
            "reference": [ref],
        }

        m = run_eval(dataset)
        metrics_txt = pretty_metrics(m)
        
        # Compare tokens between answer and reference for quick diagnosis
        ref_t = set(_tok(ref))
        ans_t = set(_tok(ans_raw))
        missing = sorted(list(ref_t - ans_t))[:20]
        extra = sorted(list(ans_t - ref_t))[:20]
        
        # Calculate simple overlap metrics
        overlap = len(ref_t & ans_t)
        total_ref = len(ref_t)
        total_ans = len(ans_t)
        precision = overlap / total_ans if total_ans > 0 else 0
        recall = overlap / total_ref if total_ref > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        compare_dict = {
            "reference_excerpt": ref[:400],
            "answer_excerpt": (ans_raw or "")[:400],
            "missing_ref_tokens_in_answer": missing,
            "extra_answer_tokens_not_in_reference": extra,
        }
        
        # Add heuristic hint when LLM metrics are NaN
        vals = [str(m.get(k)) for k in ("faithfulness","answer_relevancy","context_precision","context_recall")]
        if all(v == 'nan' for v in vals):
            metrics_txt += "\n(note: metrics require OPENAI_API_KEY or GOOGLE_API_KEY for RAGAS)"
        
        return metrics_txt, compare_dict
    except Exception as e:
        return f"(fallback evaluation failed: {e})", {}


def run_evaluation(q, ans_raw, top_docs, ground_truth, gt_map, qa_map, llm):
    """Main evaluation orchestrator."""
    metrics_txt = ""
    compare_dict = {}
    
    # If ground truth is provided, use it directly
    if ground_truth and ground_truth.strip():
        metrics_txt = evaluate_with_ground_truth(q, ans_raw, ground_truth, top_docs)
    else:
        # Enhanced evaluation for questions without ground truth
        try:
            # Check if enhanced evaluation is available
            from RAG.app.auto_evaluator import AutoEvaluator
            ENHANCED_EVAL_AVAILABLE = True
        except Exception:
            ENHANCED_EVAL_AVAILABLE = False
        
        if ENHANCED_EVAL_AVAILABLE:
            metrics_txt = evaluate_with_enhanced_system(q, ans_raw, top_docs, llm)
        else:
            metrics_txt, compare_dict = evaluate_with_fallback(q, ans_raw, top_docs, gt_map, qa_map)
    
    return metrics_txt, compare_dict
