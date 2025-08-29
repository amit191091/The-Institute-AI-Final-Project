#!/usr/bin/env python3
"""
Agent Handlers
=============

Agent tool integration functions.
"""

import re
import difflib
from typing import List, Dict, Any, Optional
from RAG.app.Agent_Components.agents import answer_needle
from RAG.app.Gradio_apps.ui_data_loader import _norm_q


def run_agent_with_tools(question: str, docs, hybrid, llm=None):
    """Run agent with tools for visibility."""
    steps = []
    try:
        # Agent tool shims
        try:
            from RAG.app.Agent_Components.agent_tools import tool_analyze_query, tool_retrieve_candidates, tool_retrieve_filtered, tool_list_figures
        except Exception:
            tool_analyze_query = None
            tool_retrieve_candidates = None
            tool_retrieve_filtered = None
            tool_list_figures = None
        
        if tool_analyze_query:
            qa = tool_analyze_query(question)
            steps.append({"action": "analyze_query", "observation": qa})
        if tool_retrieve_candidates:
            cands = tool_retrieve_candidates(question, hybrid)
            steps.append({"action": "retrieve_candidates", "observation_count": len(cands)})
        if tool_retrieve_filtered:
            fr = tool_retrieve_filtered(question, docs, hybrid)
            steps.append({"action": "filter+rerank", "observation": {"top_docs": fr.get("top_docs", [])}})
        
        # Actually generate an answer using the LLM
        ans = ""
        if tool_list_figures and re.search(r"\b(list|all|show)\b.*\bfigures\b", question, re.I):
            figs = tool_list_figures(docs)
            steps.append({"action": "list_figures", "observation_count": len(figs)})
            ans = "\n".join([f"Figure {f.get('figure_number')}: {f.get('label')} [{f.get('file')} p{f.get('page')}]" for f in figs])
        else:
            # Generate answer using the LLM with retrieved documents
            if llm is not None:
                ans = answer_needle(llm, docs, question)
            else:
                ans = "LLM not available for answer generation"
        
        # Add evaluation logic
        metrics_txt = ""
        try:
            # Import ground truth maps
            from RAG.app.Gradio_apps.ui_data_loader import load_ground_truth_and_qa
            gt_map, qa_map = load_ground_truth_and_qa()
            
            # Look for ground truth
            gts = []
            nq = _norm_q(question)
            
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
            
            # Run evaluation if we have ground truth
            if gts:
                from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval, pretty_metrics
                # Fix dataset structure for proper RAGAS evaluation
                context_texts = [d.page_content for d in docs]
                dataset = {
                    "question": [question],
                    "answer": [ans],
                    "contexts": [context_texts],  # List of context strings
                    "ground_truths": [gts],
                    "reference": [gts[0] if isinstance(gts, list) and gts else str(gts)],
                }
                m = run_eval(dataset)
                metrics_txt = pretty_metrics(m)
            else:
                pass
                
        except Exception as e:
            metrics_txt = f"(evaluation failed: {e})"
        
        # Return both the trace and the answer with evaluation
        if gts:
            ground_truth = gts[0] if isinstance(gts, list) and gts else str(gts)
            result = f"""## 🤖 Agent Answer
{ans}

## ✅ Ground Truth Answer
{ground_truth}

## 📊 Evaluation Metrics
{metrics_txt}"""
        else:
            result = f"""## 🤖 Agent Answer
{ans}

## 📊 Evaluation
No ground truth found for comparison."""
        
        return steps, result
    except Exception as e:
        return steps + [{"error": str(e)}], f"(agent failed: {e})"


def do_audit(docs):
    """Perform audit of figures."""
    try:
        from RAG.app.Agent_Components.agent_tools import tool_audit_and_fill_figures as _audit_figs
        if _audit_figs is None:
            return {"error": "tool not available"}
        return _audit_figs(docs)
    except Exception:
        return {"error": "tool not available"}


def do_plan(observations: str, llm):
    """Generate a plan based on observations."""
    try:
        from RAG.app.Agent_Components.agent_tools import tool_plan as _plan
        if _plan is None:
            return "(planner not available)"
        return _plan(observations, llm)
    except Exception:
        return "(planner not available)"
