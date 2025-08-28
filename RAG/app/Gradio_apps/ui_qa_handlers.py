"""
Q&A event handlers for the Gradio interface.
"""
import json
import re
import difflib
from pathlib import Path
from typing import List, Tuple
import gradio as gr
from RAG.app.Agent_Components.agents import answer_needle, answer_summary, answer_table, route_question_ex
from RAG.app.retrieve import apply_filters, query_analyzer, rerank_candidates
from RAG.app.logger import get_logger
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval, pretty_metrics

# Enhanced evaluation imports
try:
    from RAG.app.auto_evaluator import AutoEvaluator
    from RAG.app.enhanced_question_analyzer import EnhancedQuestionAnalyzer
    from RAG.app.evaluation_wrapper import (
        EVAL_AVAILABLE, test_google_api, test_ragas, generate_ground_truth, evaluate_rag
    )
    ENHANCED_EVAL_AVAILABLE = True
except Exception:
    ENHANCED_EVAL_AVAILABLE = False

# Agent tool shims
try:
    from RAG.app.Agent_Components.agent_tools import tool_analyze_query, tool_retrieve_candidates, tool_retrieve_filtered, tool_list_figures
except Exception:
    tool_analyze_query = None  # type: ignore
    tool_retrieve_candidates = None  # type: ignore
    tool_retrieve_filtered = None  # type: ignore
    tool_list_figures = None  # type: ignore

from RAG.app.Gradio_apps.ui_components import (
    _rows_to_df, _extract_table_figure_context, _fmt_docs, 
    _render_router_info, _rows_from_docs, _fig_sort_key
)
from RAG.app.Gradio_apps.ui_data_loader import _norm_q

def on_ask(q, ground_truth="", dbg=False, docs=None, hybrid=None, llm=None, debug=None, gt_map=None, qa_map=None):
	# Step 1: Process query and get candidates
	from RAG.app.Gradio_apps.UI_Handlers.query_processors import process_query_and_candidates
	qa, cands, filtered, top_docs, dense_docs, sparse_docs = process_query_and_candidates(q, docs, hybrid, debug)
	r, rtrace = route_question_ex(q)
	# Step 2: Check for special figure list request
	from RAG.app.Gradio_apps.UI_Handlers.query_processors import process_figure_list_request
	is_figure_list, ans_text, router_info, trace, all_figs = process_figure_list_request(q, qa, docs, r, rtrace)
	if is_figure_list:
		# Build debug outputs for figure list
		from RAG.app.Gradio_apps.UI_Handlers.debug_handlers import build_debug_outputs, create_debug_updates
		_dbg_visible, _dbg_router_md, _dbg_filters_json, _dbg_dense_md, _dbg_sparse_md, _dbg_hybrid_md, _dbg_top_df = build_debug_outputs(qa, r, rtrace, dense_docs, sparse_docs, cands, all_figs, dbg)
		_acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, fig_update = create_debug_updates(_dbg_visible, _dbg_router_md, _dbg_filters_json, _dbg_dense_md, _dbg_sparse_md, _dbg_hybrid_md, _dbg_top_df, {}, None)
		return (
			f"{router_info}\n\n{ans_text}\n\n(trace: {trace})",
			"",
			_acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, fig_update
		)
	router_info = f"{_render_router_info(r, top_docs)} | Agent: {r} | Rules: {', '.join(rtrace.get('matched', []))}"
	trace = f"Keywords: {qa.get('keywords', [])} | Filters: {qa.get('filters', {})}"
	# Build structured debug outputs
	_dbg_visible = bool(dbg)
	_dbg_router_md = f"**Route:** {r}  \n**Rules:** {', '.join(rtrace.get('matched', []))}  \n**Canonical:** {qa.get('canonical', '')}"
	_dbg_filters_json = {"filters": qa.get("filters", {}), "keywords": qa.get("keywords", []), "canonical": qa.get("canonical", "")}
	_dbg_dense_md = "Dense (top10):\n\n" + _fmt_docs(dense_docs)
	_dbg_sparse_md = "Sparse (top10):\n\n" + _fmt_docs(sparse_docs)
	_dbg_hybrid_md = "Hybrid candidates (pre-filter):\n\n" + _fmt_docs(cands)
	_dbg_top_df = _rows_to_df(_rows_from_docs(top_docs))
	# Step 3: Generate answer based on route
	from RAG.app.Gradio_apps.UI_Handlers.answer_generators import generate_summary_answer, generate_table_answer, generate_needle_answer
	
	if r == "summary":
		out = generate_summary_answer(llm, top_docs, q, router_info, trace)
		fig_path = None
	elif r == "table":
		out, fig_path = generate_table_answer(llm, top_docs, q, qa, docs, router_info, trace)
	else:
		out = generate_needle_answer(llm, top_docs, q, router_info, trace)
		fig_path = None

	# Step 4: Run evaluation
	from RAG.app.Gradio_apps.UI_Handlers.evaluation_handlers import run_evaluation
	metrics_txt, compare_dict = run_evaluation(q, out, top_docs, ground_truth, gt_map, qa_map, llm)

	# Step 5: Logging and audit
	from RAG.app.Gradio_apps.UI_Handlers.debug_handlers import log_query_and_answer, audit_query_to_file, build_debug_outputs, create_debug_updates
	log_query_and_answer(q, out, metrics_txt)
	audit_query_to_file(q, r, rtrace, out, metrics_txt, top_docs)
	
	# Step 6: Build debug outputs and updates
	_dbg_visible, _dbg_router_md, _dbg_filters_json, _dbg_dense_md, _dbg_sparse_md, _dbg_hybrid_md, _dbg_top_df = build_debug_outputs(qa, r, rtrace, dense_docs, sparse_docs, cands, top_docs, dbg)
	_acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, fig_update = create_debug_updates(_dbg_visible, _dbg_router_md, _dbg_filters_json, _dbg_dense_md, _dbg_sparse_md, _dbg_hybrid_md, _dbg_top_df, compare_dict, fig_path)
	return out, metrics_txt, _acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, fig_update


def _run_agent(question: str, docs, hybrid, llm=None):
    """Run agent with tools for visibility."""
    from RAG.app.Gradio_apps.UI_Handlers.agent_handlers import run_agent_with_tools
    return run_agent_with_tools(question, docs, hybrid, llm)


def _do_audit(docs):
    """Perform audit of figures."""
    from RAG.app.Gradio_apps.UI_Handlers.agent_handlers import do_audit
    return do_audit(docs)


def _do_plan(observations: str, llm):
    """Generate a plan based on observations."""
    from RAG.app.Gradio_apps.UI_Handlers.agent_handlers import do_plan
    return do_plan(observations, llm)
