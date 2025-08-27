"""
Q&A event handlers for the Gradio interface.
"""
import json
import re
import difflib
from pathlib import Path
from typing import List, Tuple
import gradio as gr
from RAG.app.agents import answer_needle, answer_summary, answer_table, route_question_ex
from RAG.app.retrieve import apply_filters, query_analyzer, rerank_candidates
from RAG.app.logger import get_logger
from RAG.app.eval_ragas import run_eval, pretty_metrics

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
    from RAG.app.agent_tools import tool_analyze_query, tool_retrieve_candidates, tool_retrieve_filtered, tool_list_figures
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
	qa = query_analyzer(q)
	cands = hybrid.invoke(q)
	dense_docs = []
	sparse_docs = []
	if debug and debug.get("dense") is not None:
		try:
			dense_docs = debug["dense"].invoke(q)
		except Exception:
			pass
	if debug and debug.get("sparse") is not None:
		try:
			sparse_docs = debug["sparse"].invoke(q)
		except Exception:
			pass
	filtered = apply_filters(cands, qa["filters"])
	# section/number fallback if nothing after filtering
	sec = qa["filters"].get("section") if qa and qa.get("filters") else None
	if sec and not filtered:
		def _fallback_ok(d):
			md = d.metadata or {}
			if (md.get("section") or md.get("section_type")) != sec:
				return False
			# Keep number filters when present
			fv = (qa["filters"].get("figure_number") if qa and qa.get("filters") else None)
			tv = (qa["filters"].get("table_number") if qa and qa.get("filters") else None)
			import re as _re
			if fv is not None:
				fn = md.get("figure_number")
				if str(fn) == str(fv):
					return True
				lab = str(md.get("figure_label") or md.get("caption") or "")
				return bool(_re.match(rf"^\s*figure\s*{int(str(fv))}\b", lab, _re.I))
			if tv is not None:
				tn = md.get("table_number")
				if str(tn) == str(tv):
					return True
				lab = str(md.get("table_label") or "")
				return bool(_re.match(rf"^\s*table\s*{int(str(tv))}\b", lab, _re.I))
			return True
		filtered = [d for d in docs if _fallback_ok(d)]
	top_docs = rerank_candidates(q, filtered, top_n=8)
	# Fallbacks to avoid empty contexts for metrics/answering
	if not top_docs:
		# Use unfiltered candidates
		top_docs = (cands or [])[:8]
	if not top_docs:
		# Last resort use first few indexed docs
		top_docs = docs[:8]
	# If the query is about figures, present them in ascending order by number/order for consistency
	try:
		if (qa.get("filters") or {}).get("section") == "Figure":
			top_docs = sorted(top_docs, key=_fig_sort_key)
	except Exception:
		pass
	r, rtrace = route_question_ex(q)
	# Special-case: list all figures Γאפ return a deterministic list from metadata
	try:
		if (qa.get("filters") or {}).get("section") == "Figure" and re.search(r"\b(list|all|show)\b.*\bfigures\b", q, re.I):
			# Gather all figure docs and sort by number/order/page
			all_figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure"]
			all_figs = sorted(all_figs, key=_fig_sort_key)
			# Build a clean list from normalized labels
			lines = []
			for d in all_figs:
				md = d.metadata or {}
				label = md.get("figure_label") or d.page_content.splitlines()[0]
				lines.append(f"{label} [{md.get('file_name')} p{md.get('page')} Figure]")
			ans_text = "\n".join(lines) if lines else "(no figures found)"
			router_info = f"Route: {r} | Top contexts: [all figures]"
			trace = f"Keywords: {qa['keywords']} | Filters: {qa['filters']}"
			# Compose outputs keep debug panels wired
			_dbg_visible = bool(dbg)
			_dbg_router_md = f"**Route:** {r}  \n**Rules:** {', '.join(rtrace.get('matched', []))}  \n**Canonical:** {qa.get('canonical')}"
			_dbg_filters_json = {"filters": qa.get("filters"), "keywords": qa.get("keywords"), "canonical": qa.get("canonical")}
			_dbg_dense_md = "Dense (topΓיט10):\n\n" + _fmt_docs(dense_docs)
			_dbg_sparse_md = "Sparse (topΓיט10):\n\n" + _fmt_docs(sparse_docs)
			_dbg_hybrid_md = "Hybrid candidates (pre-filter):\n\n" + _fmt_docs(cands)
			_dbg_top_df = _rows_to_df(_rows_from_docs(all_figs))
			_compare_upd = gr.update(value={}, visible=_dbg_visible)
			return (
				f"{router_info}\n\n{ans_text}\n\n(trace: {trace})",
				"",
				gr.update(visible=_dbg_visible, open=False),
				gr.update(value=_dbg_router_md, visible=_dbg_visible),
				gr.update(value=_dbg_filters_json, visible=_dbg_visible),
				gr.update(value=_dbg_dense_md, visible=_dbg_visible),
				gr.update(value=_dbg_sparse_md, visible=_dbg_visible),
				gr.update(value=_dbg_hybrid_md, visible=_dbg_visible),
				gr.update(value=_dbg_top_df, visible=_dbg_visible),
				_compare_upd,
				None,
			)
	except Exception:
		pass
	router_info = _render_router_info(r, top_docs) + f" | Agent: {r} | Rules: {', '.join(rtrace.get('matched', []))}"
	trace = f"Keywords: {qa['keywords']} | Filters: {qa['filters']}"
	# Build structured debug outputs
	_dbg_visible = bool(dbg)
	_dbg_router_md = f"**Route:** {r}  \n**Rules:** {', '.join(rtrace.get('matched', []))}  \n**Canonical:** {qa.get('canonical')}"
	_dbg_filters_json = {"filters": qa.get("filters"), "keywords": qa.get("keywords"), "canonical": qa.get("canonical")}
	_dbg_dense_md = "Dense (topΓיט10):\n\n" + _fmt_docs(dense_docs)
	_dbg_sparse_md = "Sparse (topΓיט10):\n\n" + _fmt_docs(sparse_docs)
	_dbg_hybrid_md = "Hybrid candidates (pre-filter):\n\n" + _fmt_docs(cands)
	_dbg_top_df = _rows_to_df(_rows_from_docs(top_docs))
	if r == "summary":
		ans_raw = answer_summary(llm, top_docs, q)
		out = f"{router_info}\n\n{ans_raw}\n\n(trace: {trace})"
	elif r == "table":
		# If user asked for a specific figure/table number, prioritize matching docs
		try:
			desired_fig = None
			desired_tbl = None
			if qa and qa.get("filters"):
				fv = qa["filters"].get("figure_number")
				if fv is not None:
					try:
						desired_fig = int(str(fv))
					except Exception:
						desired_fig = None
				tv = qa["filters"].get("table_number")
				if tv is not None:
					try:
						desired_tbl = int(str(tv))
					except Exception:
						desired_tbl = None
			if desired_fig is not None:
				import re as _re
				def _is_fig_match(d):
					md = d.metadata or {}
					if (md.get("section") or md.get("section_type")) != "Figure":
						return False
					fn = md.get("figure_number")
					if fn is not None and str(fn).strip().isdigit() and int(str(fn)) == desired_fig:
						return True
					lab = str(md.get("figure_label") or "")
					return bool(_re.match(rf"^\s*figure\s*{desired_fig}\b", lab, _re.I))
				matches = [d for d in top_docs if _is_fig_match(d)]
				if matches:
					top_docs = matches + [d for d in top_docs if d not in matches]
		except Exception:
			pass
		table_ctx = _extract_table_figure_context(top_docs)
		ans_raw = answer_table(llm, top_docs, q)
		# If a relevant figure was retrieved, prefer displaying via an Image component
		fig_path = None
		try:
			# Prefer the first doc that matches desired fig number (by metadata or label), else the first figure doc
			_fig_docs = [d for d in top_docs if (d.metadata or {}).get("section") == "Figure" and (d.metadata or {}).get("image_path")]
			want = None
			if qa and qa.get("filters") and qa["filters"].get("figure_number"):
				try:
					want = int(str(qa["filters"]["figure_number"]).strip())
				except Exception:
					want = None
			def _matches_want(d):
				if want is None:
					return False
				md = d.metadata or {}
				fn = md.get("figure_number")
				if fn is not None and str(fn).strip().isdigit() and int(str(fn)) == want:
					return True
				import re as _re
				lab = str(md.get("figure_label") or md.get("caption") or "")
				return bool(_re.match(rf"^\s*figure\s*{want}\b", lab, _re.I))
			if want is not None and _fig_docs:
				pref = [d for d in _fig_docs if _matches_want(d)]
				fig_doc = pref[0] if pref else (_fig_docs[0] if _fig_docs else None)
			else:
				fig_doc = _fig_docs[0] if _fig_docs else None
			# If still nothing (e.g., not in top docs), try a best-effort lookup across all docs
			if fig_doc is None and want is not None:
				_all_figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure" and (d.metadata or {}).get("image_path")]
				_pref = [d for d in _all_figs if _matches_want(d)]
				fig_doc = _pref[0] if _pref else ( _all_figs[0] if _all_figs else None )
			if fig_doc is not None:
				p = Path(str(fig_doc.metadata.get("image_path")))
				if p.exists():
					fig_path = str(p)
		except Exception:
			pass
		# We keep markdown preview for table context, and the actual image will show in a Gallery component
		out = f"{router_info}\n\nTable/Figure context preview:\n{table_ctx}\n\n---\n\n{ans_raw}\n\n(trace: {trace})"
	else:
		ans_raw = answer_needle(llm, top_docs, q)
		out = f"{router_info}\n\n{ans_raw}\n\n(trace: {trace})"

	# Enhanced evaluation for questions with or without ground truth
	metrics_txt = ""
	compare_dict = {}
	
	# If ground truth is provided, use it directly
	if ground_truth and ground_truth.strip():
		try:
			context_texts = [d.page_content for d in top_docs]
			dataset = {
				"question": [q],
				"answer": [ans_raw],
				"ground_truth": [ground_truth.strip()],
				"contexts": [context_texts]  # List of context strings, not list of lists
			}
			m = run_eval(dataset)
			metrics_txt = pretty_metrics(m)
		except Exception as e:
			metrics_txt = f"(evaluation with provided ground truth failed: {e})"
	else:
		# Enhanced evaluation for questions without ground truth
		try:
			if ENHANCED_EVAL_AVAILABLE:
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
			else:
				# Fallback to existing GT/QA maps
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
				# Helper to tokenize text for comparison
				def _tok(s):
					return re.findall(r"\w+", s.lower()) if s else []
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
		except Exception as e:
			metrics_txt = f"(enhanced evaluation failed: {e})"
			compare_dict = {}

	# Logging + audit
	log = get_logger()
	log.info("Q: %s", q)
	log.info("Answer: %s", out)
	if metrics_txt:
		log.info("Metrics:\n%s", metrics_txt)
	

	
	try:
		from RAG.app.config import settings
		settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
		entry = {
			"question": q,
			"route": r,
			"router_trace": rtrace,
			"answer": out,
			"metrics": metrics_txt,
			"contexts": [
				{
					"file": d.metadata.get("file_name"),
					"page": d.metadata.get("page"),
					"section": d.metadata.get("section"),
				}
				for d in top_docs
			],
		}
		with open(settings.LOGS_DIR/"queries.jsonl", "a", encoding="utf-8") as f:
			f.write(json.dumps(entry, ensure_ascii=False) + "\n")
	except Exception:
		pass
	# Visibility updates for debug section and children
	_acc_upd = gr.update(visible=_dbg_visible, open=False)
	_router_upd = gr.update(value=_dbg_router_md, visible=_dbg_visible)
	_filters_upd = gr.update(value=_dbg_filters_json, visible=_dbg_visible)
	_dense_upd = gr.update(value=_dbg_dense_md, visible=_dbg_visible)
	_sparse_upd = gr.update(value=_dbg_sparse_md, visible=_dbg_visible)
	_hybrid_upd = gr.update(value=_dbg_hybrid_md, visible=_dbg_visible)
	_topdocs_upd = gr.update(value=_dbg_top_df, visible=_dbg_visible)
	_compare_upd = gr.update(value=compare_dict, visible=_dbg_visible)
	# Update figure preview slot when available; leave None to avoid clearing external viewers
	fig_update = gr.update(value=fig_path) if 'fig_path' in locals() and fig_path else None
	return out, metrics_txt, _acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, fig_update


def _run_agent(question: str, docs, hybrid, llm=None):
    """Run agent with tools for visibility."""
    steps = []
    try:
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
            from RAG.app.agents import answer_needle

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
                from RAG.app.eval_ragas import run_eval, pretty_metrics
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


def _do_audit(docs):
	"""Perform audit of figures."""
	try:
		from RAG.app.agent_tools import tool_audit_and_fill_figures as _audit_figs
		if _audit_figs is None:
			return {"error": "tool not available"}
		return _audit_figs(docs)
	except Exception:
		return {"error": "tool not available"}


def _do_plan(observations: str, llm):
	"""Generate a plan based on observations."""
	try:
		from RAG.app.agent_tools import tool_plan as _plan
		if _plan is None:
			return "(planner not available)"
		return _plan(observations, llm)
	except Exception:
		return "(planner not available)"
