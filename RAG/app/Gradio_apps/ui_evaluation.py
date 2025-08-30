"""
Evaluation functionality for the Gradio interface.
"""
import difflib
import re
from RAG.app.Agent_Components.agents import answer_needle, answer_summary, answer_table, route_question_ex
from RAG.app.retrieve import apply_filters, query_analyzer, rerank_candidates
from RAG.app.Evaluation_Analysis.evaluation_utils import run_eval, pretty_metrics
from RAG.app.logger import get_logger

# Enhanced evaluation imports
try:
    from RAG.app.Evaluation_Analysis.auto_evaluator import AutoEvaluator
    from RAG.app.Evaluation_Analysis.enhanced_question_analyzer import EnhancedQuestionAnalyzer
    from RAG.app.Evaluation_Analysis.evaluation_wrapper import (
        EVAL_AVAILABLE, 
        test_google_api, 
        test_ragas, 
        generate_ground_truth, 
        evaluate_rag
    )
    ENHANCED_EVAL_AVAILABLE = True
except Exception:
    ENHANCED_EVAL_AVAILABLE = False


def on_test_google_api():
	if not ENHANCED_EVAL_AVAILABLE:
		return "Enhanced evaluation modules not available"
	return test_google_api()

def on_test_ragas():
	if not ENHANCED_EVAL_AVAILABLE:
		return "Enhanced evaluation modules not available"
	return test_ragas()

def on_generate_ground_truth(docs, hybrid, llm, num_questions: int):
	if not ENHANCED_EVAL_AVAILABLE:
		return "Enhanced evaluation modules not available"
	
	# Create a simple pipeline wrapper for evaluation
	class PipelineWrapper:
		def __init__(self, docs, hybrid, llm):
			self.docs = docs
			self.hybrid = hybrid
			self.llm = llm
		
		def query(self, question: str):
			qa = query_analyzer(question)
			cands = self.hybrid.invoke(question)
			filtered = apply_filters(cands, qa["filters"])
			top_docs = rerank_candidates(question, filtered)[:5]
			
			route = route_question_ex(question)[0]
			if route == "summary":
				ans = answer_summary(self.llm, top_docs, question)
			elif route == "table":
				ans = answer_table(self.llm, top_docs, question)
			else:
				ans = answer_needle(self.llm, top_docs, question)
			
			return {
				"answer": ans,
				"contexts": top_docs
			}
	
	pipeline = PipelineWrapper(docs, hybrid, llm)
	return generate_ground_truth(pipeline, num_questions)

def on_evaluate_rag(docs, hybrid, llm):
	if not ENHANCED_EVAL_AVAILABLE:
		return "Enhanced evaluation modules not available"
	
	# Create pipeline wrapper
	class PipelineWrapper:
		def __init__(self, docs, hybrid, llm):
			self.docs = docs
			self.hybrid = hybrid
			self.llm = llm
		
		def query(self, question: str):
			qa = query_analyzer(question)
			cands = self.hybrid.invoke(question)
			filtered = apply_filters(cands, qa["filters"])
			top_docs = rerank_candidates(question, filtered)[:5]
			
			route = route_question_ex(question)[0]
			if route == "summary":
				ans = answer_summary(self.llm, top_docs, question)
			elif route == "table":
				ans = answer_table(self.llm, top_docs, question)
			else:
				ans = answer_needle(self.llm, top_docs, question)
			
			return {
				"answer": ans,
				"contexts": top_docs
			}
	
	pipeline = PipelineWrapper(docs, hybrid, llm)
	return evaluate_rag(pipeline)


def evaluate_question_with_ground_truth(q, ans_raw, top_docs, ground_truth, gt_map, qa_map):
    """Evaluate a question with provided ground truth or fallback to enhanced evaluation."""
    metrics_txt = ""
    compare_dict = {}
    
    # If ground truth is provided, use it directly
    if ground_truth and ground_truth.strip():
        try:
            dataset = {
                "question": [q],
                "answer": [ans_raw],
                "ground_truth": [ground_truth.strip()],
                "contexts": [[d.page_content for d in top_docs]]
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
                dataset = {
                    "question": [q],
                    "answer": [ans_raw],
                    "contexts": [[d.page_content for d in top_docs]],
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
    
    return metrics_txt, compare_dict


def _norm_q(s: str) -> str:
    """Normalize question text for comparison."""
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(".,:;!?-Γאפ\u2013\u2014\"'()[]{}")
