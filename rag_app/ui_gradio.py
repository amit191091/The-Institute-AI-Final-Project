import gradio as gr
import json
from pathlib import Path

from app.agents import answer_needle, answer_summary, answer_table, route_question
from app.retrieve import apply_filters, query_analyzer, rerank_candidates
from app.eval_ragas import run_eval, pretty_metrics
from app.logger import get_logger


def _render_router_info(route: str, top_docs):
	heads = [f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]" for d in top_docs[:3]]
	return f"Route: {route} | Top contexts: {'; '.join(heads)}"


def _extract_table_figure_context(docs):
	subset = [d for d in docs if d.metadata.get("section") in ("Table", "Figure")]
	if not subset:
		return "(no table/figure contexts in top candidates)"
	out = []
	for d in subset[:3]:
		out.append(f"[{d.metadata.get('file_name')} p{d.metadata.get('page')}]\n{d.page_content[:800]}")
	return "\n\n---\n\n".join(out)


def _fmt_docs(docs, max_items=8):
	out = []
	for d in docs[:max_items]:
		out.append(f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]\n{d.page_content[:300]}")
	return "\n\n---\n\n".join(out) if out else "(none)"


def build_ui(docs, hybrid, llm, debug=None):
	log = get_logger()
	def on_ask(q, ground_truth, debug_toggle):
		qa = query_analyzer(q)
		try:
			cands = hybrid.invoke(q)
		except Exception:
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
				sparse_docs = debug["sparse"].get_relevant_documents(q)
			except Exception:
				try:
					sparse_docs = debug["sparse"].invoke(q)
				except Exception:
					pass
		filtered = apply_filters(cands, qa["filters"])
		top_docs = rerank_candidates(q, filtered, top_n=8)
		r = route_question(q)
		router_info = _render_router_info(r, top_docs)
		# Reasoning-lite trace (no full CoT): what we filtered and reranked by
		trace = f"Keywords: {qa['keywords']} | Filters: {qa['filters']}"
		debug_block = ""
		if debug_toggle:
			debug_block = (
				"\n\n=== DEBUG ===\n"
				+ "Dense (k≈10):\n" + _fmt_docs(dense_docs) + "\n\n"
				+ "Sparse (k≈10):\n" + _fmt_docs(sparse_docs) + "\n\n"
				+ "Hybrid candidates (pre-filter):\n" + _fmt_docs(cands)
			)
		if r == "summary":
			ans = answer_summary(llm, top_docs, q)
		elif r == "table":
			table_ctx = _extract_table_figure_context(top_docs)
			ans = answer_table(llm, top_docs, q)
			ans = f"{router_info}\n\nTable/Figure context preview:\n{table_ctx}\n\n---\n\n{ans}\n\n(trace: {trace}){debug_block}"
			metrics_txt = ""
			if ground_truth and ground_truth.strip():
				try:
					dataset = {
						"question": [q],
						"answer": [ans],
						"contexts": [[d.page_content for d in top_docs]],
						"ground_truth": [ground_truth],
						"ground_truths": [[ground_truth]],
					}
					m = run_eval(dataset)
					metrics_txt = pretty_metrics(m)
				except Exception as e:
					metrics_txt = f"(metrics failed: {e})"
			# Log Q&A + metrics
			log.info("Q: %s", q)
			log.info("Answer: %s", ans)
			if metrics_txt:
				log.info("Metrics:\n%s", metrics_txt)
			# append JSONL audit
			try:
				Path("logs").mkdir(exist_ok=True)
				entry = {
					"question": q,
					"route": r,
					"answer": ans,
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
				with open(Path("logs")/"queries.jsonl", "a", encoding="utf-8") as f:
					f.write(json.dumps(entry, ensure_ascii=False) + "\n")
			except Exception:
				pass
			return ans, metrics_txt
		else:
			ans = answer_needle(llm, top_docs, q)
		out = f"{router_info}\n\n{ans}\n\n(trace: {trace}){debug_block}"
		metrics_txt = ""
		if ground_truth and ground_truth.strip():
			try:
				dataset = {
					"question": [q],
					"answer": [ans],
					"contexts": [[d.page_content for d in top_docs]],
					"ground_truth": [ground_truth],
					"ground_truths": [[ground_truth]],
				}
				m = run_eval(dataset)
				metrics_txt = pretty_metrics(m)
			except Exception as e:
				metrics_txt = f"(metrics failed: {e})"
		# Log Q&A + metrics
		log.info("Q: %s", q)
		log.info("Answer: %s", out)
		if metrics_txt:
			log.info("Metrics:\n%s", metrics_txt)
		# append JSONL audit
		try:
			Path("logs").mkdir(exist_ok=True)
			entry = {
				"question": q,
				"route": r,
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
			with open(Path("logs")/"queries.jsonl", "a", encoding="utf-8") as f:
				f.write(json.dumps(entry, ensure_ascii=False) + "\n")
		except Exception:
			pass
		return out, metrics_txt

	return gr.Interface(
		fn=on_ask,
		inputs=[
			gr.Textbox(label="Question"),
			gr.Textbox(label="Ground truth (optional) for evaluation"),
			gr.Checkbox(label="Show retrieval debug (dense/sparse/hybrid)", value=False),
		],
		outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Metrics")],
		title="Hybrid RAG – Failure Reports",
		description="Router + Summary / Needle / Table QA",
	)

