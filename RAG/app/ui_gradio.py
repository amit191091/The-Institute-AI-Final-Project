import gradio as gr
import json
from pathlib import Path

from app.agents import answer_needle, answer_summary, answer_table, route_question
from app.hierarchical import _ask_table_hierarchical
from app.retrieve import apply_filters, query_analyzer, rerank_candidates
from app.eval_ragas import run_eval, pretty_metrics
from app.logger import get_logger
from app.auto_evaluator import AutoEvaluator
from app.enhanced_question_analyzer import EnhancedQuestionAnalyzer
from app.evaluation_wrapper import (
    EVAL_AVAILABLE, 
    test_google_api, 
    test_ragas, 
    generate_ground_truth, 
    evaluate_rag
)

# Initialize logger globally
log = get_logger()


def _render_router_info(route: str, top_docs):
	heads = [f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]" for d in top_docs[:3]]
	return f"Route: {route} | Top contexts: {'; '.join(heads)}"


def _extract_table_figure_context(docs):
	"""Return a Markdown-ready preview of the top Table/Figure contexts.
	If a table markdown file was generated, embed its content; otherwise fall back to the chunk text.
	"""
	subset = [d for d in docs if d.metadata.get("section") in ("Table", "Figure")]
	if not subset:
		return "(no table/figure contexts in top candidates)"
	out = []
	for d in subset[:3]:
		head = f"[{d.metadata.get('file_name')} p{d.metadata.get('page')}]"
		md_path = d.metadata.get("table_md_path")
		csv_path = d.metadata.get("table_csv_path")
		if md_path and Path(str(md_path)).exists():
			try:
				content = Path(str(md_path)).read_text(encoding="utf-8")
				link_line = f"(table files: [markdown]({md_path})" + (f" | [csv]({csv_path})" if csv_path else ")")
				out.append(f"{head}\n{link_line}\n\n{content}")
				continue
			except Exception:
				pass
		# Fallback to the chunk page content
		out.append(f"{head}\n{d.page_content[:1000]}")
	return "\n\n---\n\n".join(out)


def _fmt_docs(docs, max_items=8):
	out = []
	for d in docs[:max_items]:
		out.append(f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]\n{d.page_content[:300]}")
	return "\n\n---\n\n".join(out) if out else "(none)"


def build_ui(docs, hybrid, llm, debug=None):
	# Precompute unique sections for filters
	section_values = sorted({(d.metadata or {}).get("section") or "" for d in docs})
	section_values = [s for s in section_values if s]

	def _rows_for_df(filter_section: str | None, q: str | None, limit: int = 300):
		"""Build rows for the DB Explorer table with light filtering."""
		fs = (filter_section or "").strip()
		qq = (q or "").strip().lower()
		rows = []
		for d in docs:
			md = d.metadata or {}
			if fs and md.get("section") != fs:
				continue
			txt = d.page_content or ""
			if qq and qq not in (txt.lower() + " " + " ".join(map(str, md.values())).lower()):
				continue
			rows.append([
				md.get("file_name"),
				md.get("page"),
				md.get("section"),
				md.get("anchor"),
				0 if not txt else len(txt.split()),
				md.get("table_md_path") or "",
				md.get("table_csv_path") or "",
				md.get("image_path") or "",
				(txt[:200] + ("…" if len(txt) > 200 else "")),
			])
			if len(rows) >= limit:
				break
		return rows

	def on_ask(q, ground_truth, debug_toggle):
		qa = query_analyzer(q)
		cands = hybrid.invoke(q)
		filtered = apply_filters(cands, qa["filters"])
		top_docs = rerank_candidates(q, filtered)[:8]
		r = route_question(q)
		
		# Enhanced evaluation for questions without ground truth
		metrics_txt = ""
		if not ground_truth.strip():
			try:
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
				
			except Exception as e:
				metrics_txt = f"(enhanced evaluation failed: {e})"
		else:
			# Standard evaluation with provided ground truth
			try:
				dataset = {
					"question": [q],
					"answer": [""],  # Will be filled after getting answer
					"ground_truth": [ground_truth],
					"contexts": [[d.page_content for d in top_docs]]
				}
				metrics = run_eval(dataset)
				metrics_txt = pretty_metrics(metrics)
			except Exception as e:
				metrics_txt = f"(evaluation failed: {e})"
		
		if r == "summary":
			ans = answer_summary(llm, top_docs, q)
		elif r == "table":
			# Use hierarchical search for table questions
			ans = _ask_table_hierarchical(docs, hybrid, llm, q, qa)
		else:
			ans = answer_needle(llm, top_docs, q)
		
		# Update dataset with actual answer for evaluation
		if ground_truth.strip():
			try:
				dataset = {
					"question": [q],
					"answer": [ans],
					"ground_truth": [ground_truth],
					"contexts": [[d.page_content for d in top_docs]]
				}
				metrics = run_eval(dataset)
				metrics_txt = pretty_metrics(metrics)
			except Exception as e:
				metrics_txt = f"(evaluation failed: {e})"
		
		router_info = _render_router_info(r, top_docs)
		out = f"{router_info}\n\n{ans}\n\n(trace: {qa})"
		
		if debug_toggle:
			debug_block = f"\n\n--- DEBUG ---\nDense: {len(dense_docs)} docs\nSparse: {len(sparse_docs)} docs\nFiltered: {len(filtered)} docs\nTop: {len(top_docs)} docs"
			out += debug_block
		
		# Logging + audit
		log.info("Q: %s", q)
		log.info("Answer: %s", out)
		if metrics_txt:
			log.info("Metrics:\n%s", metrics_txt)
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

	def on_shutdown():
		"""Gracefully shutdown the server."""
		log.info("Shutdown requested by user")
		try:
			# Close any open files or connections
			import os
			import signal
			
			# Send SIGTERM to current process
			os.kill(os.getpid(), signal.SIGTERM)
			return "🔄 Server shutdown initiated..."
		except Exception as e:
			log.error(f"Error during shutdown: {e}")
			return f"❌ Shutdown error: {e}"

	# Evaluation functions
	def on_test_google_api():
		return test_google_api()

	def on_test_ragas():
		return test_ragas()

	def on_generate_ground_truth(num_questions: int):
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
				
				route = route_question(question)
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

	def on_evaluate_rag():
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
				
				route = route_question(question)
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

	# Build the Gradio interface
	with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as demo:
		gr.Markdown("# 🤖 RAG System - Gear Failure Analysis")
		gr.Markdown("Ask questions about gear failure analysis, transmission ratios, wear measurements, and more.")
		
		with gr.Tab("🔍 Query"):
			with gr.Row():
				with gr.Column(scale=3):
					question = gr.Textbox(
						label="Question",
						placeholder="e.g., What is the transmission ratio? When did the failure occur?",
						lines=3
					)
					ground_truth = gr.Textbox(
						label="Ground Truth (Optional)",
						placeholder="Expected answer for evaluation...",
						lines=2
					)
					debug_toggle = gr.Checkbox(label="Show Debug Info", value=False)
					
					with gr.Row():
						ask_btn = gr.Button("🔍 Ask", variant="primary")
						clear_btn = gr.Button("🗑️ Clear")
						shutdown_btn = gr.Button("🛑 Shutdown Server", variant="stop")
					
					answer = gr.Textbox(label="Answer", lines=10, interactive=False)
					metrics = gr.Textbox(label="Evaluation Metrics", lines=8, interactive=False)
				
				with gr.Column(scale=2):
					gr.Markdown("### 📊 System Info")
					gr.Markdown(f"**Total Documents:** {len(docs)}")
					gr.Markdown(f"**Sections:** {', '.join(section_values)}")
					
					if debug:
						gr.Markdown("### 🔧 Debug Tools")
						gr.Markdown("Debug information available")
		
		with gr.Tab("📋 Evaluation"):
			gr.Markdown("### RAGAS Evaluation Tools")
			with gr.Row():
				test_google_btn = gr.Button("Test Google API")
				test_ragas_btn = gr.Button("Test RAGAS")
			with gr.Row():
				generate_gt_btn = gr.Button("Generate Ground Truth")
				num_questions = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of Questions")
			evaluate_btn = gr.Button("Evaluate RAG System", variant="primary")
			eval_output = gr.Textbox(label="Evaluation Results", lines=15, interactive=False)
		
		with gr.Tab("🗄️ Database Explorer"):
			gr.Markdown("### Explore the loaded documents")
			with gr.Row():
				filter_section = gr.Dropdown(choices=[""] + section_values, label="Filter by Section", value="")
				search_query = gr.Textbox(label="Search Query", placeholder="Search in documents...")
			explore_btn = gr.Button("🔍 Explore")
			explore_df = gr.Dataframe(
				headers=["File", "Page", "Section", "Anchor", "Words", "MD Path", "CSV Path", "Image Path", "Preview"],
				label="Document Explorer"
			)
		
		# Event handlers
		ask_btn.click(
			on_ask,
			inputs=[question, ground_truth, debug_toggle],
			outputs=[answer, metrics]
		)
		
		clear_btn.click(
			lambda: ("", "", False, "", ""),
			outputs=[question, ground_truth, debug_toggle, answer, metrics]
		)
		
		shutdown_btn.click(
			on_shutdown,
			outputs=[answer]
		)
		
		explore_btn.click(
			_rows_for_df,
			inputs=[filter_section, search_query],
			outputs=[explore_df]
		)
		
		# Evaluation handlers
		test_google_btn.click(on_test_google_api, outputs=[eval_output])
		test_ragas_btn.click(on_test_ragas, outputs=[eval_output])
		generate_gt_btn.click(on_generate_ground_truth, inputs=[num_questions], outputs=[eval_output])
		evaluate_btn.click(on_evaluate_rag, outputs=[eval_output])
		
		# Keyboard shortcuts
		question.submit(
			on_ask,
			inputs=[question, ground_truth, debug_toggle],
			outputs=[answer, metrics]
		)
	
	return demo