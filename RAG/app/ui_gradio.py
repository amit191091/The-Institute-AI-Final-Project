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
	log = get_logger()
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
		# section fallback if nothing after filtering
		sec = qa["filters"].get("section") if qa and qa.get("filters") else None
		if sec and not filtered:
			filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
		top_docs = rerank_candidates(q, filtered, top_n=8)
		r = route_question(q)
		router_info = _render_router_info(r, top_docs)
		trace = f"Keywords: {qa['keywords']} | Filters: {qa['filters']}"
		debug_block = ""
		if debug_toggle:
			debug_block = (
				"\n\n=== DEBUG ===\n"
				+ f"Filters used: {qa['filters']}\n"
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

	# Build a sleeker Blocks UI with tabs
	with gr.Blocks(title="Hybrid RAG – Failure Reports") as demo:
		gr.Markdown("## Hybrid RAG – Failure Reports\nRouter + Summary / Needle / Table QA")
		with gr.Tabs():
			with gr.Tab("Ask"):
				q = gr.Textbox(label="Question", placeholder="Ask about figures, tables, procedures, conclusions…")
				gt = gr.Textbox(label="Ground truth (optional)")
				dbg = gr.Checkbox(label="Show retrieval debug", value=False)
				btn = gr.Button("Ask", variant="primary")
				# Render answer as Markdown so tables display nicely
				ans = gr.Markdown()
				metrics = gr.Textbox(label="Metrics", lines=3)
				btn.click(on_ask, inputs=[q, gt, dbg], outputs=[ans, metrics])

			with gr.Tab("Figures"):
				# Build a gallery of extracted figures
				fig_paths = [d.metadata.get("image_path") for d in docs if d.metadata.get("section") == "Figure" and d.metadata.get("image_path")]
				fig_paths = [str(Path(p)) for p in fig_paths if p]
				if fig_paths:
					gr.Gallery(value=fig_paths, label="Extracted Figures", allow_preview=True, columns=4, height=400)
				else:
					gr.Markdown("(No extracted figures. Enable RAG_EXTRACT_IMAGES=true and rerun.)")

			with gr.Tab("Inspect"):
				gr.Markdown("### Top indexed docs (sample)")
				sample_docs = [d for d in docs[:12]]
				gr.Textbox(value=_fmt_docs(sample_docs, max_items=12), label="Sample Contexts", lines=15)

			with gr.Tab("DB Explorer"):
				gr.Markdown("### Browse indexed documents (filters below)")
				sec_dd = gr.Dropdown(choices=section_values, label="Section filter", value=None, allow_custom_value=True)
				qbox = gr.Textbox(label="Contains (text or metadata)")
				refresh = gr.Button("Refresh")
				df = gr.Dataframe(headers=["file", "page", "section", "anchor", "words", "table_md", "table_csv", "image_path", "preview"], wrap=True)
				def _on_refresh(fs, qq):
					return gr.update(value=_rows_for_df(fs, qq))
				refresh.click(_on_refresh, inputs=[sec_dd, qbox], outputs=[df])
				# initial load
				df.value = _rows_for_df(None, None)

		return demo

	# Fallback just in case – should not reach here
	return demo

