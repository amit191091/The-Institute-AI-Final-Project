import gradio as gr
import pandas as pd
import json
from pathlib import Path
import difflib
import re
import html as _html

from app.agents import answer_needle, answer_summary, answer_table, route_question, route_question_ex
from app.retrieve import apply_filters, query_analyzer, rerank_candidates
from app.eval_ragas import run_eval, pretty_metrics
from app.logger import get_logger
from app.graphdb import run_cypher as _run_cypher  # optional, guarded by try/except in handler
try:
	from app.graph import build_graph, render_graph_html
except Exception:
	build_graph = None  # type: ignore
	render_graph_html = None  # type: ignore


def _render_router_info(route: str, top_docs):
	heads = [f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]" for d in top_docs[:3]]
	return f"Route: {route} | Top contexts: {'; '.join(heads)}"


def _rows_to_df(rows):
	"""Convert list of row lists to a pandas DataFrame with stable columns."""
	cols = [
		"file",
		"page",
		"section",
		"anchor",
		"words",
		"table_md",
		"table_csv",
		"image",
		"preview",
	]
	try:
		return pd.DataFrame(rows or [], columns=cols)
	except Exception:
		# Fallback: best-effort DataFrame without strict columns
		return pd.DataFrame(rows or [])


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
				link_line = f"(table files: [markdown]({md_path})" + (f" | [csv]({csv_path})" if csv_path else "") + ")"
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


def build_ui(docs, hybrid, llm, debug=None) -> gr.Blocks:
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

	# Auto-loaded ground-truths and QA answer maps (normalized by question)
	gt_map = {"__loaded__": False, "map": {}, "norm": {}}
	qa_map = {"__loaded__": False, "map": {}, "norm": {}}

	def _norm_q(s: str) -> str:
		if not s:
			return ""
		s = str(s).lower().strip()
		s = re.sub(r"\s+", " ", s)
		return s.strip(".,:;!?-—\u2013\u2014\"'()[]{}")

	def _load_gt_file(path: str | dict | object):
		# Normalize to a path: accept string, dict(File), or object with .name
		if isinstance(path, dict):
			path = path.get("name") or path.get("path") or ""
		elif hasattr(path, "name"):
			try:
				path = getattr(path, "name")
			except Exception:
				path = str(path)
		if not path:
			return "(no ground-truth file loaded)"
		try:
			path_str = str(path)
			p = Path(path_str)
			if not p.exists():
				return f"(file not found: {path_str})"
			# minimal loader compatible with our main util
			rows = []
			if p.suffix.lower() == ".jsonl":
				with open(p, "r", encoding="utf-8") as f:
					for line in f:
						line=line.strip()
						if not line:
							continue
						try:
							rows.append(json.loads(line))
						except Exception:
							pass
			else:
				rows = json.load(open(p, "r", encoding="utf-8"))
				if isinstance(rows, dict):
					rows = [{"question": k, "ground_truths": v} for k, v in rows.items()]
			m = {}
			for r in rows or []:
				if not isinstance(r, dict):
					continue
				q = r.get("question") or r.get("q") or r.get("prompt") or r.get("key")
				gts = r.get("ground_truths") or r.get("ground_truth") or r.get("answers") or r.get("answer") or r.get("value")
				if not q:
					continue
				if isinstance(gts, str):
					m[str(q)] = [gts]
				elif isinstance(gts, list):
					m[str(q)] = [str(x) for x in gts]
				elif gts is not None:
					m[str(q)] = [str(gts)]
			gt_map["__loaded__"] = True
			gt_map["map"] = m
			gt_map["norm"] = { _norm_q(k): v for k, v in m.items() }
			sample = ", ".join(list(m.keys())[:2])
			return f"Loaded {len(m)} ground truths from {path_str}. Sample keys: {sample}"
		except Exception as e:
			return f"(failed to load: {e})"

	def _load_qa_file(path: str | dict | object):
		# Load QA json/jsonl mapping question -> answer
		if isinstance(path, dict):
			path = path.get("name") or path.get("path") or ""
		elif hasattr(path, "name"):
			try:
				path = getattr(path, "name")
			except Exception:
				path = str(path)
		if not path:
			return "(no QA file loaded)"
		try:
			p = Path(str(path))
			if not p.exists():
				return f"(QA file not found: {p})"
			rows = []
			if p.suffix.lower() == ".jsonl":
				with open(p, "r", encoding="utf-8") as f:
					for line in f:
						line=line.strip()
						if not line:
							continue
						try:
							rows.append(json.loads(line))
						except Exception:
							pass
			else:
				rows = json.load(open(p, "r", encoding="utf-8"))
			m = {}
			for r in rows or []:
				if not isinstance(r, dict):
					continue
				q = r.get("question") or r.get("q")
				a = r.get("answer") or r.get("reference")
				if q and a:
					m[str(q)] = str(a)
			qa_map["__loaded__"] = True
			qa_map["map"] = m
			qa_map["norm"] = { _norm_q(k): v for k, v in m.items() }
			return f"Loaded {len(m)} QA pairs from {p}"
		except Exception as e:
			return f"(failed to load QA: {e})"

	# Auto-load default ground truths and QA if files exist
	try:
		for _cand in [Path("gear_wear_ground_truth.json"), Path("data")/"gear_wear_ground_truth.json"]:
			if _cand.exists():
				msg = _load_gt_file(str(_cand))
				log.info("GT auto-load: %s", msg)
				break
	except Exception:
		pass
	try:
		for _cand in [Path("gear_wear_qa.jsonl"), Path("data")/"gear_wear_qa.jsonl", Path("gear_wear_qa.json"), Path("data")/"gear_wear_qa.json"]:
			if _cand.exists():
				msg = _load_qa_file(str(_cand))
				log.info("QA auto-load: %s", msg)
				break
	except Exception:
		pass

	def on_ask(q, debug_toggle):
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
		# Fallbacks to avoid empty contexts for metrics/answering
		if not top_docs:
			# Use unfiltered candidates
			top_docs = (cands or [])[:8]
		if not top_docs:
			# Last resort use first few indexed docs
			top_docs = docs[:8]
		r, rtrace = route_question_ex(q)
		router_info = _render_router_info(r, top_docs) + f" | Agent: {r} | Rules: {', '.join(rtrace.get('matched', []))}"
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
			ans_raw = answer_summary(llm, top_docs, q)
			out = f"{router_info}\n\n{ans_raw}\n\n(trace: {trace}){debug_block}"
		elif r == "table":
			table_ctx = _extract_table_figure_context(top_docs)
			ans_raw = answer_table(llm, top_docs, q)
			out = f"{router_info}\n\nTable/Figure context preview:\n{table_ctx}\n\n---\n\n{ans_raw}\n\n(trace: {trace}){debug_block}"
		else:
			ans_raw = answer_needle(llm, top_docs, q)
			out = f"{router_info}\n\n{ans_raw}\n\n(trace: {trace}){debug_block}"

		# Always compute metrics per query using GT/QA maps with fuzzy matching and QA fallback
		metrics_txt = ""
		try:
			gts = []
			nq = _norm_q(q)
			# Exact GT match
			if gt_map.get("__loaded__") and nq in gt_map.get("norm", {}):
				gts = gt_map["norm"][nq]
			# Fuzzy GT if not exact
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
			if not gts and qa_map.get("__loaded__"):
				if nq in qa_map.get("norm", {}):
					gts = [qa_map["norm"][nq]]
				else:
					keys = list(qa_map["norm"].keys())
					best = None; best_s = 0.0
					for k in keys:
						s = difflib.SequenceMatcher(None, nq, k).ratio()
						if s > best_s:
							best_s = s; best = k
					if best is not None and best_s >= 0.75:
						gts = [qa_map["norm"][best]]
			ref = gts[0] if isinstance(gts, list) and gts else (ans_raw or "")
			dataset = {
				"question": [q],
				"answer": [ans_raw],
				"contexts": [[d.page_content for d in top_docs]],
				"ground_truths": [gts],
				"reference": [ref],
			}
			m = run_eval(dataset)
			metrics_txt = pretty_metrics(m)
			# If all metrics are NaN, hint about API keys
			vals = [str(m.get(k)) for k in ("faithfulness","answer_relevancy","context_precision","context_recall")]
			if all(v == 'nan' for v in vals):
				metrics_txt += "\n(note: metrics require OPENAI_API_KEY or GOOGLE_API_KEY for RAGAS)"
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
				dbg = gr.Checkbox(label="Show retrieval debug", value=False)
				btn = gr.Button("Ask", variant="primary")
				# Render answer as Markdown so tables display nicely
				ans = gr.Markdown()
				metrics = gr.Textbox(label="Metrics", lines=3)
				btn.click(on_ask, inputs=[q, dbg], outputs=[ans, metrics])


			with gr.Tab("Figures"):
				# Build a gallery of extracted figures
				fig_paths = [d.metadata.get("image_path") for d in docs if d.metadata.get("section") == "Figure" and d.metadata.get("image_path")]
				fig_paths = [str(Path(p)) for p in fig_paths if p]
				if fig_paths:
					# Recent Gradio versions preview by default; keep args minimal for compatibility
					gr.Gallery(value=fig_paths, label="Extracted Figures", columns=4, height=400)
				else:
					gr.Markdown("(No extracted figures. Enable RAG_EXTRACT_IMAGES=true and rerun.)")

			with gr.Tab("Inspect"):
				gr.Markdown("### Top indexed docs (sample)")
				sample_docs = [d for d in docs[:12]]
				gr.Textbox(value=_fmt_docs(sample_docs, max_items=12), label="Sample Contexts", lines=15)

			with gr.Tab("Graph"):
				gr.Markdown("### Knowledge Graph (auto-built)")
				# If the Main built a graph, it will be at logs/graph.html
				_graph_view = gr.HTML(value="")
				_graph_status = gr.Markdown()
				btn_graph = gr.Button("Generate / Refresh Graph")
				gr.Markdown("#### Graph DB (Neo4j) — optional")
				# Prefill with a sample so clicks don't pass an empty string on some Gradio builds
				cypher = gr.Textbox(label="Cypher query", value="MATCH (n) RETURN n LIMIT 10", placeholder="Type a Cypher query…", lines=2)
				btn_cypher = gr.Button("Run Cypher")
				cypher_out = gr.JSON(label="Results")

				def _gen_graph():
					try:
						if build_graph is None or render_graph_html is None:
							return gr.update(value=""), "(graph module not available; install dependencies: networkx, pyvis)"
						G = build_graph(docs)
						Path("logs").mkdir(exist_ok=True)
						out = Path("logs")/"graph.html"
						render_graph_html(G, str(out))
						html_data = out.read_text(encoding="utf-8")
						# Best-effort inline via iframe srcdoc; also provide a link for full view
						iframe = f"<p><a href='file:///{out.resolve().as_posix()}' target='_blank'>Open graph.html in browser</a></p>" \
								 f"<iframe style='width:100%;height:650px;border:1px solid #ddd' srcdoc=\"{_html.escape(html_data)}\"></iframe>"
						return gr.update(value=iframe), "Graph updated."
					except Exception as e:
						return gr.update(value=""), f"(failed to build graph: {e})"

				# Initial load if file exists
				try:
					graph_html_path = Path("logs")/"graph.html"
					if graph_html_path.exists():
						html_data = graph_html_path.read_text(encoding="utf-8")
						iframe = f"<p><a href='file:///{graph_html_path.resolve().as_posix()}' target='_blank'>Open graph.html in browser</a></p>" \
								 f"<iframe style='width:100%;height:650px;border:1px solid #ddd' srcdoc=\"{_html.escape(html_data)}\"></iframe>"
						_graph_view.value = iframe
					else:
						_graph_status.value = "(Graph not available yet – click the button to generate it.)"
				except Exception:
					_graph_status.value = "(Graph not available yet – click the button to generate it.)"
				btn_graph.click(_gen_graph, inputs=[], outputs=[_graph_view, _graph_status])

				def _run_cypher_ui(q:str=""):
					try:
						query = (q or "").strip()
						if not query:
							return {"error": "empty query", "hint": "Example: MATCH (n) RETURN n LIMIT 10"}
						rows = _run_cypher(query)  # type: ignore
						# Return rows directly; gr.JSON can render lists of dicts
						return rows if rows else {"rows": [], "note": "Query ran, no results returned."}
					except Exception as e:
						# Surface config issues clearly (e.g., missing Neo4j env vars)
						return {"error": str(e)}
				# Support both click and Enter-to-submit
				btn_cypher.click(_run_cypher_ui, inputs=[cypher], outputs=[cypher_out])
				cypher.submit(_run_cypher_ui, inputs=[cypher], outputs=[cypher_out])

			with gr.Tab("DB Explorer"):
				gr.Markdown("### Browse indexed documents (filters below)")
				# Add an '(All)' option to avoid None handling differences across versions
				sec_choices = ["(All)"] + section_values
				sec_dd = gr.Dropdown(choices=sec_choices, label="Section filter", value="(All)")
				qbox = gr.Textbox(label="Contains (text or metadata)")
				refresh = gr.Button("Refresh")
				# Initialize the table with data at construction time for v5 compatibility
				_initial_rows = _rows_for_df(None, None)
				df = gr.Dataframe(
					value=_rows_to_df(_initial_rows),
					wrap=True,
					interactive=False,
				)
				def _on_refresh(fs, qq):
					# Normalize '(All)' to no filter and return a pandas DataFrame
					fsn = None if (fs in (None, "", "(All)")) else fs
					rows = _rows_for_df(fsn, qq)
					return gr.update(value=_rows_to_df(rows))
				refresh.click(_on_refresh, inputs=[sec_dd, qbox], outputs=[df])
				# initial load handled via value above

	return demo

	# Fallback just in case – should not reach here
	return demo

