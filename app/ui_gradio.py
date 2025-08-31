import gradio as gr
import pandas as pd
import json
from pathlib import Path
import difflib
import re
import html as _html
import os


from app.agents import answer_needle, answer_summary, answer_table, route_question_ex
from app.retrieve import apply_filters, query_analyzer, rerank_candidates
from app.eval_ragas import run_eval, pretty_metrics
from app.logger import get_logger
from app.graphdb import run_cypher as _run_cypher  # optional, guarded by try/except in handler
from app.normalized_loader import load_normalized_docs
try:
	from app.graph import build_graph, render_graph_html
except Exception:
	build_graph = None  # type: ignore
	render_graph_html = None  # type: ignore

# Agent tool shims
try:
	from app.agent_tools import tool_analyze_query, tool_retrieve_candidates, tool_retrieve_filtered, tool_list_figures
except Exception:
	tool_analyze_query = None  # type: ignore
	tool_retrieve_candidates = None  # type: ignore
	tool_retrieve_filtered = None  # type: ignore
	tool_list_figures = None  # type: ignore

# Optional orchestrator trace
try:
	from app.agent_orchestrator import run as run_orchestrator
except Exception:
	run_orchestrator = None  # type: ignore


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
		"figure_number",
		"figure_order",
		"table_md",
		"table_csv",
		"image",
		"score",
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
		out.append(f"{head}\n{d.page_content[:2000]}")
	return "\n\n---\n\n".join(out)


def _fmt_docs(docs, max_items=8):
	out = []
	for d in docs[:max_items]:
		out.append(f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]\n{d.page_content[:1500]}")
	return "\n\n---\n\n".join(out) if out else "(none)"


def build_ui(docs, hybrid, llm, debug=None) -> gr.Blocks:
	log = get_logger()
	# Precompute unique sections for filters
	section_values = sorted({(d.metadata or {}).get("section") or "" for d in docs})
	section_values = [s for s in section_values if s]

	def _rows_from_docs(_docs, limit: int = 300):
		rows = []
		for d in _docs[:limit]:
			md = d.metadata or {}
			txt = d.page_content or ""
			sec = md.get("section") or md.get("section_type")
			# Build preview consistent with snapshot: Figure/Table label preferred
			prev = ""
			if sec == "Figure":
				prev = str(md.get("figure_label") or "")
				if not prev:
					prev = txt[:200]
			elif sec == "Table":
				prev = str(md.get("table_label") or "")
				if not prev:
					# Try compose from first SUMMARY line
					lines = (txt or "").splitlines()
					summ = None
					for i, ln in enumerate(lines):
						if ln.strip().upper() == "SUMMARY:" and i + 1 < len(lines):
							summ = lines[i + 1].strip(); break
					if summ:
						no = md.get('table_number')
						prev = f"Table {no}: {summ}" if no is not None and str(no).strip() != "" else summ
					else:
						prev = txt[:200]
			else:
				prev = txt[:200]
			rows.append([
				md.get("file_name"),
				md.get("page"),
				md.get("section"),
				md.get("anchor"),
				0 if not txt else len(txt.split()),
				md.get("figure_number") or "",
				md.get("figure_order") or "",
				md.get("table_md_path") or "",
				md.get("table_csv_path") or "",
				md.get("image_path") or "",
				round(float(md.get("_score") or 0.0), 4),
				prev,
			])
		return rows

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
			sec = md.get("section") or md.get("section_type")
			if sec == "Figure":
				prev = md.get("figure_label") or (txt[:200])
			elif sec == "Table":
				if md.get("table_label"):
					prev = md.get("table_label")
				else:
					lines = (txt or "").splitlines()
					summ = None
					for i, ln in enumerate(lines):
						if ln.strip().upper() == "SUMMARY:" and i + 1 < len(lines):
							summ = lines[i + 1].strip(); break
					prev = (f"Table {md.get('table_number')}: {summ}" if summ and md.get('table_number') else (summ or txt[:200]))
			else:
				prev = (txt[:200])
			if qq and qq not in (txt.lower() + " " + " ".join(map(str, md.values())).lower()):
				continue
			rows.append([
				md.get("file_name"),
				md.get("page"),
				md.get("section"),
				md.get("anchor"),
				0 if not txt else len(txt.split()),
				md.get("figure_number") or "",
				md.get("figure_order") or "",
				md.get("table_md_path") or "",
				md.get("table_csv_path") or "",
				md.get("image_path") or "",
				prev,
			])
			if len(rows) >= limit:
				break
		return rows

	# Auto-loaded ground-truths and QA answer maps (normalized by question)
	# Additionally keep optional by-id indexes to join context_free datasets (QA has id+question, GT has id->answer)
	gt_map = {"__loaded__": False, "map": {}, "norm": {}, "by_id": {}}  # type: ignore[dict-item]
	qa_map = {"__loaded__": False, "map": {}, "norm": {}, "by_id": {}}  # type: ignore[dict-item]

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
			# flexible loader: supports jsonl, flat dicts, lists, and nested context_free JSON
			loaded = None
			if p.suffix.lower() == ".jsonl":
				rows = []
				with open(p, "r", encoding="utf-8") as f:
					for line in f:
						line=line.strip()
						if not line:
							continue
						try:
							rows.append(json.loads(line))
						except Exception:
							pass
				loaded = rows
			else:
				loaded = json.load(open(p, "r", encoding="utf-8"))

			# New fast-path: context_free GT provided as a list of {id, answer/acceptable_answers}
			# Build by-id answer lists and defer question join until QA is loaded.
			if isinstance(loaded, list) and loaded and all(isinstance(it, dict) for it in loaded):
				try:
					by_id: dict[str, list[str]] = {}
					for it in loaded:
						_i = it.get("id") or it.get("qid") or it.get("question_id") or it.get("key")
						if not _i:
							continue
						vals: list[str] = []
						acc = it.get("acceptable_answers")
						ans = it.get("ground_truth") or it.get("answer") or it.get("value")
						if isinstance(acc, list) and acc:
							vals = [str(x) for x in acc]
						elif ans is not None:
							vals = [str(ans)]
						if vals:
							by_id[str(_i)] = vals
					if by_id:
						gt_map["by_id"] = by_id
						# If QA by-id is already present, join to build question->GTs
						qa_by_id = qa_map.get("by_id") or {}
						if isinstance(qa_by_id, dict) and qa_by_id:
							joined: dict[str, list[str]] = {}
							for _id, _q in qa_by_id.items():
								gts = by_id.get(str(_id))
								if _q and gts:
									joined[str(_q)] = [str(x) for x in gts]
							if joined:
								gt_map["map"] = joined
								gt_map["norm"] = { _norm_q(k): v for k, v in joined.items() }
						gt_map["__loaded__"] = True
						# Craft message reflecting whether join occurred
						samplek = list((gt_map.get("map") or {}).keys())[:2]
						if samplek:
							return f"Loaded {len(gt_map['map'])} ground truths (joined by id) from {path_str}. Sample keys: {', '.join(samplek)}"
						return f"Loaded {len(by_id)} GT ids from {path_str} (waiting for QA to join questions)"
				except Exception:
					# fall through to other parsers
					pass

			# Fast-path: context_free schema detected (top-level id -> {answer, acceptable_answers, ...})
			if isinstance(loaded, dict) and loaded and all(isinstance(v, dict) for v in loaded.values()):
				# Build by-id answers list and keep for later join with QA file
				by_id: dict[str, list[str]] = {}
				for kid, v in loaded.items():
					try:
						ans = v.get("answer")
						acc = v.get("acceptable_answers")
						vals: list[str] = []
						if isinstance(acc, list) and acc:
							vals = [str(x) for x in acc]
						elif ans is not None:
							vals = [str(ans)]
						if vals:
							by_id[str(kid)] = vals
					except Exception:
						pass
				# Save by-id map and, if QA by-id exists, join to produce question->GTs
				if by_id:
					gt_map["by_id"] = by_id
					m_joined: dict[str, list[str]] = {}
					qa_by_id = qa_map.get("by_id") or {}
					if isinstance(qa_by_id, dict) and qa_by_id:
						for _id, _q in qa_by_id.items():
							gts = by_id.get(str(_id))
							if _q and gts:
								m_joined[str(_q)] = [str(x) for x in gts]
						# Update GT maps from join (leave QA loader to set QA map)
						if m_joined:
							gt_map["map"] = m_joined
							gt_map["norm"] = { _norm_q(k): v for k, v in m_joined.items() }
					gt_map["__loaded__"] = True
					samplek = list((gt_map.get("map") or {}).keys())[:2]
					if samplek:
						return f"Loaded {len(gt_map['map'])} ground truths (joined by id) from {path_str}. Sample keys: {', '.join(samplek)}"
					return f"Loaded {len(by_id)} GT ids from {path_str} (waiting for QA to join questions)"

			def _as_list_of_qgt(obj):
				# Normalize various shapes to a list of {question, ground_truths}
				out = []
				# Case 1: already a list of dicts
				if isinstance(obj, list):
					for it in obj:
						if isinstance(it, dict):
							out.append(it)
					return out
				# Case 2: flat mapping {question: [gts]}
				if isinstance(obj, dict):
					# If values look like strings or lists, assume a direct mapping
					vals = list(obj.values())
					if vals and all(isinstance(v, (str, list)) for v in vals):
						for k, v in obj.items():
							out.append({"question": k, "ground_truths": v})
						return out
					# Otherwise, recursively search for dicts that contain a question and any answer-like field
					acc = []
					def _recurse(o):
						if isinstance(o, dict):
							q = o.get("question") or o.get("q") or o.get("prompt") or o.get("key")
							g = o.get("ground_truths") or o.get("ground_truth") or o.get("answers") or o.get("answer") or o.get("reference") or o.get("value")
							if q and g is not None:
								acc.append({"question": q, "ground_truths": g})
							else:
								for v in o.values():
									_recurse(v)
						elif isinstance(o, list):
							for i in o:
								_recurse(i)
					_recurse(obj)
					return acc
				return out

			rows = _as_list_of_qgt(loaded)

			# If no rows were found (context_free schema), extract facts (especially dates) and synthesize Q->GT pairs
			if not rows:
				def _looks_like_date(s: str) -> bool:
					import re as _re
					s = str(s)
					# ISO-like or Month name patterns
					return bool(_re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", s) or _re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", s, _re.I))
				def _synthesize_q(k: str, v: str) -> str:
					kl = (k or "").strip().lower()
					if "failure" in kl:
						return "failure date"
					if "measurement" in kl and ("start" in kl or "begin" in kl):
						return "measurement start date"
					if "measurement" in kl and ("end" in kl or "finish" in kl):
						return "measurement end date"
					if "healthy" in kl and ("through" in kl or "until" in kl):
						return "healthy through date"
					if "date" in kl:
						return k
					# Fallback: append 'date' if value looks like a date
					return (k + " date").strip()
				facts: list[dict] = []
				def _recurseFacts(o, ctx_key: str | None = None):
					if isinstance(o, dict):
						# Common event schema: { event/name/label, date }
						keys = {str(k).lower(): k for k in o.keys()}
						val_date = None
						for k, v in o.items():
							if isinstance(v, str) and _looks_like_date(v):
								val_date = v
								best_key = k
								# Prefer companion name/label/event keys for question text
								for namek in ("event", "name", "label", "title", "what"):
									if namek in keys:
										best_key = keys[namek]
										break
								q = _synthesize_q(str(best_key), str(val_date))
								facts.append({"question": q, "ground_truths": [str(val_date)]})
						# Recurse into nested
						for v in o.values():
							_recurseFacts(v, ctx_key)
					elif isinstance(o, list):
						for i in o:
							_recurseFacts(i, ctx_key)
					elif isinstance(o, str):
						if _looks_like_date(o) and ctx_key:
							facts.append({"question": _synthesize_q(ctx_key, o), "ground_truths": [str(o)]})
					# other primitives ignored
				_recurseFacts(loaded)
				rows = facts
			m: dict[str, list[str]] = {}
			for r in rows or []:
				if not isinstance(r, dict):
					continue
				q = r.get("question") or r.get("q") or r.get("prompt") or r.get("key")
				gts = r.get("ground_truths") or r.get("ground_truth") or r.get("answers") or r.get("answer") or r.get("reference") or r.get("value")
				if not q:
					continue
				# normalize to list[str]
				vals: list[str] = []
				if isinstance(gts, str):
					vals = [gts]
				elif isinstance(gts, list):
					vals = [str(x) for x in gts]
				elif gts is not None:
					vals = [str(gts)]
				if not vals:
					continue
				key = str(q)
				m.setdefault(key, [])
				# de-duplicate while preserving order
				seen = set(m[key])
				for v in vals:
					if v not in seen:
						m[key].append(v)
						seen.add(v)
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
			# Detect context_free QA: rows with id+question but no answers; join with GT by id
			is_context_free = False
			if isinstance(rows, list) and rows and all(isinstance(r, dict) and r.get("id") and r.get("question") for r in rows):
				# Consider it context_free when most rows lack 'answer'/'reference'
				no_ans = sum(1 for r in rows if not (r.get("answer") or r.get("reference")))
				is_context_free = (no_ans >= max(1, int(0.8 * len(rows))))
			m: dict[str, str] = {}
			if is_context_free:
				# Build by-id question map
				by_id_q: dict[str, str] = {}
				for r in rows:
					try:
						_i = str(r.get("id"))
						_q = str(r.get("question"))
						if _i and _q:
							by_id_q[_i] = _q
					except Exception:
						pass
				qa_map["by_id"] = by_id_q
				# If GT by-id is available, join now to create both QA reference and GT ground_truths keyed by question
				gt_by_id = gt_map.get("by_id") or {}
				if isinstance(gt_by_id, dict) and gt_by_id:
					gt_q_map: dict[str, list[str]] = {}
					for _id, _q in by_id_q.items():
						gts = gt_by_id.get(str(_id))
						if not _q or not gts:
							continue
						# Prefer the first value as canonical reference; keep all for ground_truths
						m[str(_q)] = str(gts[0])
						gt_q_map[str(_q)] = [str(x) for x in gts]
					# Update GT maps as well to ensure metrics can find ground_truths by question text
					if gt_q_map:
						gt_map["map"] = gt_q_map
						gt_map["norm"] = { _norm_q(k): v for k, v in gt_q_map.items() }
						gt_map["__loaded__"] = True
				else:
					# No GT yet — store by-id only; references will be filled once GT loads
					m = {}
			else:
				# Standard QA with inline answers
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
			# If QA was context_free but GT wasn't loaded yet, indicate waiting state in message; else, show count
			if is_context_free and not (gt_map.get("by_id") and qa_map.get("map")):
				return f"Loaded {len(qa_map.get('by_id') or {})} QA ids from {p} (waiting for GT to join answers)"
			return f"Loaded {len(m)} QA pairs from {p}"
		except Exception as e:
			return f"(failed to load QA: {e})"

	# Auto-load default ground truths and QA if files exist
	try:
		for _cand in [
			Path("gear_wear_ground_truth_context_free.json"),
			Path("data")/"gear_wear_ground_truth_context_free.json",
		]:
			if _cand.exists():
				msg = _load_gt_file(str(_cand))
				log.info("GT auto-load: %s", msg)
				break
	except Exception:
		pass
	try:
		for _cand in [
			Path("gear_wear_qa_context_free.jsonl"),
			Path("data")/"gear_wear_qa_context_free.jsonl",
		]:
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
		# section/number fallback if nothing after filtering
		sec = qa["filters"].get("section") if qa and qa.get("filters") else None
		# If user asks for sensors/thresholds, bias toward tables if empty
		try:
			intent = (qa or {}).get("intent") or {}
			q_l = (q or "").lower()
			wants_instrument = any(w in q_l for w in ("sensor", "sensors", "accelerometer", "tachometer", "threshold", "alert threshold", "limits")) or (intent.get("target_attr") == "sensors")
			if wants_instrument and not filtered:
				filtered = [d for d in cands if (d.metadata or {}).get("section") == "Table"]
		except Exception:
			pass
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
		# Optional cross-encoder reranker for better final ordering
		try:
			from app.reranker_ce import rerank as ce_rerank
			if ce_rerank is not None:
				top_docs = ce_rerank(q, top_docs, top_n=8)
		except Exception:
			pass
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
				def _fig_sort_key(d):
					md = d.metadata or {}
					fn = md.get("figure_number")
					fo = md.get("figure_order")
					pg = md.get("page")
					try:
						fnv = int(fn) if fn is not None and str(fn).strip().isdigit() else 10**9
					except Exception:
						fnv = 10**9
					try:
						fov = int(fo) if fo is not None and str(fo).strip().isdigit() else 10**9
					except Exception:
						fov = 10**9
					try:
						pgv = int(pg) if pg is not None and str(pg).strip().isdigit() else 10**9
					except Exception:
						pgv = 10**9
					an = str(md.get("anchor") or "")
					return (fnv, fov, pgv, an)
				top_docs = sorted(top_docs, key=_fig_sort_key)
		except Exception:
			pass
		r, rtrace = route_question_ex(q)
		# Optional orchestrator reasoning trace + default answer
		reasoning_trace = None
		ans_from_orch = None
		try:
			if run_orchestrator is not None and os.getenv("RAG_USE_ORCHESTRATOR", "1").lower() in ("1","true","yes"):
				reasoning_trace = run_orchestrator(q, docs, hybrid, llm, do_answer=True)
				try:
					ans_from_orch = reasoning_trace.get("answer") if isinstance(reasoning_trace, dict) else None
				except Exception:
					ans_from_orch = None
		except Exception:
			reasoning_trace = None
		# Special-case: list all figures — return a deterministic list from metadata
		try:
			if (qa.get("filters") or {}).get("section") == "Figure" and re.search(r"\b(list|all|show)\b.*\bfigures\b", q, re.I):
				# Gather all figure docs and sort by number/order/page
				all_figs = [d for d in docs if (d.metadata or {}).get("section") == "Figure"]
				def _fig_sort_key(d):
					md = d.metadata or {}
					fn = md.get("figure_number")
					fo = md.get("figure_order")
					pg = md.get("page")
					try:
						fnv = int(fn) if fn is not None and str(fn).strip().isdigit() else 10**9
					except Exception:
						fnv = 10**9
					try:
						fov = int(fo) if fo is not None and str(fo).strip().isdigit() else 10**9
					except Exception:
						fov = 10**9
					try:
						pgv = int(pg) if pg is not None and str(pg).strip().isdigit() else 10**9
					except Exception:
						pgv = 10**9
					an = str(md.get("anchor") or "")
					return (fnv, fov, pgv, an)
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
				# Compose outputs — keep debug panels wired
				_dbg_visible = bool(debug_toggle)
				sig = (rtrace.get('signals') or {}) if isinstance(rtrace, dict) else {}
				sig_txt = ", ".join([k for k, v in sig.items() if v])
				_dbg_router_md = f"**Route:** {r}  \n**Rules:** {', '.join(rtrace.get('matched', []))}  \n**Canonical:** {qa.get('canonical')}  \n**Signals:** {sig_txt}"
				_dbg_filters_json = {"filters": qa.get("filters"), "keywords": qa.get("keywords"), "canonical": qa.get("canonical")}
				_dbg_dense_md = "Dense (top≈10):\n\n" + _fmt_docs(dense_docs)
				_dbg_sparse_md = "Sparse (top≈10):\n\n" + _fmt_docs(sparse_docs)
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
					gr.update(value=reasoning_trace or {}, visible=_dbg_visible),
					None,
				)
		except Exception:
			pass
		# If orchestrator produced a route, prefer to display it
		try:
			if isinstance(reasoning_trace, dict) and reasoning_trace.get("route"):
				r = reasoning_trace.get("route") or r
		except Exception:
			pass
		router_info = _render_router_info(r, top_docs) + f" | Agent: {r} | Rules: {', '.join(rtrace.get('matched', []))}"
		trace = f"Keywords: {qa['keywords']} | Filters: {qa['filters']}"
		# Build structured debug outputs
		_dbg_visible = bool(debug_toggle)
		sig = (rtrace.get('signals') or {}) if isinstance(rtrace, dict) else {}
		sig_txt = ", ".join([k for k, v in sig.items() if v])
		_dbg_router_md = f"**Route:** {r}  \n**Rules:** {', '.join(rtrace.get('matched', []))}  \n**Canonical:** {qa.get('canonical')}  \n**Signals:** {sig_txt}"
		_dbg_filters_json = {"filters": qa.get("filters"), "keywords": qa.get("keywords"), "canonical": qa.get("canonical")}
		_dbg_dense_md = "Dense (top≈10):\n\n" + _fmt_docs(dense_docs)
		_dbg_sparse_md = "Sparse (top≈10):\n\n" + _fmt_docs(sparse_docs)
		_dbg_hybrid_md = "Hybrid candidates (pre-filter):\n\n" + _fmt_docs(cands)
		_dbg_top_df = _rows_to_df(_rows_from_docs(top_docs))
		ans_raw = ans_from_orch
		if r == "summary":
			if not ans_raw:
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
			if not ans_raw:
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
			if not ans_raw:
				ans_raw = answer_needle(llm, top_docs, q)
			out = f"{router_info}\n\n{ans_raw}\n\n(trace: {trace})"

		# Always compute metrics per query using GT/QA maps
		metrics_txt = ""
		compare_dict = {}
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
				"reference_excerpt": ref,
				"answer_excerpt": (ans_raw or ""),
				"missing_ref_tokens_in_answer": missing,
				"extra_answer_tokens_not_in_reference": extra,
			}
			# Add heuristic hint when LLM metrics are NaN
			vals = [str(m.get(k)) for k in ("faithfulness","answer_relevancy","context_precision","context_recall")]
			if all(v == 'nan' for v in vals):
				metrics_txt += "\n(note: metrics require OPENAI_API_KEY or GOOGLE_API_KEY for RAGAS)"
		except Exception as e:
			metrics_txt = f"(metrics failed: {e})"
			compare_dict = {}

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
				"reasoning_trace": reasoning_trace,
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
		# Visibility updates for debug section and children
		_acc_upd = gr.update(visible=_dbg_visible, open=False)
		_router_upd = gr.update(value=_dbg_router_md, visible=_dbg_visible)
		_filters_upd = gr.update(value=_dbg_filters_json, visible=_dbg_visible)
		_dense_upd = gr.update(value=_dbg_dense_md, visible=_dbg_visible)
		_sparse_upd = gr.update(value=_dbg_sparse_md, visible=_dbg_visible)
		_hybrid_upd = gr.update(value=_dbg_hybrid_md, visible=_dbg_visible)
		_topdocs_upd = gr.update(value=_dbg_top_df, visible=_dbg_visible)
		_compare_upd = gr.update(value=compare_dict, visible=_dbg_visible)
		_reason_upd = gr.update(value=reasoning_trace or {}, visible=_dbg_visible)
		# Update figure preview slot when available; leave None to avoid clearing external viewers
		fig_update = gr.update(value=fig_path) if 'fig_path' in locals() and fig_path else None
		return out, metrics_txt, _acc_upd, _router_upd, _filters_upd, _dense_upd, _sparse_upd, _hybrid_upd, _topdocs_upd, _compare_upd, _reason_upd, fig_update

	# Build a sleeker Blocks UI with tabs
	with gr.Blocks(title="Hybrid RAG – Failure Reports") as demo:
		gr.Markdown("## Hybrid RAG – Failure Reports\nRouter + Summary / Needle / Table QA")
		with gr.Tabs():
			with gr.Tab("Ask"):
				q = gr.Textbox(label="Question", placeholder="Ask about figures, tables, procedures, conclusions…")
				dbg = gr.Checkbox(label="Show retrieval debug", value=False)
				btn = gr.Button("Ask", variant="primary")
				ans = gr.Markdown()
				metrics = gr.Textbox(label="Metrics", lines=3)
				# Inline figure preview for the current answer
				fig_preview = gr.Image(label="Relevant figure", interactive=False, visible=True)
				with gr.Accordion("Debug (retrieval trace)", open=False, visible=False) as dbg_acc:
					dbg_router = gr.Markdown()
					dbg_filters = gr.JSON()
					dbg_dense = gr.Markdown()
					dbg_sparse = gr.Markdown()
					dbg_hybrid = gr.Markdown()
					dbg_topdocs = gr.Dataframe(interactive=False)
					dbg_compare = gr.JSON(label="Answer vs Reference (tokens)")
					dbg_reason = gr.JSON(label="Reasoning trace")
				btn.click(
					on_ask,
					inputs=[q, dbg],
					outputs=[ans, metrics, dbg_acc, dbg_router, dbg_filters, dbg_dense, dbg_sparse, dbg_hybrid, dbg_topdocs, dbg_compare, dbg_reason, fig_preview],
				)


			with gr.Tab("Figures"):
				# Build a gallery of extracted figures, sorted by figure_number/figure_order
				fig_docs = [d for d in docs if d.metadata.get("section") == "Figure" and d.metadata.get("image_path")]
				try:
					def _fig_sort_key(d):
						md = d.metadata or {}
						fn = md.get("figure_number")
						fo = md.get("figure_order")
						pg = md.get("page")
						try:
							fnv = int(fn) if fn is not None and str(fn).strip().isdigit() else 10**9
						except Exception:
							fnv = 10**9
						try:
							fov = int(fo) if fo is not None and str(fo).strip().isdigit() else 10**9
						except Exception:
							fov = 10**9
						try:
							pgv = int(pg) if pg is not None and str(pg).strip().isdigit() else 10**9
						except Exception:
							pgv = 10**9
						an = str(md.get("anchor") or "")
						return (fnv, fov, pgv, an)
					fig_docs = sorted(fig_docs, key=_fig_sort_key)
				except Exception:
					pass
				fig_paths = [str(Path(d.metadata.get("image_path"))) for d in fig_docs if d.metadata.get("image_path")]
				if fig_paths:
					# Recent Gradio versions preview by default; keep args minimal for compatibility
					gr.Gallery(value=fig_paths, label="Extracted Figures", columns=4, height=400)
				else:
					gr.Markdown("(No extracted figures. Enable RAG_EXTRACT_IMAGES=true and rerun.)")

			with gr.Tab("Agent"):
				gr.Markdown("### Agent trace (tools + observations)\nRuns retrieval via simple tools for visibility.")
				q2 = gr.Textbox(label="Question", placeholder="E.g., list all figures or show figure 3")
				run_btn = gr.Button("Run Agent")
				trace_json = gr.JSON(label="Trace")
				result_md = gr.Markdown()

				# Maintenance tools
				try:
					from app.agent_tools import tool_audit_and_fill_figures as _audit_figs, tool_plan as _plan
				except Exception:
					_audit_figs = None
					_plan = None

				def _run_agent(question: str):
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
						ans = ""
						if tool_list_figures and re.search(r"\b(list|all|show)\b.*\bfigures\b", question, re.I):
							figs = tool_list_figures(docs)
							steps.append({"action": "list_figures", "observation_count": len(figs)})
							ans = "\n".join([f"Figure {f.get('figure_number')}: {f.get('label')} [{f.get('file')} p{f.get('page')}]" for f in figs])
						return steps, (ans or "(agent run complete)")
					except Exception as e:
						return steps + [{"error": str(e)}], "(agent failed)"

				run_btn.click(_run_agent, inputs=[q2], outputs=[trace_json, result_md])

				gr.Markdown("---")
				gr.Markdown("#### Maintenance: audit and fill missing figure numbers/orders")
				audit_btn = gr.Button("Audit/Fix Figures (session-only)")
				audit_out = gr.JSON(label="Audit Summary")
				def _do_audit():
					if _audit_figs is None:
						return {"error": "tool not available"}
					return _audit_figs(docs)
				audit_btn.click(_do_audit, inputs=[], outputs=[audit_out])

				gr.Markdown("#### Planner: propose a plan to fix DB issues")
				obs = gr.Textbox(label="Observations (paste from db_snapshot.jsonl)")
				plan_btn = gr.Button("Generate Plan")
				plan_md = gr.Markdown()
				def _do_plan(observations: str):
					if _plan is None:
						return "(planner not available)"
					return _plan(observations, llm)
				plan_btn.click(_do_plan, inputs=[obs], outputs=[plan_md])

			with gr.Tab("Inspect"):
				gr.Markdown("### Top indexed docs (sample)")
				sample_docs = [d for d in docs[:12]]
				gr.Textbox(value=_fmt_docs(sample_docs, max_items=12), label="Sample Contexts", lines=15)

			with gr.Tab("Graph"):
				gr.Markdown("### Knowledge Graph (auto-built)")
				# If the Main built a graph, it will be at logs/graph.html
				_graph_view = gr.HTML(value="")
				_graph_status = gr.Markdown()
				src = gr.Dropdown(
					choices=[
						"Docs co-mention (default)",
						"Normalized graph.json",
						"Normalized chunks",
						"Neo4j (live)"
					],
					value="Docs co-mention (default)",
					label="Graph source"
				)
				btn_graph = gr.Button("Generate / Refresh Graph")
				gr.Markdown("#### Graph DB (Neo4j) — optional")
				# Prefill with a sample so clicks don't pass an empty string on some Gradio builds
				cypher = gr.Textbox(label="Cypher query", value="MATCH (n) RETURN n LIMIT 10", placeholder="Type a Cypher query…", lines=2)
				btn_cypher = gr.Button("Run Cypher")
				cypher_out = gr.JSON(label="Results")

				def _build_graph_from_normalized_json():
					from pathlib import Path as _P
					import json as _json
					import networkx as _nx
					p = _P("logs")/"normalized"/"graph.json"
					if not p.exists():
						raise RuntimeError("logs/normalized/graph.json not found")
					data = _json.loads(p.read_text(encoding="utf-8"))
					G2 = _nx.Graph()
					def _node_label(n):
						t = n.get("type")
						pid = str(n.get("id"))
						props = n.get("props") or {}
						if t == "Figure":
							return props.get("caption") or pid
						if t == "Table":
							return props.get("title") or pid
						if t == "Section":
							return props.get("title") or pid
						if t == "Event":
							return props.get("date") or pid
						return pid
					for n in (data.get("nodes") or []):
						nid = str(n.get("id"))
						G2.add_node(nid, type=n.get("type"), label=_node_label(n))
					for e in (data.get("edges") or []):
						u = str(e.get("from")); v = str(e.get("to"))
						if u and v:
							G2.add_edge(u, v, type=e.get("type"))
					return G2

				def _build_graph_from_normalized_chunks():
					import networkx as _nx
					p = "logs/normalized/chunks.jsonl"
					ndocs = load_normalized_docs(p)
					if not ndocs:
						raise RuntimeError("logs/normalized/chunks.jsonl not found or empty")
					G3 = _nx.Graph()
					for d in ndocs:
						md = d.metadata or {}
						cid = str(md.get("chunk_id") or md.get("anchor") or id(d))
						clabel = f"chunk:{cid}"
						G3.add_node(clabel, type="Chunk", label=clabel)
						# Page linkage
						file = md.get("file_name"); page = md.get("page")
						if file and page is not None:
							pid = f"{file}#p{page}"
							G3.add_node(pid, type="Page", label=pid)
							G3.add_edge(clabel, pid, type="ON_PAGE")
						# Table/Figure linkage
						if md.get("table_number"):
							tnode = f"tbl:{int(md['table_number'])}"
							G3.add_node(tnode, type="Table", label=md.get("table_label") or tnode)
							G3.add_edge(clabel, tnode, type="REFERS_TO")
						if md.get("figure_number"):
							fnode = f"fig:{int(md['figure_number'])}"
							G3.add_node(fnode, type="Figure", label=md.get("figure_label") or fnode)
							G3.add_edge(clabel, fnode, type="REFERS_TO")
					return G3

				def _build_graph_from_neo4j():
					import networkx as _nx
					rows = _run_cypher("MATCH (a)-[r]->(b) RETURN a, type(r) as t, b LIMIT 200")  # type: ignore
					G4 = _nx.Graph()
					def _nid(x):
						try:
							props = x.get("properties") or {}
							labels = x.get("labels") or []
							label = labels[0] if labels else "Node"
							# Prefer stable id keys
							for k in ("id", "name", "title", "date"):
								if k in props and props[k]:
									return f"{label}:{props[k]}"
							return f"{label}:{props}"  # fallback
						except Exception:
							return str(x)
					for r in rows or []:
						a = r.get("a") or r.get("A") or r.get("n")
						b = r.get("b") or r.get("B") or r.get("m")
						t = r.get("t") or "REL"
						if not a or not b:
							continue
						na = _nid(a); nb = _nid(b)
						G4.add_node(na, type="Neo4j")
						G4.add_node(nb, type="Neo4j")
						G4.add_edge(na, nb, type=str(t))
					return G4

				def _gen_graph(source_choice: str):
					try:
						if render_graph_html is None:
							return gr.update(value=""), "(graph module not available; install dependencies: networkx, pyvis)"
						# Select source
						if source_choice == "Docs co-mention (default)":
							if build_graph is None:
								return gr.update(value=""), "(build_graph not available)"
							G = build_graph(docs)
						elif source_choice == "Normalized graph.json":
							G = _build_graph_from_normalized_json()
						elif source_choice == "Normalized chunks":
							G = _build_graph_from_normalized_chunks()
						elif source_choice == "Neo4j (live)":
							G = _build_graph_from_neo4j()
						else:
							if build_graph is None:
								return gr.update(value=""), "(unknown source)"
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
				btn_graph.click(_gen_graph, inputs=[src], outputs=[_graph_view, _graph_status])

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

