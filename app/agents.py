from typing import List, Protocol, Tuple, Dict
from typing import Optional

from langchain.schema import Document

from app.prompts import (
	NEEDLE_PROMPT,
	NEEDLE_SYSTEM,
	SUMMARY_PROMPT,
	SUMMARY_SYSTEM,
	TABLE_PROMPT,
	TABLE_SYSTEM,
)
from app.logger import trace_func
try:
	import os
	from app.query_intent import get_intent  # optional LLM-based intent
except Exception:
	get_intent = None  # type: ignore

# Optional Neo4j helpers; guarded import
try:
	from app.graphdb import query_table_cells  # type: ignore
except Exception:
	query_table_cells = None  # type: ignore


class LLMCallable(Protocol):
	def __call__(self, prompt: str) -> str:  # noqa: D401
		...


@trace_func
def simplify_question(q: str) -> Dict:
	"""Return a very simple, mostly-binary intent and a canonical query string.
	No LLM, just regex/keywords. Keys:
	  - canonical: str
	  - wants_date, wants_table, wants_figure, wants_summary, wants_value, wants_exact: bool
	  - table_number, figure_number, case_id, target_attr, event: optional str
	"""
	import re
	ql = (q or "").lower()
	out: Dict[str, object] = {
		"canonical": "",
		"wants_date": False,
		"wants_table": False,
		"wants_figure": False,
		"wants_summary": False,
		"wants_value": False,
		"wants_exact": False,
		"table_number": None,
		"figure_number": None,
		"case_id": None,
		"target_attr": None,
		"event": None,
	}
	# Flags
	out["wants_exact"] = any(w in ql for w in ("exact", "precise"))
	out["wants_summary"] = any(w in ql for w in ("summary", "summarize", "overview", "conclusion", "overall"))
	# Treat metrics/inventory and ratio questions as table lookups
	_ratio_hit = bool(re.search(r"\b(transmission|transmition|gear)\s+ratio\b|\bz\s*driv(?:ing|en)\b|\bzdriv", ql))
	out["wants_table"] = ("table" in ql) or _ratio_hit or bool(re.search(r"\bwear depth\b|\brms\b|\bfme\b|\bcrest factor\b", ql))
	# Treat instrumentation/sensor inventory and threshold questions as table-style lookups
	if any(w in ql for w in ("sensor", "sensors", "accelerometer", "tachometer", "instrumentation", "threshold", "alert threshold", "limits")):
		out["wants_table"] = True
	out["wants_figure"] = any(w in ql for w in ("figure", "fig ", "image", "graph", "plot"))
	out["wants_date"] = any(w in ql for w in ("when", "date", "day")) or bool(re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", ql))
	out["wants_value"] = any(w in ql for w in ("what is", "value", "how much", "amount")) or out["wants_table"]

	# Extract specifics
	mt = re.search(r"\btable\s*(\d{1,2})\b", ql)
	if mt:
		out["table_number"] = mt.group(1)
	mf = re.search(r"\bfigure\s*(\d{1,2})\b", ql)
	if mf:
		out["figure_number"] = mf.group(1)
	mc = re.search(r"\b(w\d{1,3})\b", ql)
	if mc:
		out["case_id"] = mc.group(1).upper()

	# Attribute
	if "wear depth" in ql:
		out["target_attr"] = "wear depth"
	elif re.search(r"\brms\b", ql):
		out["target_attr"] = "rms"
	elif "crest factor" in ql:
		out["target_attr"] = "crest factor"
	elif re.search(r"\bfme\b", ql):
		out["target_attr"] = "fme"
	elif _ratio_hit or "gear ratio" in ql:
		out["target_attr"] = "transmission ratio"
	elif any(w in ql for w in ("sensor", "sensors", "accelerometer", "tachometer", "instrumentation")):
		out["target_attr"] = "sensors"

	# Event for date-style questions
	if any(w in ql for w in ("failure", "failed")):
		out["event"] = "failure date"
	elif any(w in ql for w in ("measurement", "measuerment", "measured")) and any(w in ql for w in ("start", "started", "begin")):
		out["event"] = "measurement start date"
	elif any(w in ql for w in ("initial wear", "onset of wear", "first wear")):
		out["event"] = "initial wear date"
	elif any(w in ql for w in ("healthy",)) and any(w in ql for w in ("through", "until")):
		out["event"] = "healthy through date"

	# Build canonical
	canonical = ""
	if out["wants_table"] and out.get("case_id"):
		attr = out.get("target_attr") or "value"
		canonical = f"table lookup: {attr} for case {out['case_id']}"
	elif out["wants_table"] and out.get("table_number"):
		attr = out.get("target_attr") or "value"
		canonical = f"table {out['table_number']} {attr}"
	elif out["wants_figure"] and out.get("figure_number"):
		canonical = f"figure {out['figure_number']}"
	elif out["wants_date"]:
		ev = out.get("event") or "date in timeline"
		if out["wants_exact"]:
			canonical = f"exact {ev}"
		else:
			canonical = ev
	elif out["wants_summary"]:
		canonical = "summary of report"
	else:
		# Strip filler words to keep it minimal
		qq = re.sub(r"\b(please|kindly|could you|can you|what is|find|tell me)\b", " ", ql)
		qq = re.sub(r"\s+", " ", qq).strip()
		canonical = qq[:120]
	out["canonical"] = canonical
	return out


@trace_func
def route_question_ex(q: str) -> Tuple[str, Dict]:
	"""Return (route, trace) where trace explains which rule fired.
	Routes: summary | table | needle
	"""
	ql = q.lower()
	simp = (get_intent(q) if get_intent is not None and (os.getenv("RAG_USE_LLM_ROUTER", "0").lower() in ("1","true","yes")) else simplify_question(q))
	trace: Dict = {"matched": [], "route": None, "simplified": simp}
	# Summary cues
	if simp.get("wants_summary"):
		trace["matched"].append("summary_keywords")
		trace["route"] = "summary"
		return "summary", trace
	# Date questions should be answered with an exact value (one line) -> route to needle
	if any(m in ql for m in ("timeline", "chronology", "when did", "what happened on", "what happend on")) or simp.get("wants_date"):
		trace["matched"].append("timeline_date")
		trace["route"] = "needle"
		return "needle", trace
	# Table/Figure cues
	if simp.get("wants_table") or any(w in ql for w in ("table", "chart", "value", "figure", "fig ", "image", "graph", "plot")):
		trace["matched"].append("table_figure_keywords")
		trace["route"] = "table"
		return "table", trace
	# Default
	trace["matched"].append("fallback_needle")
	trace["route"] = "needle"
	return "needle", trace


@trace_func
def route_question(q: str) -> str:
	# Backward-compatible wrapper
	r, _ = route_question_ex(q)
	return r


@trace_func
def render_context(docs: List[Document], max_chars: int = 8000) -> str:
	out, n = [], 0
	for d in docs:
		md = d.metadata or {}
		sec = md.get('section') or md.get('section_type') or 'Text'
		extra = ''
		# Surface numbering cues to help LLM cite correctly
		if sec == 'Figure' and md.get('figure_number'):
			extra = f" (Figure {md.get('figure_number')})"
		if sec == 'Table' and md.get('table_number'):
			lbl = md.get('table_label')
			if lbl and isinstance(lbl, str):
				extra = f" ({lbl})"
			else:
				extra = f" (Table {md.get('table_number')})"
		header = f"[{md.get('file_name')} p{md.get('page')} {sec}{extra}]"
		piece = f"{header}\n{d.page_content}".strip()
		n += len(piece)
		if n > max_chars:
			break
		out.append(piece)
	return "\n\n".join(out)


@trace_func
def answer_summary(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = SUMMARY_SYSTEM + "\n" + SUMMARY_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


@trace_func
def answer_needle(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = NEEDLE_SYSTEM + "\n" + NEEDLE_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


@trace_func
def answer_table(llm: LLMCallable, docs: List[Document], question: str) -> str:
	# First, deterministic KV scan for common attributes like transmission ratio, wear depth, sensors
	ql = (question or "").lower()
	want_ratio = any(w in ql for w in ("transmission ratio", "transmition ratio", "gear ratio", "z / z", "zdriv", "driving/driven"))
	if want_ratio:
		# Search for KV mini-docs emitted by expand_table_kv_docs
		for d in docs:
			md = d.metadata or {}
			if (md.get("section") == "TableCell" or md.get("section_type") == "TableCell"):
				k = str(md.get("kv_key") or "").lower()
				v = str(md.get("kv_value") or "").strip()
				if not k:
					continue
				# Various header forms observed in extracted tables
				if ("transmission ratio" in k) or ("gear" in k and "ratio" in k) or ("z" in k and ("driv" in k)):
					if v:
						return v
	# Fallback to LLM over table/figure contexts
	table_docs = [d for d in docs if d.metadata.get("section") in ("Table", "Figure", "TableCell")] or docs
	# Sort to push most table-like content first
	table_docs = table_docs + [d for d in docs if d not in table_docs]
	ctx = render_context(table_docs)
	prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
	return llm(prompt).strip()


@trace_func
def answer_graph(llm: LLMCallable, docs: List[Document], question: str) -> str:
	"""Graph RAG agent (minimal):
	- If question mentions a table number and a key, try Neo4j TableCell lookup.
	- Else, fall back to needle over provided contexts.
	"""
	import re

	q = (question or "").strip()
	m = re.search(r"table\s*(\d+)", q, re.I)
	# Rough key heuristic: words after 'for' or 'of' or quoted phrase
	key: Optional[str] = None
	km = re.search(r"(?:for|of)\s+([A-Za-z0-9_ /%-]{3,})", q, re.I)
	qm = re.search(r"['\"]([^'\"]{3,})['\"]", q)
	if km:
		key = km.group(1).strip()
	elif qm:
		key = qm.group(1).strip()
	if m and key and query_table_cells:
		try:
			tnum = int(m.group(1))
			rows = query_table_cells(tnum, key, doc_id=None, limit=3)  # type: ignore[misc]
			if rows:
				r = rows[0]
				val = str(r.get("value") or "").strip()
				unit = str(r.get("unit") or "").strip()
				cite = f"[{r.get('file')} p{r.get('page')}]"
				if val:
					return f"{val} {unit}".strip() + f" {cite}"
		except Exception:
			pass
	# Fallback to needle
	return answer_needle(llm, docs, question)

