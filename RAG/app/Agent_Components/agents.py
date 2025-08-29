from typing import List, Protocol, Tuple, Dict

from langchain.schema import Document

from RAG.app.Agent_Components.prompts import (
	NEEDLE_PROMPT,
	NEEDLE_SYSTEM,
	SUMMARY_PROMPT,
	SUMMARY_SYSTEM,
	TABLE_PROMPT,
	TABLE_SYSTEM,
)


class LLMCallable(Protocol):
	def __call__(self, prompt: str) -> str:  # noqa: D401
		...


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
	out["wants_table"] = ("table" in ql) or bool(re.search(r"\bwear depth\b|\brms\b|\bfme\b|\bcrest factor\b", ql))
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


def route_question_ex(q: str) -> Tuple[str, Dict]:
	"""Return (route, trace) where trace explains which rule fired.
	Routes: summary | table | needle
	"""
	ql = q.lower()
	simp = simplify_question(q)
	trace: Dict = {"matched": [], "route": None, "simplified": simp}
	# Summary cues
	if simp.get("wants_summary"):
		trace["matched"].append("summary_keywords")
		trace["route"] = "summary"
		return "summary", trace
	if any(m in ql for m in ("timeline", "chronology", "when did", "what happened on", "what happend on")) or simp.get("wants_date"):
		trace["matched"].append("timeline_date")
		trace["route"] = "summary"
		return "summary", trace
	# Table/Figure cues
	if simp.get("wants_table") or any(w in ql for w in ("table", "chart", "value", "figure", "fig ", "image", "graph", "plot")):
		trace["matched"].append("table_figure_keywords")
		trace["route"] = "table"
		return "table", trace
	# Default
	trace["matched"].append("fallback_needle")
	trace["route"] = "needle"
	return "needle", trace


def route_question(q: str) -> str:
	# Backward-compatible wrapper
	r, _ = route_question_ex(q)
	return r


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


def answer_summary(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = SUMMARY_SYSTEM + "\n" + SUMMARY_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


def answer_needle(llm: LLMCallable, docs: List[Document], question: str) -> str:
	# Check if this is a wear range query and use specialized retriever
	try:
		from RAG.app.retrieve_modules.retrieve_wear_range import is_wear_range_query, wear_range_retriever
		if is_wear_range_query(question):
			return wear_range_retriever(question, docs)
	except ImportError:
		pass  # Fall back to LLM if specialized retriever not available
	
	ctx = render_context(docs)
	prompt = NEEDLE_SYSTEM + "\n" + NEEDLE_PROMPT.format(context=ctx, question=question)
	result = llm(prompt).strip()
	
	# Try to parse JSON response and extract answer field
	try:
		import json
		parsed = json.loads(result)
		if isinstance(parsed, dict) and "answer" in parsed:
			return parsed["answer"]
	except (json.JSONDecodeError, KeyError):
		pass
	
	# Fallback to original response if JSON parsing fails
	return result


def answer_table(llm: LLMCallable, docs: List[Document], question: str) -> str:
	table_docs = [d for d in docs if d.metadata.get("section") in ("Table", "Figure")] or docs
	# Sort to push most table-like content first
	table_docs = table_docs + [d for d in docs if d not in table_docs]
	ctx = render_context(table_docs)
	prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
	result = llm(prompt).strip()
	
	# Try to parse JSON response and extract answer field
	try:
		import json
		parsed = json.loads(result)
		if isinstance(parsed, dict) and "answer" in parsed:
			return parsed["answer"]
	except (json.JSONDecodeError, KeyError):
		pass
	
	# Fallback to original response if JSON parsing fails
	return result


def analyze_source_requirement(q: str) -> Dict:
	"""Analyze question to determine which data source is most appropriate.
	
	Returns:
		Dict with:
		- source_type: "report" | "database" | "both"
		- confidence: float (0-1)
		- reasoning: str
	"""
	import re
	ql = (q or "").lower()
	
	# Keywords that indicate database queries
	database_keywords = [
		"database", "all cases", "statistics", "across", "multiple", "various",
		"different", "range", "distribution", "average", "mean", "median",
		"standard deviation", "variance", "trend", "pattern", "comparison",
		"between cases", "across cases", "all wear cases", "database analysis"
	]
	
	# Keywords that indicate report-specific queries
	report_keywords = [
		"this case", "this gearbox", "this failure", "the report", "the study",
		"specific case", "particular case", "investigated gearbox", "mg-5025a",
		"ins haifa", "this vessel", "the gearbox", "this analysis"
	]
	
	# Check for database intent
	database_score = sum(1 for keyword in database_keywords if keyword in ql)
	
	# Check for report intent  
	report_score = sum(1 for keyword in report_keywords if keyword in ql)
	
	# Check for specific wear case queries (W1, W2, etc.) - these should be from report
	wear_case_match = re.search(r"\bw\d{1,3}\b", ql)
	if wear_case_match:
		report_score += 2  # Strong indicator for report source
	
	# Determine source type
	if database_score > report_score:
		source_type = "database"
		confidence = min(0.9, database_score / 3)
		reasoning = f"Question contains {database_score} database-related keywords"
	elif report_score > database_score:
		source_type = "report"
		confidence = min(0.9, report_score / 3)
		reasoning = f"Question contains {report_score} report-specific keywords"
	else:
		source_type = "report"  # Default to report for ambiguous questions
		confidence = 0.5
		reasoning = "Ambiguous question, defaulting to main report"
	
	return {
		"source_type": source_type,
		"confidence": confidence,
		"reasoning": reasoning,
		"database_score": database_score,
		"report_score": report_score
	}

