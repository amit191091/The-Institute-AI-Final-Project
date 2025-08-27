from typing import List, Protocol, Tuple, Dict

from langchain.schema import Document

from app.prompts import (
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
	# include common misspelling 'transmition'
	out["wants_table"] = ("table" in ql) or bool(re.search(r"\bwear depth\b|\brms\b|\bfme\b|\bcrest factor\b|\btransmission ratio\b|\btransmition ratio\b|\bgear ratio\b|\bzdriving\s*/\s*zdriven\b", ql))
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
	elif re.search(r"\b(transmission ratio|transmition ratio|gear ratio|zdriving/zdriven|z driving/z driven|z_driving/z_driven)\b", ql):
		out["target_attr"] = "transmission ratio"
		out["wants_table"] = True
	else:
		# Fuzzy fallback: any mention of ratio with transmission/gear stem implies table lookup
		if ("ratio" in ql) and ("transm" in ql or "gear" in ql or "zdriv" in ql):
			out["target_attr"] = "transmission ratio"
			out["wants_table"] = True

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
	if simp.get("wants_table") or any(w in ql for w in ("table", "chart", "value", "figure", "fig ", "image", "graph", "plot", "gear ratio", "transmission ratio", "transmition ratio", "zdriving/zdriven")):
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
	# Deterministic extraction for common numeric lookups before LLM
	val = _try_extract_transmission_ratio(docs)
	if val:
		cite = _first_table_citation(docs)
		return f"Transmission ratio: {val}{(' ' + cite) if cite else ''}"
	ctx = render_context(docs)
	prompt = NEEDLE_SYSTEM + "\n" + NEEDLE_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


def answer_table(llm: LLMCallable, docs: List[Document], question: str) -> str:
	table_docs = [d for d in docs if d.metadata.get("section") in ("Table", "Figure")] or docs
	# Sort to push most table-like content first
	table_docs = table_docs + [d for d in docs if d not in table_docs]
	# Deterministic extraction for common numeric lookups
	val = _try_extract_transmission_ratio(table_docs)
	if val:
		cite = _first_table_citation(table_docs)
		return f"Transmission ratio: {val}{(' ' + cite) if cite else ''}"
	ctx = render_context(table_docs)
	prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
	return llm(prompt).strip()


# --- Helpers ---------------------------------------------------------------

def _first_table_citation(docs: List[Document]) -> str:
	for d in docs:
		md = d.metadata or {}
		if (md.get("section") or md.get("section_type")) == "Table":
			fn = md.get("file_name")
			pg = md.get("page")
			tn = md.get("table_number")
			if tn:
				try:
					tn = int(tn)
				except Exception:
					pass
			label = md.get("table_label")
			if label:
				return f"[{fn} p{pg} {label}]"
			if tn:
				return f"[{fn} p{pg} Table {tn}]"
			return f"[{fn} p{pg} Table]"
	return ""


def _try_extract_transmission_ratio(docs: List[Document]) -> str | None:
	import re
	for d in docs:
		pc = d.page_content or ""
		txt = pc.lower()
		# 1) Direct regex search anywhere
		pats = [
			r"transmission\s+ratio[^\n\r\|]*\|\s*([0-9]+\s*/\s*[0-9]+)",
			r"transmission\s+ratio[^\n\r]*?([0-9]+\s*[:/]\s*[0-9]+)",
			r"gear\s+ratio[^\n\r]*?([0-9]+\s*[:/]\s*[0-9]+)",
			r"\bzdriving\s*/\s*zdriven\b[^\n\r]*?([0-9]+\s*/\s*[0-9]+)",
		]
		for p in pats:
			m = re.search(p, txt, re.I)
			if m:
				return m.group(1).replace(" ", "")
		# 2) Parse MARKDOWN block explicitly
		try:
			md_block = None
			m1 = re.search(r"\bMARKDOWN:\s*(.+?)\nRAW:\s*", pc, flags=re.S)
			if m1:
				md_block = m1.group(1)
			if not md_block:
				# fallback: use entire text to parse rows
				md_block = pc
			# Iterate markdown rows
			for line in md_block.splitlines():
				if '|' not in line:
					continue
				# normalize and split markdown row
				parts = [c.strip().lower() for c in line.strip().strip('|').split('|')]
				if not parts:
					continue
				# find label cell
				label = parts[0]
				if any(k in label for k in ("transmission ratio", "gear ratio", "zdriving", "zdriven")):
					# value is next non-empty cell
					for cell in parts[1:]:
						valm = re.search(r"([0-9]+\s*[:/]\s*[0-9]+)", cell)
						if valm:
							return valm.group(1).replace(" ", "")
			# 3) Load external table files if present in metadata (no need for upstream enrichment)
			try:
				md = getattr(d, 'metadata', {}) or {}
				paths = []
				if md.get('table_md_path'):
					paths.append(md.get('table_md_path'))
				if md.get('table_csv_path'):
					paths.append(md.get('table_csv_path'))
				for pth in paths:
					try:
						from pathlib import Path as _P
						p = _P(str(pth))
						if not p.exists():
							continue
						text = p.read_text(encoding='utf-8', errors='ignore').lower()
						# Try direct pattern first
						m = re.search(r"\b(transmission|gear)\s+ratio[^\n\r]*?([0-9]+\s*[:/]\s*[0-9]+)", text, re.I)
						if m:
							return m.group(2).replace(" ", "")
						# Otherwise scan markdown rows
						for line in text.splitlines():
							if '|' not in line:
								continue
							parts = [c.strip().lower() for c in line.strip().strip('|').split('|')]
							if not parts:
								continue
							if any(k in parts[0] for k in ("transmission ratio", "gear ratio", "zdriving", "zdriven")):
								for cell in parts[1:]:
									mv = re.search(r"([0-9]+\s*[:/]\s*[0-9]+)", cell)
									if mv:
										return mv.group(1).replace(" ", "")
					except Exception:
						continue
			except Exception:
				pass
		except Exception:
			pass
	return None

