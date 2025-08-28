import os
import re
from typing import Dict, List
from app.logger import get_logger

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from app.agents import simplify_question
# Optional Cross-Encoder reranker
try:
	from app.reranker_ce import rerank as ce_rerank  # type: ignore
except Exception:  # pragma: no cover
	ce_rerank = None  # type: ignore
from app.logger import trace_func
try:
	from app.query_intent import get_intent  # optional LLM router
except Exception:
	get_intent = None  # type: ignore


@trace_func
def query_analyzer(q: str) -> Dict:
	"""Extract keywords, case/client IDs, dates to build metadata filters.
	Also returns a 'canonical' simplified query from a rules-based pre-agent.
	"""
	filt: Dict[str, str] = {}
	simp = (get_intent(q) if get_intent is not None and (os.getenv("RAG_USE_LLM_ROUTER", "0").lower() in ("1","true","yes")) else simplify_question(q))
	# Safer patterns: require word boundaries; avoid matching 'id' inside 'did'
	# Accept 'case: XYZ' or 'case id: XYZ' or 'client: ABC' but not bare 'id'
	mcase = re.search(r"\bcase(?:\s*id)?\b[:\-\s]*([A-Za-z0-9_-]{3,})", q, re.I)
	if mcase:
		filt["case_id"] = mcase.group(1)
	mclient = re.search(r"\bclient(?:\s*id)?\b[:\-\s]*([A-Za-z0-9_-]{3,})", q, re.I)
	if mclient:
		filt["client_id"] = mclient.group(1)
	mdate = re.search(r"(20\d{2}-\d{2}-\d{2})", q)
	if mdate:
		filt["incident_date"] = mdate.group(1)
	ql = q.lower()
	# Section hints from simplifier or raw tokens
	if bool(simp.get("wants_table")) or "table" in ql:
		filt["section"] = "Table"
	elif bool(simp.get("wants_figure")) or any(w in ql for w in ("figure", "image", "fig ", "photo", "plot", "graph")):
		filt["section"] = "Figure"
	# Specific number hints
	if simp.get("table_number"):
		filt["table_number"] = str(simp.get("table_number"))
	if simp.get("figure_number"):
		filt["figure_number"] = str(simp.get("figure_number"))
	# Case id from simplifier (e.g., W26)
	if simp.get("case_id") and "case_id" not in filt:
		filt["case_id"] = str(simp.get("case_id"))
	return {
		"filters": filt,
		"keywords": re.findall(r"[A-Za-z0-9°%]+", q)[:10],
		"canonical": str(simp.get("canonical") or "").strip() or None,
		"intent": simp,  # expose full simplifier intent for downstream routing/augmentation
	}


@trace_func
def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
	if not filters:
		return docs
	def ok(meta: dict):
		for k, v in (filters or {}).items():
			if k == "section":
				sec = (meta.get("section") or meta.get("section_type"))
				# Treat TableCell mini-docs as part of Table for filtering purposes
				if v == "Table":
					if sec not in ("Table", "TableCell"):
						return False
				else:
					if sec != v:
						return False
			elif k == "figure_number":
				# Support int/str and fallback to label prefix
				mn = meta.get("figure_number")
				if str(mn) == str(v):
					continue
				label = str(meta.get("figure_label") or meta.get("caption") or "")
				import re as _re
				if not _re.match(rf"^\s*figure\s*{int(str(v))}\b", label, _re.I):
					return False
			elif k == "table_number":
				mn = meta.get("table_number")
				if str(mn) == str(v):
					continue
				label = str(meta.get("table_label") or "")
				import re as _re
				if not _re.match(rf"^\s*table\s*{int(str(v))}\b", label, _re.I):
					return False
			else:
				if meta.get(k) != v:
					return False
		return True
	out = [d for d in docs if ok(d.metadata)]
	try:
		if os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1", "true", "yes"):
			log = get_logger()
			log.debug("FILTER: %d -> %d using %s", len(docs), len(out), filters)
	except Exception:
		pass
	return out


@trace_func
def build_hybrid_retriever(dense_store, sparse_retriever, dense_k: int = 10):
	"""Create an ensemble retriever with tunable weights via env vars.
	Defaults favor sparse slightly for keyword-heavy tech PDFs.
	"""
	print("this is me on the hybrid retriver")
	dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
	try:
		sw = float(os.getenv("RAG_SPARSE_WEIGHT", "0.65"))
		dw = float(os.getenv("RAG_DENSE_WEIGHT", "0.35"))
		total = (sw + dw) or 1.0
		sw, dw = sw / total, dw / total
	except Exception:
		sw, dw = 0.6, 0.4
	return EnsembleRetriever(retrievers=[sparse_retriever, dense], weights=[sw, dw])


@trace_func
def lexical_overlap(a: str, b: str) -> float:
	A, B = set(a.lower().split()), set(b.lower().split())
	if not A or not B:
		return 0.0
	return len(A & B) / len(A | B)


@trace_func
def rerank_candidates(query: str, candidates: List[Document], top_n: int = 8) -> List[Document]:
	# If CE reranker is enabled and available, prefer it
	try:
		if os.getenv("RAG_USE_CE_RERANKER", "0").lower() in ("1", "true", "yes") and ce_rerank is not None:
			return ce_rerank(query, candidates, top_n=top_n)
	except Exception:
		pass
	ql = query.lower()
	kws = set(re.findall(r"[A-Za-z0-9°%]+", ql))
	# Section preference based on query intent
	sec_pref = None
	if "table" in ql:
		sec_pref = "Table"
	elif any(w in ql for w in ("figure", "image", "fig ", "plot", "graph", "photo")):
		sec_pref = "Figure"
	# Sensor/metric/threshold inventory tends to live in tables; bias accordingly
	if any(w in ql for w in ("sensor", "sensors", "accelerometer", "tachometer", "instrumentation", "threshold", "alert threshold", "limits")):
		sec_pref = "Table"

	# Detect month/day phrases to boost timeline/date matches
	_months = [
		"january","february","march","april","may","june","july","august","september","october","november","december"
	]
	month_in_q = None
	day_in_q: str | None = None
	for m in _months:
		if m in ql:
			month_in_q = m
			break
	# capture patterns like "June 13th" or "June 13"
	md = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?", ql)
	if md:
		month_in_q = md.group(1)
		day_in_q = md.group(2)
	# also support ISO-like dates
	iso_in_q = re.search(r"20\d{2}-\d{2}-\d{2}", ql)

	def _len_penalty(n: int, is_figure: bool) -> float:
		# Loosen penalty for short figure captions so multiple figures can surface
		if is_figure:
			if n < 80:
				return 0.97
			if n > 2000:
				return 0.92
			return 1.0
		if n < 120:
			return 0.93
		if n > 3000:
			return 0.9
		return 1.0

	def _extractor_bonus(md: dict) -> float:
		if md.get("section") != "Table":
			return 0.0
		ext = str(md.get("extractor", ""))
		if ext.startswith("pdfplumber"):
			return 0.08
		if ext.startswith("tabula"):
			return 0.05
		if ext.startswith("camelot"):
			return 0.03
		if ext.startswith("synth"):
			return 0.0
		return 0.0

	scored = []
	for d in candidates:
		md = d.metadata or {}
		base = lexical_overlap(query, d.page_content)
		meta = " ".join(map(str, md.values()))
		meta_boost = 0.2 * lexical_overlap(" ".join(kws), meta)
		# Extra boost for instrumentation/threshold queries matching table or metadata
		if any(w in ql for w in ("sensor", "sensors", "accelerometer", "tachometer", "instrumentation")):
			if (md.get("section") or md.get("section_type")) == "Table":
				meta_boost += 0.2
			if any(k in str(meta).lower() for k in ("sensor", "accelerometer", "tachometer")):
				meta_boost += 0.1
		sec_boost = 0.15 if (sec_pref and (md.get("section") or md.get("section_type")) == sec_pref) else 0.0
		src_boost = _extractor_bonus(md)

		# Date/timeline boost: if query mentions a month/day or ISO date and doc contains it
		date_boost = 0.0
		text_l = d.page_content.lower()
		try:
			if iso_in_q and iso_in_q.group(0) in text_l:
				date_boost += 0.25
			if month_in_q and month_in_q in text_l:
				date_boost += 0.18
				if day_in_q and re.search(rf"\b{month_in_q}\s+{day_in_q}(?:st|nd|rd|th)?\b", text_l):
					date_boost += 0.17
		except Exception:
			pass


		# Metadata token boost for dates (from attach_metadata)
		tokens_boost = 0.0
		try:
			months_md = [str(x).lower() for x in (md.get("month_tokens") or [])]
			days_md = [str(x) for x in (md.get("day_tokens") or [])]
			if month_in_q and months_md and month_in_q in months_md:
				tokens_boost += 0.12
			if day_in_q and days_md and day_in_q in days_md:
				tokens_boost += 0.12
		except Exception:
			pass

		# Bonus for explicit numbering for figures to reduce ambiguity
		number_bonus = 0.0
		try:
			if (md.get("section") == "Figure" or md.get("section_type") == "Figure") and md.get("figure_number"):
				number_bonus += 0.08
			if (md.get("section") == "Table" or md.get("section_type") == "Table") and md.get("table_number"):
				number_bonus += 0.05
			# Extra boost if the query asks for a specific Figure/Table number
			mf = re.search(r"\bfigure\s*(\d{1,3})\b", ql)
			if mf and str(md.get("figure_number")) == mf.group(1):
				number_bonus += 0.25
			mt = re.search(r"\btable\s*(\d{1,3})\b", ql)
			if mt and str(md.get("table_number")) == mt.group(1):
				number_bonus += 0.2
		except Exception:
			pass

		score = (base + meta_boost + sec_boost + src_boost + date_boost + tokens_boost + number_bonus) * _len_penalty(len(d.page_content), (md.get("section") == "Figure" or md.get("section_type") == "Figure"))
		scored.append((score, len(d.page_content), d))

	# Sort and dedupe by (file,page,section,anchor/path) and collapse near-duplicate figure captions
	scored.sort(key=lambda x: (-x[0], x[1]))
	seen = set()
	seen_fig_captions = set()
	unique: List[Document] = []
	for s, ln, d in scored:
		md = d.metadata or {}
		key = (
			md.get("file_name"),
			md.get("page"),
			md.get("section") or md.get("section_type"),
			md.get("anchor") or md.get("table_md_path") or md.get("table_csv_path") or md.get("image_path")
		)
		if key in seen:
			continue
		# Collapse duplicates by similar figure captions to surface distinct figures
		try:
			if (md.get("section") == "Figure" or md.get("section_type") == "Figure"):
				cap = (md.get("figure_label") or "").strip().lower()
				cap_sig = cap[:80]
				if cap_sig and cap_sig in seen_fig_captions:
					continue
				if cap_sig:
					seen_fig_captions.add(cap_sig)
		except Exception:
			pass
		seen.add(key)
		unique.append(d)
		if len(unique) >= top_n:
			break

	try:
		if os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1", "true", "yes"):
			log = get_logger()
			for i, d in enumerate(unique[:top_n], start=1):
				md = d.metadata or {}
				log.debug("RERANK[%d]: %s p%s %s", i, md.get("file_name"), md.get("page"), md.get("section"))
	except Exception:
		pass
	return unique[:top_n]

