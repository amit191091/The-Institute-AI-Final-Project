import os
import re
from typing import Dict, List
from app.logger import get_logger

from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from app.agents import simplify_question


def query_analyzer(q: str) -> Dict:
	"""Extract keywords, case/client IDs, dates to build metadata filters.
	Also returns a 'canonical' simplified query from a rules-based pre-agent.
	"""
	filt: Dict[str, str] = {}
	simp = simplify_question(q)
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
	# Case id from simplifier (e.g., W26)
	if simp.get("case_id") and "case_id" not in filt:
		filt["case_id"] = str(simp.get("case_id"))
	return {
		"filters": filt,
		"keywords": re.findall(r"[A-Za-z0-9ֲ°%]+", q)[:10],
		"canonical": str(simp.get("canonical") or "").strip() or None,
	}


def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
	if not filters:
		return docs
	def ok(meta):
		return all(meta.get(k) == v for k, v in filters.items())
	out = [d for d in docs if ok(d.metadata)]
	try:
		if os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1", "true", "yes"):
			log = get_logger()
			log.debug("FILTER: %d -> %d using %s", len(docs), len(out), filters)
	except Exception:
		pass
	return out


def build_hybrid_retriever(dense_store, sparse_retriever, dense_k: int = 10):
	"""Create an ensemble retriever with tunable weights via env vars.
	Defaults favor sparse slightly for keyword-heavy tech PDFs.
	"""
	dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
	try:
		sw = float(os.getenv("RAG_SPARSE_WEIGHT", "0.6"))
		dw = float(os.getenv("RAG_DENSE_WEIGHT", "0.4"))
		total = (sw + dw) or 1.0
		sw, dw = sw / total, dw / total
	except Exception:
		sw, dw = 0.6, 0.4
	return EnsembleRetriever(retrievers=[sparse_retriever, dense], weights=[sw, dw])


def lexical_overlap(a: str, b: str) -> float:
	A, B = set(a.lower().split()), set(b.lower().split())
	if not A or not B:
		return 0.0
	return len(A & B) / len(A | B)


def rerank_candidates(query: str, candidates: List[Document], top_n: int = 8) -> List[Document]:
	ql = query.lower()
	kws = set(re.findall(r"[A-Za-z0-9ֲ°%]+", ql))
	# Section preference based on query intent
	sec_pref = None
	if "table" in ql:
		sec_pref = "Table"
	elif any(w in ql for w in ("figure", "image", "fig ", "plot", "graph", "photo")):
		sec_pref = "Figure"

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

	def _len_penalty(n: int) -> float:
		# Mild penalty for very short or extremely long contexts
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
		sec_boost = 0.15 if (sec_pref and md.get("section") == sec_pref) else 0.0
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

		score = (base + meta_boost + sec_boost + src_boost + date_boost + tokens_boost) * _len_penalty(len(d.page_content))
		scored.append((score, len(d.page_content), d))

	# Sort and dedupe by (file,page,section,anchor/path)
	scored.sort(key=lambda x: (-x[0], x[1]))
	seen = set()
	unique: List[Document] = []
	for s, ln, d in scored:
		md = d.metadata or {}
		key = (
			md.get("file_name"),
			md.get("page"),
			md.get("section"),
			md.get("anchor") or md.get("table_md_path") or md.get("table_csv_path") or md.get("image_path")
		)
		if key in seen:
			continue
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