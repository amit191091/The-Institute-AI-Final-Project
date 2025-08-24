import os
import re
from typing import Dict, List
from app.logger import get_logger
from app.config import settings

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever


def query_analyzer(q: str) -> Dict:
	"""Extract keywords, case/client IDs, dates to build metadata filters."""
	filt: Dict[str, str] = {}
	# Require at least MIN_CASE_ID_LENGTH chars for case/id to avoid filtering on trivial digits like "1"
	mcase = re.search(rf"(?:case|id)[:\-\s]*([A-Za-z0-9_-]{{{settings.MIN_CASE_ID_LENGTH},}})", q, re.I)
	if mcase:
		filt["case_id"] = mcase.group(1)
	mclient = re.search(r"(?:client)[:\-\s]*([A-Za-z0-9_-]+)", q, re.I)
	if mclient:
		filt["client_id"] = mclient.group(1)
	mdate = re.search(r"(20\d{2}-\d{2}-\d{2})", q)
	if mdate:
		filt["incident_date"] = mdate.group(1)
	ql = q.lower()
	if "table" in ql:
		filt["section"] = "Table"
	elif any(w in ql for w in ("figure", "image", "fig ", "photo", "plot", "graph")):
		filt["section"] = "Figure"
	
	# Special handling for wear depth questions - don't filter by case_id if asking about wear depth
	# because the table documents don't have case_id metadata, they contain all wear cases
	if "wear depth" in ql and "case_id" in filt:
		del filt["case_id"]
	
	# Force table inclusion for specific queries that should find table data
	if any(term in ql for term in ["wear depth", "module", "gear type", "transmission ratio", "lubricant", "teeth"]):
		filt["section"] = "Table"
	
	return {"filters": filt, "keywords": re.findall(r"[A-Za-z0-9°%]+", q)[:settings.MAX_KEYWORDS]}


def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
	if not filters:
		return docs
	def ok(meta):
		return all(meta.get(k) == v for k, v in filters.items())
	out = [d for d in docs if ok(d.metadata)]
	return out


def build_hybrid_retriever(dense_store, sparse_retriever, dense_k: int = 10):
	dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
	return EnsembleRetriever(retrievers=[sparse_retriever, dense], weights=settings.HYBRID_RETRIEVER_WEIGHTS)


def lexical_overlap(a: str, b: str) -> float:
	A, B = set(a.lower().split()), set(b.lower().split())
	if not A or not B:
		return 0.0
	return len(A & B) / len(A | B)


def rerank_candidates(query: str, candidates: List[Document], top_n: int = settings.DEFAULT_TOP_N) -> List[Document]:
	kws = set(re.findall(r"[A-Za-z0-9°%]+", query.lower()))
	# domain synonyms for overlap
	syn = {
		"gear": ["tooth","teeth","mesh","gmf","transmission","sprocket","pinion"],
		"ratio": ["gear","transmission","18/35","18:35","gear ratio","speed ratio"],
		"module": ["mod","m=","gear module","tooth module"],
		"lubricant": ["oil","15w/40","15w-40","lubrication","grease"],
		"rms": ["rootmean","amplitude","level","vibration level"],
		"wear": ["damage","degradation","erosion","abrasion","fatigue"],
		"depth": ["thickness","measurement","size","dimension"],
		"failure": ["breakdown","malfunction","defect","issue","problem"],
		"vibration": ["oscillation","shaking","tremor","frequency"],
		"teeth": ["tooth","gear teeth","sprocket teeth","pinion teeth"],
		"transmission": ["gearbox","drive","power transmission","speed reduction"],
		"measurement": ["reading","value","data","result","finding"],
		"parameter": ["setting","configuration","specification","property"],
		"analysis": ["examination","investigation","study","assessment"],
		"case": ["instance","example","scenario","situation","occurrence"]
	}
	expanded = set(kws)
	for k, alts in syn.items():
		if k in kws:
			expanded.update(alts)
	query_lex = " ".join(expanded)
	
	scored = []
	for d in candidates:
		# Enhanced scoring with multiple factors
		content_lower = d.page_content.lower()
		meta_text = " ".join(map(str, d.metadata.values())).lower()
		
		# Base lexical overlap
		base = lexical_overlap(query_lex, content_lower)
		
		# Enhanced metadata boost
		meta_boost = settings.METADATA_BOOST_FACTOR * lexical_overlap(" ".join(kws), meta_text)
		
		# Length penalty for very long documents (favor concise, relevant content)
		length_penalty = min(1.0, 500 / max(len(content_lower), 1))
		
		# Section type bonus
		section_bonus = 0.0
		if d.metadata.get("section") == "Table" and any(term in query.lower() for term in ["table", "data", "value", "measurement", "parameter"]):
			section_bonus = 0.1
		elif d.metadata.get("section") == "Figure" and any(term in query.lower() for term in ["figure", "image", "plot", "graph", "diagram"]):
			section_bonus = 0.1
		
		# Final score
		score = base + meta_boost + section_bonus
		score *= length_penalty
		
		scored.append((score, len(d.page_content), d))
	
	# Sort by score (descending), then by content length (ascending for concise results)
	scored.sort(key=lambda x: (-x[0], x[1]))
	return [d for _, __, d in scored[:top_n]]