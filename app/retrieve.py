import re
from typing import Dict, List

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever


def query_analyzer(q: str) -> Dict:
	"""Extract keywords, case/client IDs, dates to build metadata filters."""
	filt: Dict[str, str] = {}
	# Require at least 3 chars for case/id to avoid filtering on trivial digits like "1"
	mcase = re.search(r"(?:case|id)[:\-\s]*([A-Za-z0-9_-]{3,})", q, re.I)
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
	return {"filters": filt, "keywords": re.findall(r"[A-Za-z0-9°%]+", q)[:10]}


def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
	if not filters:
		return docs
	def ok(meta):
		return all(meta.get(k) == v for k, v in filters.items())
	return [d for d in docs if ok(d.metadata)]


def build_hybrid_retriever(dense_store, sparse_retriever, dense_k: int = 10):
	dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
	return EnsembleRetriever(retrievers=[sparse_retriever, dense], weights=[0.5, 0.5])


def lexical_overlap(a: str, b: str) -> float:
	A, B = set(a.lower().split()), set(b.lower().split())
	if not A or not B:
		return 0.0
	return len(A & B) / len(A | B)


def rerank_candidates(query: str, candidates: List[Document], top_n: int = 8) -> List[Document]:
	kws = set(re.findall(r"[A-Za-z0-9°%]+", query.lower()))
	scored = []
	for d in candidates:
		base = lexical_overlap(query, d.page_content)
		meta = " ".join(map(str, d.metadata.values()))
		boost = 0.2 * lexical_overlap(" ".join(kws), meta)
		score = base + boost
		scored.append((score, len(d.page_content), d))
	scored.sort(key=lambda x: (-x[0], x[1]))
	return [d for _, __, d in scored[:top_n]]

