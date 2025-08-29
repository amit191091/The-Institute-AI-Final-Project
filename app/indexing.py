from typing import List, Any, Dict
import os
from pathlib import Path

 

from langchain_core.documents import Document
from app.logger import trace_func
from langchain_community.retrievers import BM25Retriever


@trace_func
def to_documents(records: List[dict]) -> List[Document]:
	return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in records]

@trace_func
def _sanitize_metadata(md: Dict[str, Any] | None) -> Dict[str, Any]:
	"""Ensure metadata values are only str, int, float, bool, or None for vector stores like Chroma.
	Lists/tuples and dicts are JSON-serialized; non-serializable fallbacks become str().
	"""
	if not md:
		return {}
	out: Dict[str, Any] = {}
	for k, v in md.items():
		if v is None or isinstance(v, (str, int, float, bool)):
			out[k] = v
		elif isinstance(v, (list, tuple)):
			try:
				if all((x is None) or isinstance(x, (str, int, float, bool)) for x in v):
					out[k] = ", ".join("" if x is None else str(x) for x in v)
				else:
					import json as _json
					out[k] = _json.dumps(v, ensure_ascii=False)
			except Exception:
				out[k] = str(v)
		elif isinstance(v, dict):
			try:
				import json as _json
				out[k] = _json.dumps(v, ensure_ascii=False)
			except Exception:
				out[k] = str(v)
		else:
			out[k] = str(v)
	return out

@trace_func
def _sanitize_docs(docs: List[Document]) -> List[Document]:
	sanitized: List[Document] = []
	for d in docs:
		md = getattr(d, "metadata", None) or {}
		sanitized.append(Document(page_content=d.page_content, metadata=_sanitize_metadata(md)))
	return sanitized


@trace_func
def build_dense_index(docs: List[Document], embedding_fn):
	"""Build a dense index with an env-selectable backend.
	Backends by priority/order:
	  - RAG_VECTOR_BACKEND=faiss|docarray|chroma (default: chroma)
	  - If chroma selected and fails, fallback to docarray then faiss.
	If chroma is selected and RAG_CHROMA_DIR is set, persist to that directory (and optional RAG_CHROMA_COLLECTION).
	"""
	# Lazy import to avoid OS-specific failures at import time
	backend = (os.getenv("RAG_VECTOR_BACKEND", "chroma") or "chroma").strip().lower()
	sdocs = _sanitize_docs(docs)
	def _try_chroma():
		from langchain_community.vectorstores import Chroma
		persist_dir = os.getenv("RAG_CHROMA_DIR")
		collection = os.getenv("RAG_CHROMA_COLLECTION", None)
		if persist_dir:
			Path(persist_dir).mkdir(parents=True, exist_ok=True)
			vs = Chroma.from_documents(documents=sdocs, embedding=embedding_fn, persist_directory=persist_dir, collection_name=collection) if collection else Chroma.from_documents(documents=sdocs, embedding=embedding_fn, persist_directory=persist_dir)
			try:
				vs.persist()
			except Exception:
				pass
			return vs
		return Chroma.from_documents(documents=sdocs, embedding=embedding_fn)
	def _try_docarray():
		from langchain_community.vectorstores import DocArrayInMemorySearch
		return DocArrayInMemorySearch.from_documents(sdocs, embedding=embedding_fn)
	def _try_faiss():
		from langchain_community.vectorstores import FAISS
		return FAISS.from_documents(sdocs, embedding_fn)
	# explicit selection
	if backend == "faiss":
		try:
			return _try_faiss()
		except Exception as e2:
			# fallback chain
			try:
				return _try_docarray()
			except Exception:
				return _try_chroma()
	if backend == "docarray":
		try:
			return _try_docarray()
		except Exception:
			try:
				return _try_faiss()
			except Exception:
				return _try_chroma()
	# default chroma with graceful fallback
	try:
		return _try_chroma()
	except Exception:
		try:
			return _try_docarray()
		except Exception:
			return _try_faiss()

 

@trace_func
def dump_chroma_snapshot(vectorstore, out_path: Path) -> None:
	"""Try to dump a lightweight snapshot of a Chroma vector store: ids, metadatas, and preview of docs.
	Safe no-op if the vector store doesn't support .get().
	"""
	try:
		include = ["ids", "metadatas", "documents"]
		# Optional: include embeddings if explicitly requested
		if (os.getenv("RAG_CHROMA_DUMP_EMBEDDINGS", "0").lower() in ("1", "true", "yes")):
			include.append("embeddings")
		data = None
		# Try public API first
		try:
			data = vectorstore.get(include=include)  # type: ignore[attr-defined]
		except Exception:
			pass
		# Fallback to underlying collection
		if data is None:
			coll = getattr(vectorstore, "_collection", None)
			if coll is not None:
				try:
					data = coll.get(include=include)  # type: ignore[attr-defined]
				except Exception:
					pass
		if not data:
			return
		ids = data.get("ids") or []
		mets = data.get("metadatas") or []
		docs = data.get("documents") or []
		embs = data.get("embeddings") or []
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with open(out_path, "w", encoding="utf-8") as f:
			for i, vid in enumerate(ids):
				md = mets[i] if i < len(mets) else {}
				doc = docs[i] if i < len(docs) else ""
				emb = embs[i] if i < len(embs) else None
				rec = {
					"id": vid,
					"file": (md or {}).get("file_name"),
					"page": (md or {}).get("page"),
					"section": (md or {}).get("section"),
					"anchor": (md or {}).get("anchor"),
					"words": len((doc or "").split()),
					"preview": (doc or "")[:200],
				}
				if emb is not None:
					try:
						rec["embedding_dim"] = len(emb)
					except Exception:
						rec["embedding_dim"] = None
				import json as _json
				f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
	except Exception:
		return


@trace_func
def build_sparse_retriever(docs: List[Document], k: int = 10):
	bm25 = BM25Retriever.from_documents(docs)
	bm25.k = k
	print("im here on the sparse retriver trying to use bm25")
	return bm25


@trace_func
def expand_table_kv_docs(docs: List[Document]) -> List[Document]:
	"""Create additional Documents from table markdown by extracting key/value cells.
	Heuristics:
	  - Look for MARKDOWN block inside the table doc; parse pipe-rows until RAW or end.
	  - Identify columns whose header contains 'feature'/'parameter' as key, and 'value' as value.
	  - Fallback: use the first two non-empty columns.
	  - Emit one Document per non-empty key/value with section='TableCell' and kv_* metadata.
	"""
	out = list(docs)
	seen: set[tuple[str, int | None, str]] = set()
	for d in docs:
		md = d.metadata or {}
		if (md.get("section") or md.get("section_type")) != "Table":
			continue
		text = d.page_content or ""
		if "MARKDOWN:" not in text:
			continue
		try:
			md_part = text.split("MARKDOWN:", 1)[1]
			if "\nRAW:" in md_part:
				md_part = md_part.split("\nRAW:", 1)[0]
			lines = [ln.strip() for ln in md_part.splitlines() if ln.strip().startswith("|")]
			if len(lines) < 2:
				continue
			# Parse header
			header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
			# Build index map of non-empty header names
			header_names: list[tuple[int, str]] = []
			for i, h in enumerate(header_cells):
				hh = " ".join(h.split())
				if hh and hh.lower() != "---":
					header_names.append((i, hh))
			# Find key/value columns by name
			key_idx = None
			val_idx = None
			for i, h in header_names:
				hl = h.lower()
				if key_idx is None and ("feature" in hl or "parameter" in hl or "name" == hl):
					key_idx = i
				if val_idx is None and ("value" in hl):
					val_idx = i
			# Fallback to first two non-empty columns
			if key_idx is None or val_idx is None:
				non_empty = [i for i, h in header_names]
				if len(non_empty) >= 2:
					key_idx = key_idx if key_idx is not None else non_empty[0]
					# choose a different index for value
					for cand in non_empty:
						if cand != key_idx:
							val_idx = cand
							break
			if key_idx is None or val_idx is None:
				continue
			# Iterate data rows (skip header + separator)
			for row_line in lines[2:]:
				cells = [c.strip() for c in row_line.strip("|").split("|")]
				if max(key_idx, val_idx) >= len(cells):
					continue
				k = " ".join((cells[key_idx] or "").split())
				v = " ".join((cells[val_idx] or "").split())
				if not k or not v:
					continue
				# Emit a compact KV doc (dedupe by doc_id, table_number, kv_text)
				kv_text = f"{k}: {v}"
				key = (str(md.get("doc_id", "")), md.get("table_number"), kv_text)
				if key in seen:
					continue
				seen.add(key)
				kv_meta = _sanitize_metadata({
					**md,
					"section": "TableCell",
					"kv_key": k,
					"kv_value": v,
					"source_section": "Table",
				})
				out.append(Document(page_content=kv_text, metadata=kv_meta))
		except Exception:
			continue
	return out

