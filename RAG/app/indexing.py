from typing import List, Any, Dict
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from RAG.app.config import settings


def to_documents(records: List[dict]) -> List[Document]:
	return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in records]


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


def _sanitize_docs(docs: List[Document]) -> List[Document]:
	"""Sanitize document metadata to avoid Chroma upsert errors on complex types."""
	sanitized = []
	for doc in docs:
		# Create a copy with sanitized metadata
		clean_metadata = {}
		if doc.metadata:
			for k, v in doc.metadata.items():
				# Convert complex types to strings
				if isinstance(v, (list, dict, tuple)):
					clean_metadata[k] = str(v)
				elif v is not None:
					clean_metadata[k] = v
		sanitized.append(Document(page_content=doc.page_content, metadata=clean_metadata))
	return sanitized


def build_dense_index(docs: List[Document], embedding_fn):
	"""Build a dense index; try Chroma first, fallback to DocArrayInMemorySearch.
	If env RAG_CHROMA_DIR is set, persist to that directory (and use optional RAG_CHROMA_COLLECTION).
	Otherwise, use the configured INDEX_DIR.
	"""
	# Lazy import to avoid OS-specific failures at import time
	try:
		from langchain_community.vectorstores import Chroma
		# Sanitize metadata to avoid Chroma upsert errors on complex types
		sdocs = _sanitize_docs(docs)
		persist_dir = os.getenv("RAG_CHROMA_DIR")
		collection = os.getenv("RAG_CHROMA_COLLECTION", None)
		
		# Use configured INDEX_DIR if no environment variable is set
		if not persist_dir:
			persist_dir = str(settings.INDEX_DIR)
			
		if persist_dir:
			Path(persist_dir).mkdir(parents=True, exist_ok=True)
			if collection:
				vs = Chroma.from_documents(documents=sdocs, embedding=embedding_fn, persist_directory=persist_dir, collection_name=collection)
			else:
				vs = Chroma.from_documents(documents=sdocs, embedding=embedding_fn, persist_directory=persist_dir)
			# Chroma 0.4.x automatically persists documents, no need for manual persist()
			return vs
		else:
			return Chroma.from_documents(documents=sdocs, embedding=embedding_fn)
	except Exception:
		# Fallback pure-Python: DocArray, then FAISS
		try:
			from langchain_community.vectorstores import DocArrayInMemorySearch

			return DocArrayInMemorySearch.from_documents(_sanitize_docs(docs), embedding=embedding_fn)
		except Exception as e:
			try:
				from langchain_community.vectorstores import FAISS

				return FAISS.from_documents(_sanitize_docs(docs), embedding_fn)
			except Exception as e2:
				raise RuntimeError(f"Failed to initialize vector store: {e2}")


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


def build_sparse_retriever(docs: List[Document], k: int = 10):
	bm25 = BM25Retriever.from_documents(docs)
	bm25.k = k
	return bm25

