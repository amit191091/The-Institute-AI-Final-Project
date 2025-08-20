from typing import List

from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever


def to_documents(records: List[dict]) -> List[Document]:
	return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in records]


def build_dense_index(docs: List[Document], embedding_fn):
	"""Build a dense index; try Chroma first, fallback to DocArrayInMemorySearch."""
	# Lazy import to avoid OS-specific failures at import time
	try:
		from langchain_community.vectorstores import Chroma

		return Chroma.from_documents(documents=docs, embedding=embedding_fn)
	except Exception:
		# Fallback pure-Python: DocArray, then FAISS
		try:
			from langchain_community.vectorstores import DocArrayInMemorySearch

			return DocArrayInMemorySearch.from_documents(docs, embedding=embedding_fn)
		except Exception as e:
			try:
				from langchain_community.vectorstores import FAISS

				return FAISS.from_documents(docs, embedding_fn)
			except Exception as e2:
				raise RuntimeError(f"Failed to initialize vector store: {e2}")


def build_sparse_retriever(docs: List[Document], k: int = 10):
	bm25 = BM25Retriever.from_documents(docs)
	bm25.k = k
	return bm25

