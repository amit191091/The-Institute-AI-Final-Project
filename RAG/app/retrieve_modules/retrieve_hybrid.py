from typing import List, Any
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever


def build_hybrid_retriever(dense_store, sparse_retriever, dense_k: int = 10):
    """Build a hybrid retriever combining dense and sparse retrieval."""
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": dense_k})
    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid


def lexical_overlap(a: str, b: str) -> float:
    """Calculate lexical overlap between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
