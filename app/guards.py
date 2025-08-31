import os
from collections import Counter
from typing import List, Tuple, Optional

from langchain.schema import Document

# Optional single-file scope used during focused evals
SCOPE_FILE = os.getenv("RAG_SCOPE_SINGLE_FILE", "").strip()


def filter_to_scope(docs: List[Document]) -> List[Document]:
    if not SCOPE_FILE:
        return docs
    out = []
    for d in docs:
        md = (getattr(d, "metadata", {}) or {})
        fn = md.get("file_name") or md.get("file_path") or ""
        if str(fn) == SCOPE_FILE:
            out.append(d)
    return out


def dominant_file(docs: List[Document]) -> Tuple[Optional[str], float]:
    names = [
        (getattr(d, "metadata", {}) or {}).get("file_name")
        for d in docs
        if (getattr(d, "metadata", {}) or {}).get("file_name")
    ]
    if not names:
        return None, 0.0
    c = Counter(names).most_common(1)[0]
    return c[0], c[1] / max(1, len(names))


def enforce_domain(top_docs: List[Document], min_share: float = 0.7) -> List[Document]:
    """Fail-closed domain gate.
    - If SCOPE_FILE is set, keep only docs from that file (may return empty).
    - Else, if a single file accounts for >= min_share of docs, keep only that file.
    - Else, return empty (mixed domain considered unsafe).
    """
    if not top_docs:
        return []
    if SCOPE_FILE:
        return filter_to_scope(top_docs)
    dom, share = dominant_file(top_docs)
    if dom and share >= min_share:
        return [d for d in top_docs if (getattr(d, "metadata", {}) or {}).get("file_name") == dom]
    return []
