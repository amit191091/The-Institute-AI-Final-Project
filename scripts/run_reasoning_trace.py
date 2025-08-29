from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import List

from langchain.schema import Document

from app.pipeline import _LLM, ask, _get_embeddings
from app.retrieve import build_hybrid_retriever
from app.indexing import build_sparse_retriever, build_dense_index


def load_docs_from_snapshot() -> List[Document]:
    logs = Path("logs")
    full = logs / "db_snapshot_full.jsonl"
    if not full.exists():
        print("[run_reasoning_trace] Missing logs/db_snapshot_full.jsonl")
        return []
    docs: List[Document] = []
    with open(full, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            txt = rec.get("text") or ""
            md = rec.get("metadata") or {}
            if not md:
                md = {
                    "file_name": rec.get("file"),
                    "page": rec.get("page"),
                    "section": rec.get("section"),
                    "anchor": rec.get("anchor"),
                }
            docs.append(Document(page_content=txt, metadata=md))
    return docs


def main():
    # Ensure orchestrator is enabled
    os.environ.setdefault("RAG_USE_ORCHESTRATOR", "1")
    # Build lightweight retriever stack from snapshot
    docs = load_docs_from_snapshot()
    emb = _get_embeddings()
    # Try to reuse persisted Chroma; else in-memory dense index
    dense_store = None
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
        persist_dir = os.getenv("RAG_CHROMA_DIR")
        collection = os.getenv("RAG_CHROMA_COLLECTION")
        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            dense_store = Chroma(embedding_function=emb, persist_directory=persist_dir, collection_name=collection) if collection else Chroma(embedding_function=emb, persist_directory=persist_dir)
    except Exception:
        dense_store = None
    if dense_store is None:
        dense_store = build_dense_index(docs, emb)
    sparse = build_sparse_retriever(docs)
    hybrid = build_hybrid_retriever(dense_store, sparse)
    llm = _LLM()
    # Question from argv or default sample
    question = " ".join(sys.argv[1:]).strip() or "List two hallmark spectral changes that defined the moderate wear stage."
    ans = ask(docs, hybrid, llm, question)
    print("\n[run_reasoning_trace] Answer preview:\n", (ans or "")[:400])
    # Read last queries.jsonl entry and print its reasoning_trace field
    qlog = Path("logs") / "queries.jsonl"
    if qlog.exists():
        try:
            last = None
            with open(qlog, "r", encoding="utf-8") as f:
                for ln in f:
                    last = ln
            if last:
                entry = json.loads(last)
                print("\n[run_reasoning_trace] route:", entry.get("route"))
                print("[run_reasoning_trace] reasoning_trace:")
                print(json.dumps(entry.get("reasoning_trace"), ensure_ascii=False, indent=2))
        except Exception as e:
            print("[run_reasoning_trace] Failed reading queries.jsonl:", e)


if __name__ == "__main__":
    main()
