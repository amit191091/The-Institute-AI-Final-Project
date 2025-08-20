from __future__ import annotations

import os
from pathlib import Path
from typing import List
import json
from datetime import datetime

from dotenv import load_dotenv

from app.config import settings
from app.loaders import load_many
from app.chunking import structure_chunks
from app.metadata import attach_metadata
from app.indexing import build_dense_index, build_sparse_retriever, to_documents
from app.retrieve import apply_filters, build_hybrid_retriever, query_analyzer, rerank_candidates
from app.agents import answer_needle, answer_summary, answer_table, route_question
from app.ui_gradio import build_ui
from app.validate import validate_min_pages
from app.logger import get_logger
from app.retrieve import lexical_overlap


def _discover_input_paths() -> List[Path]:
    """Collect input files: root Gear wear Failure.pdf and files under data/"""
    candidates: List[Path] = []
    root_pdf = Path("Gear wear Failure.pdf")
    if root_pdf.exists():
        candidates.append(root_pdf)
    if settings.DATA_DIR.exists():
        for ext in ("*.pdf", "*.docx", "*.doc", "*.txt"):
            candidates.extend(settings.DATA_DIR.glob(ext))
    return candidates


def _get_embeddings():
    """Prefer Google embeddings, fallback to OpenAI."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception:
        GoogleGenerativeAIEmbeddings = None  # type: ignore
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None  # type: ignore

    if os.getenv("GOOGLE_API_KEY") and GoogleGenerativeAIEmbeddings is not None:
        return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_GOOGLE)
    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_OPENAI)
    raise RuntimeError(
        "No embedding backend available. Set GOOGLE_API_KEY or OPENAI_API_KEY and install langchain-google-genai or langchain-openai."
    )


class _LLM:
    """Simple callable LLM wrapper preferring Gemini; fallback to OpenAI via LangChain chat models."""

    def __init__(self) -> None:
        self._backend = None
        self._which = None
        # Prefer Google Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._backend = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                self._which = "google"
            except Exception:
                self._backend = None
        # Fallback to OpenAI
        if self._backend is None and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI

                model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
                self._backend = ChatOpenAI(model=model, temperature=0.2)
                self._which = "openai"
            except Exception:
                self._backend = None

    def __call__(self, prompt: str) -> str:
        if self._backend is not None:
            try:
                resp = self._backend.invoke(prompt)
                # LangChain AIMessage has .content
                return getattr(resp, "content", str(resp))
            except Exception as e:  # pragma: no cover
                return f"[LLM error] {e}\n\n{prompt[-400:]}"
        # Final fallback echo
        return "[LLM not configured] " + prompt[-400:]


def build_pipeline(paths: List[Path]):
    log = get_logger()
    records = []
    # ingest
    for path, elements in load_many(paths):
        # basic ingestion validation: min pages
        try:
            pages_list = [
                int(getattr(getattr(e, "metadata", None), "page_number", 0))
                for e in elements
                if getattr(getattr(e, "metadata", None), "page_number", None) is not None
            ]
            pages = sorted(set(pages_list))
            ok, msg = validate_min_pages(len(set(pages)), settings.MIN_PAGES)
            if not ok:
                print(f"[WARN] {path.name}: {msg}")
        except Exception:
            pass
        chunks = structure_chunks(elements, str(path))
        for ch in chunks:
            records.append(attach_metadata(ch, client_id=os.getenv("CLIENT_ID"), case_id=path.stem))
    # Section histogram after metadata attachment
    sec_hist = {}
    for r in records:
        sec = (r.get("metadata", {}) or {}).get("section")
        sec_hist[sec] = sec_hist.get(sec, 0) + 1
    if sec_hist:
        log.info("Section histogram: %s", sorted(sec_hist.items(), key=lambda x: (-x[1], str(x[0]))))
    # vectorization
    docs = to_documents(records)
    tbl_cnt = sum(1 for d in docs if (d.metadata or {}).get("section") == "Table")
    fig_cnt = sum(1 for d in docs if (d.metadata or {}).get("section") == "Figure")
    log.info(
        "Ingestion complete: %d files -> %d chunks -> %d documents (tables=%d, figures=%d)",
        len(paths),
        len(records),
        len(docs),
        tbl_cnt,
        fig_cnt,
    )
    embeddings = _get_embeddings()
    dense = build_dense_index(docs, embeddings)
    sparse = build_sparse_retriever(docs, k=settings.SPARSE_K)
    hybrid = build_hybrid_retriever(dense, sparse, dense_k=settings.DENSE_K)
    # expose per-retriever diagnostics
    try:
        dense_ret = dense.as_retriever(search_kwargs={"k": settings.DENSE_K})
    except Exception:
        dense_ret = None
    debug = {"dense": dense_ret, "sparse": sparse}
    return docs, hybrid, debug


def ask(docs, hybrid, llm: _LLM, question: str, ground_truth: str | None = None) -> str:
    log = get_logger()
    qa = query_analyzer(question)
    # Compatible retrieval call
    try:
        candidates = hybrid.get_relevant_documents(question)
    except Exception:
        candidates = hybrid.invoke(question)
    # enforce rerank pool size
    candidates = candidates[: settings.RERANK_TOP_K]
    filtered = apply_filters(candidates, qa["filters"])  # metadata filters
    # Fallback: if section filter applied but nothing left, pull from all docs with that section
    try:
        sec = qa["filters"].get("section")
    except Exception:
        sec = None
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
    top_docs = rerank_candidates(question, filtered, top_n=settings.CONTEXT_TOP_N)
    route = route_question(question)
    # Logging: route, filters, counts, scores
    def _doc_head(d):
        md = getattr(d, "metadata", {}) or {}
        return f"{md.get('file_name')} p{md.get('page')} {md.get('section')}#{md.get('anchor', '')}"

    def _score(d):
        base = lexical_overlap(question, d.page_content)
        meta_text = " ".join(map(str, (getattr(d, "metadata", {}) or {}).values()))
        boost = 0.2 * lexical_overlap(" ".join(qa["keywords"]), meta_text)
        return round(base + boost, 4)

    log.info(
        "Q: %s | route=%s | keywords=%s | filters=%s | pool=%d | filtered=%d | using=%d",
        question,
        route,
        qa["keywords"],
        qa["filters"],
        len(candidates),
        len(filtered),
        len(top_docs),
    )
    for i, d in enumerate(top_docs, start=1):
        log.info("ctx[%d] score=%.4f | %s", i, _score(d), _doc_head(d))
    # Route to sub-agent
    if route == "summary":
        ans = answer_summary(llm, top_docs, question)
    elif route == "table":
        ans = answer_table(llm, top_docs, question)
    else:
        ans = answer_needle(llm, top_docs, question)
    # Append JSONL trace (lightweight)
    try:
        log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "route": route,
            "keywords": qa["keywords"],
            "filters": qa["filters"],
            "contexts": [
                {
                    "file": d.metadata.get("file_name"),
                    "page": d.metadata.get("page"),
                    "section": d.metadata.get("section"),
                    "anchor": d.metadata.get("anchor"),
                    "score": _score(d),
                }
                for d in top_docs
            ],
            "answer_preview": (ans or "")[:400],
        }
        with open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return ans


def main() -> None:
    load_dotenv()
    paths = _discover_input_paths()
    if not paths:
        print("No input files found. Place PDFs/DOCs under data/ or the root PDF.")
        return
    docs, hybrid, debug = build_pipeline(paths)
    llm = _LLM()
    # Launch Gradio UI
    if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
        print("[HEADLESS] Ingestion complete. Skipping UI launch.")
        return
    else:
        try:
            ui = build_ui(docs, hybrid, llm, debug)
            share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
            ui.launch(share=share)
        except Exception as e:
            print(f"UI failed to launch: {e}")
            # fallback single query demo
            print(ask(docs, hybrid, llm, "Summarize the failure modes described."))


if __name__ == "__main__":
    main()
# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load .env file


# # def Full_pipeline():
# #     print("starting full pipeline")
# #     # Placeholder for the full pipeline logic
# #     #1.file extraction + Parsing+ chunking avg chunk size :250-500, 800 tokens for table\diagram
# #     #2.metadata generation - filename, pagenumber, chunk_summary, keywords, section_type clientID\CaseID etc..
# #     #3.indexing - tables to csv\markdown , tableid, pagenum, anchor saving + small text summarization of table, vector metadate to filter retrival etc
# #     #4.Hybrid retrieval
# #     #5.Multi document support
# #     #6.gradio QA agent
# #     print("pipeline ended")




# def main():
#     print("hello world bitches")
#     return

# if __name__ == "__main__":
#     main()