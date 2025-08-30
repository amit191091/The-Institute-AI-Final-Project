"""
Build alternative indexes using LlamaIndex exports and LlamaParse outputs.

This lets us compare retrieval quality vs. our native pipeline, while persisting to
separate Chroma databases for A/B testing.

Env toggles:
  - RAG_ENABLE_LLAMAINDEX=1      -> Build index from app.llamaindex_export artifacts
  - RAG_USE_LLAMAPARSE=1         -> Build index from LlamaParse parsed text (if available)
  - RAG_CHROMA_DIR_LLX           -> Persist dir for LlamaIndex Chroma (default: index/chroma_llamaindex)
  - RAG_CHROMA_DIR_LP            -> Persist dir for LlamaParse Chroma (default: index/chroma_llamaparse)
  - RAG_CHROMA_COLLECTION_LLX/LP -> Optional collection names
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from RAG.app.logger import get_logger, trace_func

logger = get_logger()


def _sanitize_metadata(md: Dict[str, Any] | None) -> Dict[str, Any]:
    if not md:
        return {}
    out: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            try:
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
def _build_chroma_at(docs: List[Document], embedding_fn, persist_dir: Path, collection: str | None = None):
    """Build a Chroma index at a specific directory without changing global settings."""
    from langchain_community.vectorstores import Chroma
    persist_dir.mkdir(parents=True, exist_ok=True)
    sdocs = [Document(page_content=d.page_content, metadata=_sanitize_metadata(d.metadata or {})) for d in docs]
    vs = Chroma.from_documents(documents=sdocs, embedding=embedding_fn, persist_directory=str(persist_dir), collection_name=collection) if collection else Chroma.from_documents(documents=sdocs, embedding=embedding_fn, persist_directory=str(persist_dir))
    try:
        vs.persist()
    except Exception:
        pass
    return vs


@trace_func
def build_alt_indexes(paths: list[Path], embeddings) -> Dict[str, Any]:
    """Build and persist alternate indexes (LlamaIndex exports and LlamaParse),
    returning a dict of retrievers for ad-hoc comparison. Safe no-op if dependencies missing.
    """
    alt: Dict[str, Dict[str, Any]] = {}

    # LlamaIndex exports -> index/chroma_llamaindex
    try:
        if os.getenv("RAG_ENABLE_LLAMAINDEX", "0").lower() in ("1", "true", "yes"):
            export_root = Path("data") / "elements" / "llamaindex"
            llx_docs = _docs_from_llamaindex_exports(export_root)
            if llx_docs:
                dir_llx = Path(os.getenv("RAG_CHROMA_DIR_LLX", "index/chroma_llamaindex"))
                coll_llx = os.getenv("RAG_CHROMA_COLLECTION_LLX", None)
                dense_llx = _build_chroma_at(llx_docs, embeddings, dir_llx, coll_llx)
                from RAG.app.Data_Management.indexing import build_sparse_retriever
                sparse_llx = build_sparse_retriever(llx_docs, k=10)
                alt["llamaindex"] = {
                    "docs": llx_docs,
                    "dense": dense_llx,
                    "sparse": sparse_llx,
                }
                try:
                    logger.info("[LLX] Built %d docs -> Chroma at %s", len(llx_docs), str(dir_llx))
                except Exception:
                    pass
    except Exception:
        pass

    # LlamaParse -> index/chroma_llamaparse
    try:
        if os.getenv("RAG_USE_LLAMAPARSE", "0").lower() in ("1", "true", "yes"):
            lp_docs = _docs_from_llamaparse(paths)
            if lp_docs:
                dir_lp = Path(os.getenv("RAG_CHROMA_DIR_LP", "index/chroma_llamaparse"))
                coll_lp = os.getenv("RAG_CHROMA_COLLECTION_LP", None)
                dense_lp = _build_chroma_at(lp_docs, embeddings, dir_lp, coll_lp)
                from RAG.app.Data_Management.indexing import build_sparse_retriever
                sparse_lp = build_sparse_retriever(lp_docs, k=10)
                alt["llamaparse"] = {
                    "docs": lp_docs,
                    "dense": dense_lp,
                    "sparse": sparse_lp,
                }
                try:
                    logger.info("[LP] Built %d docs -> Chroma at %s", len(lp_docs), str(dir_lp))
                except Exception:
                    pass
    except Exception:
        pass

    return alt

@trace_func
def _docs_from_llamaindex_exports(root: Path) -> List[Document]:
    """Read exported chunks.jsonl under data/elements/llamaindex/* and convert to Documents."""
    docs: List[Document] = []
    if not root.exists():
        return docs
    for stem_dir in root.glob("*/"):
        cj = stem_dir / "chunks.jsonl"
        if not cj.exists():
            continue
        with open(cj, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get("content") or ""
                md = {
                    "file_name": rec.get("file"),
                    "page": rec.get("page"),
                    "section": rec.get("type") or "Text",
                    "anchor": f"llx-p{rec.get('page')}-{i}",
                    "source": "llamaindex",
                }
                docs.append(Document(page_content=text, metadata=md))
    return docs

@trace_func
def _docs_from_llamaparse(paths: List[Path]) -> List[Document]:
    """Parse PDFs via LlamaParse into Documents.
    Requires llama-parse and LLAMA_CLOUD_API_KEY. Falls back to empty list if unavailable.
    """
    try:
        from llama_parse import LlamaParse, ParseResultType  # type: ignore
    except Exception:
        return []
    key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not key:
        return []
    parser = LlamaParse(api_key=key, result_type=ParseResultType.MD, auto_mode=True)
    out: List[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() != ".pdf":
                continue
            results = parser.load_data(str(p))  # returns list of Documents
            # Prepare validation output dirs
            stem_norm = p.stem.replace(" ", "-").lower()
            lp_root = Path("data") / "elements" / "llamaparse" / stem_norm
            (lp_root / "tables").mkdir(parents=True, exist_ok=True)
            (lp_root / "images").mkdir(parents=True, exist_ok=True)
            chunks_path = lp_root / "chunks.jsonl"
            cj = open(chunks_path, "w", encoding="utf-8")
            for i, d in enumerate(results, start=1):
                text = getattr(d, "text", "") or ""
                meta = getattr(d, "metadata", {}) or {}
                # Heuristic section classification: treat markdown tables as Table when detected
                section = "Table" if ("|" in text and "\n|" in text and "---" in text) else "Text"
                page_val = meta.get("page_label") or meta.get("page") if isinstance(meta, dict) else None
                md = {
                    "file_name": p.name,
                    "page": page_val,
                    "section": section,
                    "anchor": f"lp-{p.stem}-p{page_val}-{i}",
                    "source": "llamaparse",
                }
                out.append(Document(page_content=text, metadata=md))
                # Write chunk record for visibility
                try:
                    rec = {"file": p.name, "page": page_val, "type": section, "metadata": meta, "content": text}
                    cj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                # If looks like a markdown table block, save it as .md for inspection
                if section == "Table":
                    try:
                        (lp_root / "tables" / f"{stem_norm}-p{page_val}-table-{i}.md").write_text(text, encoding="utf-8")
                    except Exception:
                        pass
            try:
                cj.close()
            except Exception:
                pass
            # Mirror existing pipeline images for convenience (LlamaParse may not export images)
            try:
                img_dir = Path("data") / "images"
                if img_dir.exists():
                    for img in img_dir.glob("*.png"):
                        if img.name.lower().startswith(stem_norm):
                            import shutil as _sh
                            _sh.copy2(img, lp_root / "images" / img.name)
            except Exception:
                pass
        except Exception:
            continue
    return out


