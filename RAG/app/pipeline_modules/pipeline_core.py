#!/usr/bin/env python3
"""
Core Pipeline Module
===================

Main orchestration functions for the RAG pipeline.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Any, Dict

from RAG.app.config import settings
from RAG.app.Data_Management.indexing import build_dense_index, build_sparse_retriever, dump_chroma_snapshot
from RAG.app.retrieve import build_hybrid_retriever
from RAG.app.logger import get_logger

# Import our modular components
from RAG.app.pipeline_modules.pipeline_ingestion import (
    clean_run_outputs,
    discover_input_paths,
    ingest_documents,
    convert_records_to_documents
)
from RAG.app.Evaluation_Analysis.validate import validate_document_pages
from RAG.app.pipeline_modules.pipeline_utils import (
    get_embeddings,
    LLM,
    orchestrate_graph_building,
    import_normalized_graph_data,
    log_normalized_graph_summary
)
from RAG.app.Data_Management.indexing import dump_chroma_snapshot

# Import additional required modules
from RAG.app.loaders import load_many
from RAG.app.chunking import structure_chunks
from RAG.app.Data_Management.metadata import attach_metadata
from RAG.app.Data_Management.normalized_loader import load_normalized_docs
from RAG.app.Data_Management.indexing import to_documents
from RAG.app.Evaluation_Analysis.validate import validate_min_pages


def build_pipeline(paths: List[Path]) -> Tuple[List, Any, Dict[str, Any]]:
    """Build the complete RAG pipeline using modular ingestion."""
    log = get_logger()
    
    # Use the dedicated ingestion module instead of duplicating logic
    records = ingest_documents(paths)
    
    # Section histogram after metadata attachment
    sec_hist = {}
    for r in records:
        sec = (r.get("metadata", {}) or {}).get("section")
        sec_hist[sec] = sec_hist.get(sec, 0) + 1
    if sec_hist:
        log.debug("Section histogram: %s", sorted(sec_hist.items(), key=lambda x: (-x[1], str(x[0]))))
    
    # vectorization
    # Optional: prefer normalized chunks.jsonl if feature flag enabled
    use_normalized = os.getenv("RAG_USE_NORMALIZED", "0").lower() in ("1", "true", "yes")
    if use_normalized and (settings.LOGS_DIR / "normalized" / "chunks.jsonl").exists():
        docs = load_normalized_docs(settings.LOGS_DIR / "normalized" / "chunks.jsonl")
        log.info("Using normalized docs for indexing: %d", len(docs))
    else:
        docs = to_documents(records)
    
    # Write a quick DB snapshot for debugging
    try:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        snap_path = settings.LOGS_DIR / "db_snapshot.jsonl"
        with open(snap_path, "w", encoding="utf-8") as f:
            for d in docs:
                md = d.metadata or {}
                txt = d.page_content or ""
                sec = md.get("section") or md.get("section_type")
                # Build a stable, human-oriented preview
                preview_str = ""
                try:
                    lines = (txt or "").splitlines()
                    if sec == "Figure":
                        # Prefer normalized label (e.g., "Figure N: ...")
                        preview_str = md.get("figure_label") or ""
                        if not preview_str:
                            # Extract CAPTION line
                            cap = None
                            for i, ln in enumerate(lines):
                                if ln.strip().upper() == "CAPTION:" and i + 1 < len(lines):
                                    cap = lines[i + 1].strip()
                                    break
                            preview_str = cap or (lines[0].strip() if lines else "")
                    elif sec == "Table":
                        preview_str = md.get("table_label") or ""
                        if not preview_str:
                            table_no = md.get("table_number")
                            summ = None
                            for i, ln in enumerate(lines):
                                if ln.strip().upper() == "SUMMARY:" and i + 1 < len(lines):
                                    summ = lines[i + 1].strip()
                                    break
                            if summ:
                                preview_str = f"Table {table_no}: {summ}" if table_no is not None else summ
                            else:
                                preview_str = lines[0].strip() if lines else ""
                    else:
                        preview_str = (txt or "")[:200]
                except Exception:
                    preview_str = (txt or "")[:200]
                rec = {
                    "file": md.get("file_name"),
                    "page": md.get("page"),
                    "section": md.get("section"),
                    "anchor": md.get("anchor"),
                    # Deterministic IDs for traceability
                    "doc_id": md.get("doc_id"),
                    "chunk_id": md.get("chunk_id"),
                    "content_hash": md.get("content_hash"),
                    # Table metadata
                    "table_md_path": md.get("table_md_path"),
                    "table_csv_path": md.get("table_csv_path"),
                    "table_number": md.get("table_number"),
                    "table_label": md.get("table_label"),
                    "table_associated_text_preview": md.get("table_associated_text_preview"),
                    "table_associated_anchor": md.get("table_associated_anchor"),
                    # Figure metadata
                    "image_path": md.get("image_path"),
                    "figure_number": md.get("figure_number"),
                    "figure_label": md.get("figure_label"),
                    "figure_associated_text_preview": md.get("figure_associated_text_preview"),
                    "figure_associated_anchor": md.get("figure_associated_anchor"),
                    # Content preview
                    "preview": preview_str,
                    "content_length": len(txt or ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Failed to write DB snapshot: {e}")
    
    # indexing
    embeddings = get_embeddings()
    dense_index = build_dense_index(docs, embeddings)
    sparse_retriever = build_sparse_retriever(docs)
    hybrid = build_hybrid_retriever(dense_index, sparse_retriever)
    
    # Optional: build graph database
    try:
        G = orchestrate_graph_building(docs)
        if G:
            log.info("Built graph database with %d nodes", len(G.nodes()))
            # Import normalized graph data if available
            import_normalized_graph_data(G)
            log_normalized_graph_summary(G)
    except Exception as e:
        log.warning(f"Failed to build graph database: {e}")
        G = None
    
    # Optional: dump Chroma snapshot
    try:
        dump_chroma_snapshot(dense_index)
    except Exception as e:
        log.warning(f"Failed to dump Chroma snapshot: {e}")
    
    return docs, hybrid, {"graph": G}


# run_pipeline function removed - use RAGService.run_pipeline() instead


def get_pipeline_components():
    """Get the main pipeline components for testing or external use."""
    return {
        "embeddings": get_embeddings(),
        "llm": LLM(),
        "paths": discover_input_paths(),
    }
