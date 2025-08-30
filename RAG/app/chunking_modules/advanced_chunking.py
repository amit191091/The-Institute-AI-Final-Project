#!/usr/bin/env python3
"""
Advanced Chunking Functions
===========================

Advanced chunking capabilities extracted from app/chunking.py
Includes semantic processing, token management, table analysis, and chunk creation.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
from RAG.app.logger import get_logger
from RAG.app.config import settings
from RAG.app.utils import approx_token_len, truncate_to_tokens, sha1_short
from RAG.app.Data_Management.metadata import extract_keywords
from RAG.app.chunking_modules.chunking_config import (
    _get_chunking_setting, _get_chunking_int_setting, _get_chunking_float_setting,
    _get_chunking_string_setting
)

# Define token constants using centralized configuration
try:
    # Get settings from centralized config with environment variable overrides
    MAX_TOK = _get_chunking_int_setting("TEXT_MAX_TOK", 500)
    TEXT_TARGET_TOK = _get_chunking_int_setting("TEXT_TARGET_TOK", 375)
    TEXT_MAX_TOK = _get_chunking_int_setting("TEXT_MAX_TOK", 500)
    FIGURE_TABLE_MAX_TOK = _get_chunking_int_setting("FIGURE_TABLE_MAX_TOK", 800)
    SEMANTIC = _get_chunking_setting("USE_SEMANTIC_CHUNKING", True)
except Exception as e:
    # Fallback to safe defaults
    MAX_TOK = 500
    TEXT_TARGET_TOK, TEXT_MAX_TOK = 375, 500
    FIGURE_TABLE_MAX_TOK = 800
    SEMANTIC = True


def _get_max_tokens_for_type(section_type: str) -> int:
    """Return appropriate max tokens based on content type."""
    if section_type and section_type.lower() in ("table", "figure", "image"):
        return FIGURE_TABLE_MAX_TOK
    return TEXT_MAX_TOK


def _get_target_tokens_for_type(section_type: str) -> int:
    """Return appropriate target tokens based on content type."""
    if section_type and section_type.lower() in ("table", "figure", "image"):
        return int(FIGURE_TABLE_MAX_TOK * 0.6)  # ~480 for 800 max
    return TEXT_TARGET_TOK


def _analyze_table_markdown(md_text: str) -> str:
    """Structured analysis on markdown to extract basic numeric stats per column."""
    try:
        lines = [ln for ln in (md_text or "").splitlines() if ln.strip().startswith("|")]
        if len(lines) < 3:
            return ""
        hdr = [c.strip() for c in lines[0].strip("|").split("|")]
        data_rows = []
        for ln in lines[2:]:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) != len(hdr):
                continue
            data_rows.append(cells)
        stats = []
        for ci, name in enumerate(hdr):
            vals: List[float] = []
            for r in data_rows:
                try:
                    v = r[ci].replace(",", "")
                    if v.endswith("%"):
                        v = v[:-1]
                    vals.append(float(v))
                except (ValueError, IndexError):
                    continue
            if vals:
                stats.append((name or f"col{ci}", min(vals), max(vals)))
        if not stats:
            return ""
        lines_out = ["ANALYSIS:"] + [f"- {n}: min={vmin:g}, max={vmax:g}" for (n, vmin, vmax) in stats[:4]]
        return "\n".join(lines_out)
    except Exception as e:
        return ""


def _create_chunk_dict(content: str, anchor_local: str, doc_id: str, page: Optional[int], section_type: str, file_path: str, page_ord: Dict[int, int], section_stack: List[Dict[str, Any]]) -> Dict[str, Any]:
    content_c = truncate_to_tokens(content, MAX_TOK).strip()
    chunk_preview = (content_c or "").splitlines()[0][:200]
    content_hash = sha1_short(content_c)
    chunk_id = f"{doc_id}#p{page}:{section_type or 'Text'}/{anchor_local}"
    try:
        order_val = page_ord.get(int(page)) if page is not None else None
    except Exception as e:
        order_val = None
    _sec = _current_section_context(section_stack)
    return {
        "file_name": file_path,
        "page": page,
        "section_type": section_type or "Text",
        "section": section_type or "Text",
        "anchor": anchor_local,
        "order": order_val,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "section_id": _sec.get("id") if _sec else None,
        "section_title": _sec.get("title") if _sec else None,
        "section_level": _sec.get("level") if _sec else None,
        "section_parent_id": _sec.get("parent_id") if _sec else None,
        "section_breadcrumbs": _sec.get("breadcrumbs") if _sec else None,
        "content": content_c,
        "preview": chunk_preview,
        "keywords": extract_keywords(content_c),
    }


def _is_textual_chunk(ch: Dict[str, Any]) -> bool:
    """Identify textual vs. non-textual chunks."""
    sec = (ch.get("section") or ch.get("section_type") or "Text")
    return sec not in ("Table", "Figure")


def _semantic_groups(ss: List[str]) -> List[List[str]]:
    """Advanced semantic sentence grouping with sentence transformers."""
    if not SEMANTIC:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("RAG_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        st = SentenceTransformer(model_name)
        emb = st.encode(ss, normalize_embeddings=True, show_progress_bar=False)
        th = float(os.getenv("RAG_SEMANTIC_SIM_THRESHOLD", "0.52"))
        groups: List[List[str]] = []
        cur: List[str] = []
        cur_tok = 0
        for i in range(len(ss)):
            if not cur:
                cur = [ss[i]]
                cur_tok = approx_token_len(ss[i])
                continue
            sim = float(emb[i] @ emb[i-1])
            s_tok = approx_token_len(ss[i])
            if sim >= th and (cur_tok + s_tok) <= MAX_TOK:
                cur.append(ss[i])
                cur_tok += s_tok
            else:
                groups.append(cur)
                cur = [ss[i]]
                cur_tok = s_tok
            if cur_tok >= MAX_TOK:
                groups.append(cur)
                cur = []
                cur_tok = 0
        if cur:
            groups.append(cur)
        return groups
    except Exception as e:
        return []
