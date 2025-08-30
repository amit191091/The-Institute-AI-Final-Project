#!/usr/bin/env python3
"""
Heading Detection Functions
==========================

Heading detection and section hierarchy functions extracted from app/chunking.py
Includes style-based detection, heuristic analysis, and section context management.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import re
from RAG.app.utils import slugify
from RAG.app.chunking_modules.chunking_config import (
    _get_chunking_setting, _get_chunking_int_setting, _get_chunking_list_setting
)


def _is_heading_by_style(md: Any) -> bool:
    """Best-effort: detect heading using metadata style hints (font size/bold)."""
    try:
        if md is None:
            return False
        font_size = None
        bold = False
        if isinstance(md, dict):
            font_size = md.get("font_size")
            bold = bool(md.get("bold"))
        else:
            font_size = getattr(md, "font_size", None)
            bold = bool(getattr(md, "bold", False))
        min_font = float(_get_chunking_int_setting("HEADING_MIN_LENGTH", 12))
        if isinstance(font_size, (int, float)) and float(font_size) >= min_font:
            return True
        if bold:
            return True
    except (ValueError, TypeError):
        return False
    return False


def _looks_like_heading(line: str) -> bool:
    """Heuristic: short-ish line, mostly Title Case, not ending with period, not a table/caption marker."""
    ln = (line or "").strip()
    if not ln:
        return False
    # Skip page headers like "1 | P a g e"
    if "| p a g e" in ln.lower() or "| page" in ln.lower():
        return False
    # Skip obvious captions
    low = ln.lower()
    if low.startswith("figure ") or low.startswith("fig.") or low.startswith("table "):
        return False
    # Length constraint from config
    min_length = _get_chunking_int_setting("HEADING_MIN_LENGTH", 3)
    max_length = _get_chunking_int_setting("HEADING_MAX_LENGTH", 100)
    if len(ln) < min_length or len(ln) > max_length:
        return False
    # Avoid lines that look like sentences
    if ln.endswith(('.', '!', '?', ';')):
        return False
    # Title case proportion
    words = [w for w in ln.split() if w.isalpha()]
    if not words:
        return False
    cap = sum(1 for w in words if w[0].isupper())
    if cap / max(1, len(words)) < 0.5:
        return False
    return True


def _heading_level(line: str) -> int:
    """Infer heading level based on numbering prefix or known keywords."""
    import re as _re
    ln = (line or "").strip()
    # Numeric like "1.", "2.1", "3.4.2": depth = dot count + 1 (max 3)
    m = _re.match(r"^\d+(?:\.\d+)*\b", ln)
    if m:
        parts = m.group(0).split('.')
        return min(3, len(parts))
    # Advanced heading keywords
    if _re.match(r"^(chapter|section)\s+\d+(?:\.\d+)*\b", ln, _re.I):
        return 1 if ln.lower().startswith("chapter") else 2
    if _re.match(r"^appendix\s+[a-z]", ln, _re.I):
        return 1
    # Known primary sections
    primary = {"introduction","executive summary","summary","system description","conclusion","recommendations"}
    if (ln or "").strip().lower() in primary:
        return 1
    # Fallback medium level
    return 2


def _detect_heading_in_text(raw: str, md: Any = None) -> Optional[Tuple[str, int]]:
    # Prefer style-based if metadata indicates heading
    try:
        if _is_heading_by_style(md):
            first_line = (raw or "").strip().splitlines()[0] if raw else None
            if first_line:
                return (first_line.strip(), _heading_level(first_line))
    except Exception as e:
        pass
    # Check first ~10 non-empty lines for a heading candidate
    lines = [l.strip() for l in (raw or "").splitlines()]
    seen = 0
    for l in lines:
        if not l.strip():
            continue
        seen += 1
        if _looks_like_heading(l):
            return (l.strip(), _heading_level(l))
        if seen >= 10:
            break
    return None


def _make_section_id(doc: str, title: str, counter: int) -> str:
    return f"{doc}#sec-{counter:03d}-{slugify(title)[:40]}"


def _update_section_context(new_title: str, new_level: int, section_stack: List[Dict[str, Any]], section_counter: int, doc_id: str) -> Tuple[Dict[str, Any], int]:
    # Pop deeper or equal levels
    while section_stack and section_stack[-1]["level"] >= new_level:
        section_stack.pop()
    section_counter += 1
    sec_id = _make_section_id(doc_id, new_title, section_counter)
    parent_id = section_stack[-1]["id"] if section_stack else None
    breadcrumbs = (section_stack[-1]["breadcrumbs"] + [new_title]) if section_stack else [new_title]
    ctx = {"id": sec_id, "title": new_title, "level": new_level, "parent_id": parent_id, "breadcrumbs": breadcrumbs}
    section_stack.append(ctx)
    return ctx, section_counter


def _current_section_context(section_stack: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return section_stack[-1] if section_stack else None
