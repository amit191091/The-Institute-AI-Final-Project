#!/usr/bin/env python3
"""LlamaParse integration for enhanced table extraction.

This module provides LlamaParse integration for enhanced table parsing when an API key is available.
Extracted from app/loaders.py to maintain compatibility and functionality.

Environment variables:
    RAG_USE_LLAMA_PARSE: Enable LlamaParse extraction (default: 0)
    LLAMA_CLOUD_API_KEY: API key for LlamaParse service

Functions:
    _try_llamaparse_tables: Extract tables via LlamaParse, returning Markdown tables as Table elements
"""

import os
import re
from pathlib import Path
from typing import List
from RAG.app.logger import get_logger
from RAG.app.loader_modules.loader_utils import _env_enabled
from RAG.app.loader_modules.element_types import Table

# Optional LlamaParse for enhanced parsing
try:
    from llama_parse import LlamaParse  # type: ignore
    try:
        # Some versions expose an enum for result types
        from llama_parse import ResultType  # type: ignore
    except Exception:
        ResultType = None  # type: ignore
except Exception:
    LlamaParse = None  # type: ignore
    ResultType = None  # type: ignore


def _try_llamaparse_tables(pdf_path: Path) -> List[Table]:
    """Extract tables via LlamaParse, returning Markdown tables as Table elements.

    Simple, robust behavior:
    - Requires LLAMA_CLOUD_API_KEY and `llama-parse` installed.
    - Requests Markdown output and extracts pipe-table blocks by structure:
      header-row (| ... |), separator (| --- | ...), then one or more data rows.
    - Deduplicates tables by a normalized header key to avoid repeats.
    """
    if not LlamaParse or not _env_enabled("RAG_USE_LLAMA_PARSE"):
        return []
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        # Be explicit so it's easy to fix.
        try:
            get_logger().warning("LlamaParse skipped: missing LLAMA_CLOUD_API_KEY")
        except Exception:
            pass
        return []

    log = get_logger()
    elements: List[Table] = []
    seen_header_keys: set[str] = set()

    try:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
        # Use string to avoid enum compatibility issues across versions
        parser = LlamaParse(api_key=api_key, result_type="markdown")  # type: ignore[arg-type]

        docs = parser.load_data(str(pdf_path))
        full_md = "\n\n".join([getattr(d, "text", str(d)) for d in (docs or []) if getattr(d, "text", None) or str(d)])
        lines = (full_md or "").splitlines()

        # Helper: normalized header key
        def _header_key(header_line: str) -> str:
            cols = [c.strip().lower() for c in header_line.strip().strip("|").split("|")]
            return "|".join(cols[:8])

        # Helper: inspect up to two preceding non-empty lines for a title
        title_pat = re.compile(r"^\s*(?:#{1,6}\s*)?(table\s*\d*[:\.]?\s*.*)$", re.IGNORECASE)
        def _extract_title(start_idx: int) -> tuple[str | None, int | None]:
            j = start_idx - 1
            seen = 0
            while j >= 0 and seen < 3:
                s = lines[j].strip()
                if s:
                    # Markdown heading as-is
                    if s.startswith("#"):
                        return s, _parse_table_number(s)
                    # "Table N: Title" pattern
                    m = title_pat.match(s)
                    if m:
                        raw = m.group(1).strip()
                        return f"### {raw}", _parse_table_number(raw)
                    seen += 1
                j -= 1
            return None, None

        def _parse_table_number(text: str | None) -> int | None:
            if not text:
                return None
            m = re.search(r"table\s*(\d+)", text, flags=re.IGNORECASE)
            return int(m.group(1)) if m else None

        i = 0
        table_idx = 0
        n = len(lines)
        while i < n:
            line = lines[i].rstrip()
            if not line.startswith("|"):
                i += 1
                continue
            # Look ahead for the separator line (with --- between pipes)
            j = i + 1
            # Skip blank lines between header and separator (rare but safe)
            while j < n and lines[j].strip() == "":
                j += 1
            if j >= n:
                break
            sep = lines[j].strip()
            if not (sep.startswith("|") and "---" in sep):
                # Not a table header; advance
                i += 1
                continue

            # Collect subsequent pipe-rows as the table body
            k = j + 1
            rows: List[str] = [lines[i], lines[j]]
            while k < n and lines[k].lstrip().startswith("|"):
                rows.append(lines[k])
                k += 1

            # Only accept if there's at least one data row
            if len(rows) >= 3:
                # Try to include a nearby title above the table
                title_line, parsed_num = _extract_title(i)
                full_rows = ([] if not title_line else [title_line, ""]) + rows
                md_tbl = "\n".join(full_rows).strip()
                key = _header_key(rows[0])
                if key not in seen_header_keys:
                    seen_header_keys.add(key)
                    meta = {
                        "extractor": "llamaparse",
                        "table_index": table_idx,
                    }
                    if title_line:
                        # Strip markdown hashes for the bare title value
                        bare = re.sub(r"^\s*#{1,6}\s*", "", title_line).strip()
                        meta["table_title"] = bare
                        if parsed_num is not None:
                            meta["table_number"] = parsed_num
                    elements.append(
                        Table(
                            text=md_tbl,
                            metadata=meta,
                        )
                    )
                    table_idx += 1
            # Move to next block after current table
            i = k

        log.info("llamaparse: extracted %d tables", len(elements))
    except Exception as e:
        try:
            log.warning(f"LlamaParse extraction failed: {e}")
        except Exception:
            pass
    return elements
