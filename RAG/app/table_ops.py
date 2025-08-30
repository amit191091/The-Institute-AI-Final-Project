from __future__ import annotations

"""Deterministic table utilities used by agents/tools.

Functions:
- markdown_to_df(md_path) -> pandas.DataFrame
- filter_rows(df, constraints) -> pandas.DataFrame
- read_kv(df, keys) -> list[str]

Handles common variants for instrumentation keys like sensitivity (mV/g)
and sampling rate (kS/sec, Hz, kHz).
"""
from RAG.app.logger import trace_func

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import re

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Avoid importing external types for annotations to keep this module lightweight.
# Use Any for document-like objects.

@trace_func
def _split_markdown_row(line: str) -> List[str]:
    # Strip leading/trailing pipes and split; also trim cells
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts

@trace_func
def markdown_to_df(md_path: str | Path):
    """Parse a GitHub-flavored markdown table into a DataFrame.

    Best-effort: ignores non-table lines; supports single header row with a
    separator line like |---|---|.
    """
    if pd is None:
        raise RuntimeError("pandas is required for table_ops.markdown_to_df")
    p = Path(str(md_path))
    if not p.exists():
        raise FileNotFoundError(str(p))
    lines = p.read_text(encoding="utf-8").splitlines()
    # Detect table blocks by presence of pipe and a separator row following header
    rows: List[List[str]] = []
    header: List[str] | None = None
    for i, ln in enumerate(lines):
        if "|" not in ln:
            continue
        # Separator looks like | --- | --- |
        if i + 1 < len(lines) and re.search(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1]):
            header = _split_markdown_row(ln)
            # Advance past separator and start collecting data rows
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                rows.append(_split_markdown_row(lines[j]))
                j += 1
            break
    if header is None:
        # Fallback: try to infer from first row that has multiple pipes
        for ln in lines:
            if ln.count("|") >= 2:
                header = _split_markdown_row(ln)
                break
    if header is None:
        # Give a single-column DF with raw content when cannot parse
        return pd.DataFrame({"_raw": [p.read_text(encoding="utf-8")]})
    # Normalize row lengths
    ncol = len(header)
    norm_rows = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in rows]
    df = pd.DataFrame(norm_rows, columns=header)
    return df

@trace_func
def filter_rows(df, constraints: Dict[str, Any]):
    """Filter a DataFrame by simple constraints.

    Supported keys (best-effort):
    - contains: str or list[str] present anywhere in the row (case-insensitive)
    - column: specific column name to search; with 'value' as the search text
    - regex: pattern to match across the row
    """
    if df is None or getattr(df, "empty", True):
        return df
    try:
        mask = None
        contains = constraints.get("contains")
        column = constraints.get("column")
        value = constraints.get("value")
        regex = constraints.get("regex")
        if contains:
            terms: Sequence[str] = contains if isinstance(contains, (list, tuple)) else [str(contains)]
            def _row_has_terms(row):
                text = " ".join(map(lambda x: str(x).lower(), row.values.tolist()))
                return all(t.lower() in text for t in terms)
            mask = df.apply(_row_has_terms, axis=1)
        if column and value is not None and column in df.columns:
            col = df[column].astype(str).str.lower()
            m2 = col.str.contains(str(value).lower(), regex=False, na=False)
            mask = m2 if mask is None else (mask & m2)
        if regex:
            pat = re.compile(str(regex), re.I)
            def _row_regex(row):
                text = " ".join(map(lambda x: str(x), row.values.tolist()))
                return bool(pat.search(text))
            m3 = df.apply(_row_regex, axis=1)
            mask = m3 if mask is None else (mask & m3)
        return df[mask] if mask is not None else df
    except Exception:
        return df

@trace_func
def read_kv(df, keys: Iterable[str]) -> List[str]:
    """Read values from a table given likely key names.

    - Matches columns by case-insensitive equality or containment with provided keys.
    - Also scans cells for unit patterns (mV/g for sensitivity; Hz/kHz/kS/sec for rate).
    Returns a list of string values discovered.
    """
    out: List[str] = []
    if df is None or getattr(df, "empty", True):
        return out
    kl = [str(k).strip().lower() for k in keys]
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    # Direct column hits
    hit_cols: List[str] = []
    for k in kl:
        for lc, orig in cols_lower.items():
            if lc == k or (k in lc):
                hit_cols.append(orig)
    hit_cols = list(dict.fromkeys(hit_cols))  # dedupe preserve order
    # Sensitivity and sampling rate patterns
    sens_pat = re.compile(r"\b\d+(?:\.\d+)?\s*m\s*v\s*/\s*g\b", re.I)
    rate_pat = re.compile(r"\b\d+(?:\.\d+)?\s*(?:k?s/sec|hz|khz)\b", re.I)
    # Collect from hit columns first
    for c in hit_cols:
        try:
            vals = [str(v).strip() for v in df[c].tolist() if str(v).strip()]
            out.extend(vals)
        except Exception:
            pass
    # Scan all cells for patterns if keys suggest instrumentation
    if any("sensitivity" in k for k in kl) or any("mV/g" in k for k in kl):
        try:
            for _, row in df.iterrows():
                for v in row.values.tolist():
                    s = str(v)
                    if sens_pat.search(s):
                        out.append(s.strip())
        except Exception:
            pass
    if any("sample" in k and "rate" in k for k in kl) or any(x in kl for x in ("sampling rate", "rate", "hz", "khz", "ks/sec")):
        try:
            for _, row in df.iterrows():
                for v in row.values.tolist():
                    s = str(v)
                    if rate_pat.search(s):
                        out.append(s.strip())
        except Exception:
            pass
    # Final cleanup: uniquify, keep concise
    out_u = []
    seen = set()
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        out_u.append(v)
    return out_u[:10]


# --- Schema-agnostic table QA helpers ---
@trace_func
def _normalize_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s

@trace_func
def _is_kv_like(df) -> bool:
    try:
        if df is None or getattr(df, "empty", True):
            return False
        cols = [str(c).strip().lower() for c in df.columns]
        if len(cols) < 2:
            return False
        # Heuristic: first col looks like a key column
        keyish = ("feature" in cols[0]) or ("parameter" in cols[0]) or (cols[0] in ("name", "key", "attribute"))
        valish = any("value" in c for c in cols[1:])
        # Also treat any two-column table as kv-like if the first has short text entries
        if not (keyish or valish):
            if len(cols) == 2:
                try:
                    avg_len = float(pd.Series([len(str(x)) for x in df.iloc[:, 0]]).mean()) if pd is not None else 15.0
                except Exception:
                    avg_len = 15.0
                keyish = avg_len <= 30
        return bool(keyish or valish)
    except Exception:
        return False

@trace_func
def _tokenize_q(question: str) -> List[str]:
    t = re.sub(r"[^a-z0-9μ%/\.]+", " ", (question or "").lower())
    t = t.replace("μ", "u").replace("µ", "u")
    return [w for w in t.split() if w]

@trace_func
def _numeric_tokens(question: str) -> List[str]:
    ql = (question or "").lower().replace("μ", "u").replace("µ", "u")
    # Capture numbers and units (e.g., 400, 400um, 6.4, 6/13)
    toks = re.findall(r"\b\d+(?:\.\d+)?(?:\s*[a-z%/]+)?\b", ql)
    return toks

@trace_func
def _best_column_match(df, q_tokens: List[str]) -> Optional[int]:
    try:
        cols = [str(c) for c in df.columns]
        norm = [_normalize_name(c) for c in cols]
        best_i, best_s = None, 0.0
        for i, cname in enumerate(norm):
            score = 0
            for t in q_tokens:
                if t in cname:
                    score += 2
            # boost numeric-oriented columns if question asks for a value/how much
            if any(t in ("value", "amount", "depth", "ratio", "rate", "sensitivity") for t in cname.split()):
                score += 1
            if score > best_s:
                best_s, best_i = score, i
        return best_i
    except Exception:
        return None

@trace_func
def _row_overlap_score(values: List[str], q_tokens: List[str]) -> float:
    v = _normalize_name(" ".join(values))
    return sum(1 for t in q_tokens if t in v) / max(1, len(set(q_tokens)))

@trace_func
def _extract_tables_from_doc_text(text: str):
    """Yield markdown-like blocks from a table doc's text after 'MARKDOWN:' prefix."""
    if not text:
        return []
    out = []
    try:
        if "MARKDOWN:" in text:
            md_part = text.split("MARKDOWN:", 1)[1]
            if "\nRAW:" in md_part:
                md_part = md_part.split("\nRAW:", 1)[0]
            # Capture only pipe rows
            lines = [ln for ln in md_part.splitlines() if "|" in ln]
            if lines:
                out.append("\n".join(lines))
    except Exception:
        return []
    return out


@trace_func
def _markdown_block_to_df(md_block: str):
    if pd is None:
        return None
    try:
        # Write to a temporary-like string path via pandas.read_csv with '|' sep is error-prone;
        # reuse our own parser: mimic markdown_to_df but for in-memory block.
        lines = md_block.splitlines()
        rows: List[List[str]] = []
        header: List[str] | None = None
        for i, ln in enumerate(lines):
            if "|" not in ln:
                continue
            if i + 1 < len(lines) and re.search(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1]):
                header = _split_markdown_row(ln)
                j = i + 2
                while j < len(lines) and "|" in lines[j]:
                    rows.append(_split_markdown_row(lines[j]))
                    j += 1
                break
        if header is None:
            # fallback to first row
            for ln in lines:
                if ln.count("|") >= 2:
                    header = _split_markdown_row(ln)
                    break
        if header is None:
            return None
        ncol = len(header)
        norm_rows = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in rows]
        return pd.DataFrame(norm_rows, columns=header)
    except Exception:
        return None


@trace_func
def build_tables_from_docs(docs: List[Any]) -> List[Tuple[Any, Any]]:
    """Extract DataFrames from table/kv docs along with their source Document for citation."""
    if pd is None:
        return []
    out: List[Tuple[Any, Any]] = []
    for d in docs or []:
        try:
            md = d.metadata or {}
            sec = md.get("section") or md.get("section_type")
            # TableCell mini-docs: create a tiny 2-col DF
            if sec == "TableCell":
                k = str(md.get("kv_key") or "").strip()
                v = str(md.get("kv_value") or "").strip()
                if k and v:
                    out.append((pd.DataFrame({"Key": [k], "Value": [v]}), d))
                    continue
            # Proper Table docs: parse markdown block
            if sec == "Table":
                text = d.page_content or ""
                blocks = _extract_tables_from_doc_text(text)
                for blk in blocks:
                    df = _markdown_block_to_df(blk)
                    if df is not None and not df.empty:
                        out.append((df, d))
        except Exception:
            continue
    return out


@trace_func
def natural_table_lookup(question: str, docs: List[Any]) -> Tuple[Optional[str], Optional[Any]]:
    """General, schema-agnostic lookup:
    - Builds DataFrames from docs.
    - If KV-like: fuzzy match key from question and return paired value.
    - Else (matrix): pick best row by overlap with Q tokens and best column by header similarity; return that cell.
    - If numeric token present in Q, try reverse lookup: find cell containing that numeric and return sibling cell from an ID-like column.
    Returns (answer_text, source_doc) or (None, None).
    """
    if pd is None:
        return None, None
    tables = build_tables_from_docs(docs)
    if not tables:
        return None, None
    q_tokens = _tokenize_q(question)
    q_norm = " ".join(q_tokens)
    q_nums = _numeric_tokens(question)

    # Pass 1: KV-like tables
    for df, src in tables:
        try:
            if _is_kv_like(df):
                # Pick key/value columns
                cols = list(df.columns)
                key_col = 0
                val_col = 1 if len(cols) > 1 else None
                # Prefer named columns when present
                for i, c in enumerate(cols):
                    cl = str(c).lower()
                    if i == 0 and any(x in cl for x in ("feature", "parameter", "name", "key", "attribute")):
                        key_col = i
                    if val_col is None or ("value" in cl):
                        val_col = i if ("value" in cl) else val_col
                # Fallback: first two columns
                if val_col is None:
                    val_col = 1 if len(cols) > 1 else 0
                # Score keys by token overlap
                best_idx, best_score = None, 0.0
                for i, row in df.iterrows():
                    key_text = _normalize_name(str(row.iloc[key_col]))
                    sc = sum(1 for t in q_tokens if t in key_text)
                    # light unit/number match boost
                    if q_nums and any(n in str(row.values) for n in q_nums):
                        sc += 1
                    if sc > best_score:
                        best_score, best_idx = sc, i
                if best_idx is not None:
                    val = str(df.iloc[best_idx, val_col]).strip()
                    if val:
                        return val, src
        except Exception:
            continue

    # Pass 2: matrix-style tables
    for df, src in tables:
        try:
            if df is None or df.empty:
                continue
            # Reverse lookup if numeric in Q: find matching cell; return sibling from ID-like column
            if q_nums:
                id_col = None
                for i, c in enumerate(df.columns):
                    cl = _normalize_name(c)
                    if any(tok in cl for tok in ("case", "id", "name", "label")):
                        id_col = i; break
                for i, row in df.iterrows():
                    row_vals = [str(x) for x in row.values.tolist()]
                    if any(any(num in v.lower().replace("μ", "u").replace("µ", "u") for num in q_nums) for v in row_vals):
                        # Prefer non-numeric, short text answer if possible
                        ans = None
                        if id_col is not None:
                            ans = str(row.iloc[id_col]).strip()
                        if not ans:
                            # otherwise return the column with highest header match
                            col_idx = _best_column_match(df, q_tokens) or 0
                            ans = str(row.iloc[col_idx]).strip()
                        if ans:
                            return ans, src
            # Otherwise, choose best row by overlap and best column by header match
            col_idx = _best_column_match(df, q_tokens)
            if col_idx is None:
                # Heuristic: if only one numeric column exists, take it
                col_idx = 0
                try:
                    num_counts = [sum(bool(re.search(r"\d", str(x))) for x in df.iloc[:, i]) for i in range(len(df.columns))]
                    if num_counts:
                        col_idx = int(max(range(len(num_counts)), key=lambda i: num_counts[i]))
                except Exception:
                    col_idx = 0
            best_i, best_s = None, 0.0
            for i, row in df.iterrows():
                vals = [str(x) for x in row.values.tolist()]
                sc = _row_overlap_score(vals, q_tokens)
                if sc > best_s:
                    best_s, best_i = sc, i
            if best_i is not None:
                val = str(df.iloc[best_i, col_idx]).strip()
                if val:
                    return val, src
        except Exception:
            continue

    return None, None
