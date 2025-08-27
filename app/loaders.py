#!/usr/bin/env python3
"""Lightweight document loader for tables and figures/images.

This module focuses on robust, dependency-optional extraction for the pipeline:
 - Tables via: pdfplumber (default on), Camelot (optional), Tabula (optional, Java).
 - Images via: PyMuPDF (default on; fast and reliable on Windows).
 - LlamaParse (optional) for enhanced table parsing when an API key is available.

Environment flags (1/true/on to enable):
 - RAG_USE_PDFPLUMBER (default: 1)
 - RAG_USE_CAMELOT (default: 0)
 - RAG_USE_TABULA (default: 0)
 - RAG_USE_PYMUPDF (default: 1)
 - RAG_USE_LLAMA_PARSE (default: 0) requires LLAMA_CLOUD_API_KEY

The primary entrypoints are:
 - load_elements(path: Path) -> List[Element]
 - load_many(paths: List[Path]) -> List[tuple[Path, List[Element]]]

Returned Element objects carry:
 - category: "table" or "image"
 - text: markdown for tables; brief tag for images
 - metadata: includes extractor name, page_number, and file paths (e.g., image_path)
"""

import os
import re
from pathlib import Path
from typing import List
from app.logger import get_logger

# Optional imports with graceful fallbacks
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import camelot
except ImportError:
    camelot = None

try:
    import tabula
except ImportError:
    tabula = None

# Images via PyMuPDF (fitz)
try:
    import fitz  # type: ignore
except Exception:
    fitz = None  # type: ignore

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

# Simple element classes for testing
class Element:
    """Minimal carrier used by the pipeline prior to chunking/indexing.

    Attributes:
        text: Raw content. For tables, prefer markdown; for images, a short tag.
        category: One of {"table", "image"}.
        metadata: Dict with extractor name, page_number, and optional file paths.
    """

    def __init__(self, text: str = "", category: str = "", metadata=None):
        self.text = text
        self.category = category
        self.metadata = metadata or {}

class Table(Element):
    """Convenience wrapper for table elements."""

    def __init__(self, text: str = "", metadata=None):
        super().__init__(text, "table", metadata)


class Figure(Element):
    """Convenience wrapper for figure/image elements."""

    def __init__(self, text: str = "", metadata=None):
        super().__init__(text, "image", metadata)

# Use central application logger from app.logger

def _env_enabled(var_name: str, default: bool = False) -> bool:
    """Return True if the env var is a truthy flag.

    Truthy values: 1, true, yes, on (case-insensitive). Defaults to `default`.
    """
    value = os.environ.get(var_name, "1" if default else "0")
    return str(value).lower() in ("1", "true", "yes", "on")

def _export_tables_to_files(elements: List[Element], path: Path) -> None:
    """Persist detected table elements to data/elements as Markdown and CSV files.

    For each table element, write two files:
      - Markdown: <stem>-table-XX.md (always)
      - CSV:      <stem>-table-XX.csv (best-effort parsed from markdown rows)

    Attach file paths back into element.metadata as table_md_path/table_csv_path
    and set a stable table_number (1-based) if not already present.

    Controlled by env RAG_EXPORT_TABLES (default: on).
    """
    if not _env_enabled("RAG_EXPORT_TABLES", True):
        return
    try:
        out_dir = Path("data") / "elements"
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    # Determine document-order tables
    tbl_indices = [i for i, e in enumerate(elements) if str(getattr(e, "category", "")).lower() == "table"]
    for order, idx in enumerate(tbl_indices, start=1):
        e = elements[idx]
        text = (getattr(e, "text", "") or "").strip()
        if not text:
            continue
        stem = f"{path.stem}-table-{order:02d}"
        md_path = out_dir / f"{stem}.md"
        csv_path = out_dir / f"{stem}.csv"
        # Write Markdown as-is
        try:
            md_path.write_text(text, encoding="utf-8")
        except Exception:
            continue
        # Best-effort CSV from markdown row lines
        try:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            csv_rows: List[str] = []
            for ln in lines:
                if not ln.startswith("|"):
                    continue
                if "---" in ln:
                    # header separator
                    continue
                cells = [c.strip() for c in ln.strip("|").split("|")]
                # CSV escaping of quotes
                csv_rows.append(
                    ",".join(f'"{c.replace("\"", "\"\"")}"' for c in cells)
                )
            if csv_rows:
                csv_path.write_text("\n".join(csv_rows), encoding="utf-8")
        except Exception:
            # ignore CSV failures; MD is still useful
            pass
        # Attach metadata
        try:
            meta = getattr(e, "metadata", None)
            if isinstance(meta, dict):
                meta.setdefault("table_number", order)
                meta["table_md_path"] = md_path.as_posix()
                if csv_path.exists():
                    meta["table_csv_path"] = csv_path.as_posix()
            elif meta is not None:
                # SimpleNamespace or other attr-based containers
                try:
                    if getattr(meta, "table_number", None) is None:
                        setattr(meta, "table_number", order)
                    setattr(meta, "table_md_path", md_path.as_posix())
                    if csv_path.exists():
                        setattr(meta, "table_csv_path", csv_path.as_posix())
                except Exception:
                    pass
        except Exception:
            pass

def _table_to_markdown(table) -> str:
    """Convert a sequence of rows (list[list]) into GitHub-flavored markdown."""
    if not table or len(table) < 2:
        return ""
    
    lines = []
    # Header
    header = "| " + " | ".join(str(cell or "") for cell in table[0]) + " |"
    lines.append(header)
    
    # Separator
    separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
    lines.append(separator)
    
    # Data rows
    for row in table[1:]:
        data_row = "| " + " | ".join(str(cell or "") for cell in row) + " |"
        lines.append(data_row)
    
    return "\n".join(lines)

def _try_pdfplumber_tables(pdf_path: Path) -> List[Element]:
    """Extract tables with pdfplumber.

    Enabled by RAG_USE_PDFPLUMBER (default: on). Produces Table elements with
    markdown text and metadata: extractor, page_number, table_index.
    """
    if not pdfplumber or not _env_enabled("RAG_USE_PDFPLUMBER", True):
        return []
    
    log = get_logger()
    debug = _env_enabled("RAG_PDFPLUMBER_DEBUG")
    
    if debug:
        log.info(f"pdfplumber: Starting extraction from {pdf_path.name}")
    
    elements = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                
                if debug:
                    log.info(f"pdfplumber: Page {page_num} - found {len(tables)} tables")
                
                for i, table in enumerate(tables):
                    if table and len(table) > 1:
                        markdown = _table_to_markdown(table)
                        metadata = {
                            "extractor": "pdfplumber",
                            "page_number": page_num,
                            "table_index": i
                        }
                        elements.append(Table(text=markdown, metadata=metadata))
    
    except Exception as e:
        log.warning(f"pdfplumber extraction failed: {e}")
    
    if debug:
        log.info(f"pdfplumber: Extracted {len(elements)} tables total")
    
    return elements

def _try_camelot_tables(pdf_path: Path) -> List[Element]:
    """Extract tables using Camelot (optional).

    Requires: camelot-py[base] and Ghostscript installed on the system.
    Enabled by RAG_USE_CAMELOT (default: off).
    """
    if not camelot or not _env_enabled("RAG_USE_CAMELOT"):
        return []
    
    log = get_logger()
    debug = _env_enabled("RAG_CAMELOT_DEBUG")
    
    if debug:
        log.info(f"camelot: Starting extraction from {pdf_path.name}")
    
    elements = []
    
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all")
        
        if debug:
            log.info(f"camelot: Found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            markdown = table.df.to_markdown(index=False)
            metadata = {
                "extractor": "camelot",
                "page_number": table.page,
                "table_index": i
            }
            elements.append(Table(text=markdown, metadata=metadata))
    
    except Exception as e:
        if debug:
            log.warning(f"camelot extraction failed: {e}")
    
    if debug:
        log.info(f"camelot: Extracted {len(elements)} tables total")
    
    return elements

def _try_tabula_tables(pdf_path: Path) -> List[Element]:
    """Extract tables using Tabula (optional).

    Requires: Java runtime installed and accessible. Enabled by RAG_USE_TABULA (default: off).
    """
    if not tabula or not _env_enabled("RAG_USE_TABULA"):
        return []
    
    log = get_logger()
    debug = _env_enabled("RAG_TABULA_DEBUG")
    
    if debug:
        log.info(f"tabula: Starting extraction from {pdf_path.name}")
    
    elements = []
    
    try:
        dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
        
        if debug:
            log.info(f"tabula: Found {len(dfs)} tables")
        
        for i, df in enumerate(dfs):
            markdown = df.to_markdown(index=False)
            metadata = {
                "extractor": "tabula",
                "table_index": i
            }
            elements.append(Table(text=markdown, metadata=metadata))
    
    except Exception as e:
        if debug:
            log.warning(f"tabula extraction failed: {e}")
    
    if debug:
        log.info(f"tabula: Extracted {len(elements)} tables total")
    
    return elements


def _try_pymupdf_images(pdf_path: Path) -> List[Element]:
    """Extract embedded images using PyMuPDF (fitz).

    Saves images to data/images/<stem>-p<page>-img<idx>.png and returns Figure elements
    with metadata: extractor, page_number, image_path, figure_order (per-page index).
    Controlled by RAG_USE_PYMUPDF (default: on).
    """
    if not fitz or not _env_enabled("RAG_USE_PYMUPDF", True):
        return []
    log = get_logger()
    out_dir = Path("data") / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    elements: List[Element] = []
    try:
        doc = fitz.open(str(pdf_path))  # type: ignore[misc]
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            images = page.get_images(full=True)
            for idx, img in enumerate(images, start=1):
                try:
                    xref = img[0]
                    base = f"{pdf_path.stem}-p{pno+1}-img{idx}.png"
                    out_path = out_dir / base
                    pix = doc.extract_image(xref)
                    # Some images are already PNG/JPEG; keep extension consistent for consumer simplicity
                    with open(out_path, "wb") as f:
                        f.write(pix.get("image", b""))
                    meta = {
                        "extractor": "pymupdf",
                        "page_number": pno + 1,
                        "image_path": str(out_path.as_posix()),
                        "figure_order": idx,
                    }
                    # Minimal text payload to let chunker identify this as a figure
                    txt = f"[FIGURE]\nImage file: {out_path.as_posix()}\nPage: {pno+1}"
                    elements.append(Figure(text=txt, metadata=meta))
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"PyMuPDF image extraction failed: {e}")
    return elements


def _try_llamaparse_tables(pdf_path: Path) -> List[Element]:
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
    elements: List[Element] = []
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

def load_elements(path: Path) -> List[Element]:
    """Extract tables and images from a single document path.

    Runs all enabled extractors, merges results, and emits a concise per-extractor summary.
    """
    log = get_logger()
    log.info(f"Loading elements from: {path}")
    # Log active toggles once per call to help debug unexpected extractor runs
    try:
        toggles = {
            "pdfplumber": _env_enabled("RAG_USE_PDFPLUMBER", True),
            "camelot": _env_enabled("RAG_USE_CAMELOT", False),
            "tabula": _env_enabled("RAG_USE_TABULA", False),
            "llamaparse": _env_enabled("RAG_USE_LLAMA_PARSE", False),
            "pymupdf": _env_enabled("RAG_USE_PYMUPDF", True),
        }
        log.info(
            "toggles: pdfplumber=%s, camelot=%s, tabula=%s, llamaparse=%s, pymupdf=%s",
            toggles["pdfplumber"],
            toggles["camelot"],
            toggles["tabula"],
            toggles["llamaparse"],
            toggles["pymupdf"],
        )
    except Exception:
        pass
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    elements: List[Element] = []

    # Try each enabled extractor (tables)
    extractors = [
        ("pdfplumber", _try_pdfplumber_tables),
        ("camelot", _try_camelot_tables),
        ("tabula", _try_tabula_tables),
        ("llamaparse", _try_llamaparse_tables),
    ]
    # Optional exclusivity: set RAG_EXCLUSIVE_EXTRACTOR to one of the names above
    # to run only that extractor (useful for smoke tests).
    exclusive = os.environ.get("RAG_EXCLUSIVE_EXTRACTOR")
    if exclusive:
        extractors = [e for e in extractors if e[0] == exclusive]
    
    for extractor_name, extractor_func in extractors:
        try:
            extracted = extractor_func(path)
            elements.extend(extracted)
            log.info(f"{extractor_name}: extracted {len(extracted)} elements")
        except Exception as e:
            log.warning(f"{extractor_name}: extraction failed: {e}")
    # Images (PyMuPDF)
    try:
        imgs = _try_pymupdf_images(path)
        elements.extend(imgs)
        if imgs:
            log.info(f"pymupdf: extracted {len(imgs)} images")
    except Exception as e:
        log.warning(f"pymupdf: extraction failed: {e}")

    # Persist table elements to files (MD + CSV) and backfill metadata with paths
    try:
        _export_tables_to_files(elements, path)
    except Exception as e:
        log = get_logger()
        try:
            log.warning(f"table export failed: {e}")
        except Exception:
            pass

    # Log summary
    tables = [e for e in elements if e.category == "table"]
    table_extractors = sorted(set(
        e.metadata.get("extractor", "") for e in tables
    ))
    images = [e for e in elements if e.category in ("image", "figure")]

    log.info(f"{path.name}: tables={len(tables)} via {table_extractors} | images={len(images)}")

    for i, table in enumerate(tables, 1):
        first_line = table.text.split('\n')[0] if table.text else ""
        log.info(f"{path.name}: Table {i} ({table.metadata.get('extractor', '?')}) -> {first_line[:60]}...")
    for i, fig in enumerate(images, 1):
        log.info(f"{path.name}: Figure {i} ({fig.metadata.get('extractor', '?')}) -> {fig.metadata.get('image_path', '')}")

    return elements


def load_many(paths: List[Path]) -> List[Element]:
    """Batch-extract elements for multiple paths, returning (path, elements) tuples.

    The pipeline expects this exact return shape, so callers can iterate
    `for path, elements in load_many(paths)`.
    """
    all_results = []
    for path in paths:
        try:
            elements = load_elements(path)
            # Pipeline expects (path, elements) tuples
            all_results.append((path, elements))
        except Exception as e:
            get_logger().warning(f"Failed to load {path}: {e}")
            all_results.append((path, []))  # Empty elements on failure
    
    return all_results


if __name__ == "__main__":
    """Lightweight CLI for extractor smoke tests.

    Usage (PowerShell):
      python -m app.loaders "Gear wear Failure.pdf"

    Honors env flags described in the module docstring. Prints table/image counts
    and first few entries to stdout.
    """
    import sys
    try:
        from dotenv import load_dotenv  # type: ignore
        # Do not override existing env from the shell; let inline flags win.
        load_dotenv(override=False)
    except Exception:
        pass
    log = get_logger()
    if len(sys.argv) < 2:
        print("usage: python -m app.loaders <path-to-pdf>")
        sys.exit(2)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"not found: {p}")
        sys.exit(2)
    els = load_elements(p)
    nt = sum(1 for e in els if e.category == "table")
    ni = sum(1 for e in els if e.category in ("image", "figure"))
    print(f"{p.name}: tables={nt} images={ni}")
    # Sample prints
    for e in els[: min(5, len(els))]:
        if e.category == "table":
            head = (e.text or "").splitlines()[0] if (e.text or "") else ""
            print(f"  - table via {e.metadata.get('extractor')} p={e.metadata.get('page_number')} | {head}")
        else:
            print(f"  - image via {e.metadata.get('extractor')} p={e.metadata.get('page_number')} -> {e.metadata.get('image_path')}")
    try:
        log.info("[smoke] %s: tables=%d images=%d", p.name, nt, ni)
    except Exception:
        pass