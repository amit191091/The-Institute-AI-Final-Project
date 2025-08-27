from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    # Match indexing.to_documents usage
    from langchain_core.documents import Document  # type: ignore
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _section_from_type(ctype: str, section_title: str | None) -> str:
    t = (ctype or "").lower()
    if t == "figure_caption":
        return "FigureCaption"
    if t == "figure" or t == "image":
        return "Figure"
    if t == "table_data":
        return "Table"
    # For everything else, reuse the canonical section title
    return section_title or "Results and Analysis"


def load_normalized_docs(chunks_path: str | os.PathLike[str]) -> List[Document]:
    """Read logs/normalized/chunks.jsonl and convert each chunk to a Document.

    Metadata keys are mapped to align with the existing UI and retriever logic.
    Safe: returns an empty list if the file is missing or malformed.
    """
    p = Path(chunks_path)
    if not p.exists():
        return []
    docs: List[Document] = []
    with io.open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            extras: Dict[str, Any] = c.get("extras") or {}
            source_id = str(c.get("source_id") or "")
            fig_no = None
            tbl_no = None
            m = re.match(r"fig\s*:?\s*(\d+)", source_id, re.I)
            if m:
                try:
                    fig_no = int(m.group(1))
                except Exception:
                    fig_no = None
            m = re.match(r"tbl\s*:?\s*(\d+)", source_id, re.I)
            if m:
                try:
                    tbl_no = int(m.group(1))
                except Exception:
                    tbl_no = None

            section = _section_from_type(str(c.get("type") or ""), c.get("section_title"))
            # Build table/figure display labels when applicable
            table_label = None
            figure_label = None
            if section == "Figure" and (str(c.get("type")) == "figure_caption"):
                figure_label = txt
            if section == "Table" and tbl_no is not None:
                # Prefer a label-like form
                if re.match(r"^table\s*\d+\b", txt, re.I):
                    table_label = txt
                else:
                    table_label = f"Table {tbl_no}: {txt}"

            md = {
                # Core fields expected by pipeline UI and retriever
                "file_name": extras.get("file_title") or c.get("document_id"),
                "page": _coerce_int(c.get("page_start")),
                "section": section,
                "anchor": c.get("anchor"),
                # Figure/Table metadata
                "figure_number": fig_no,
                "figure_label": figure_label,
                "table_number": tbl_no,
                "table_label": table_label,
                # Asset paths
                "image_path": extras.get("image_uri"),
                "table_md_path": extras.get("table_md"),
                "table_csv_path": extras.get("table_csv"),
                # Provenance
                "doc_id": c.get("document_id"),
                "chunk_id": extras.get("source_chunk_id") or c.get("id"),
                "anchor_norm": c.get("anchor"),
                # Hybrid metadata
                "section_title": c.get("section_title"),
                "wear_stage": c.get("wear_stage"),
                "sensor_type": c.get("sensor_type"),
                "measurement_type": c.get("measurement_type"),
                "speed": c.get("speed"),
                "date": c.get("date"),
                "date_start": c.get("date_start"),
                "date_end": c.get("date_end"),
                "anchor_type": c.get("type"),
            }
            docs.append(Document(page_content=txt, metadata=md))
    return docs
