#!/usr/bin/env python3
"""
I/O Handlers
===========

File I/O operations functions.
"""

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from RAG.app.logger import trace_func


@dataclass
class SnapshotRow:
    """Data class for snapshot rows."""
    raw: Dict[str, Any]

    @property
    def file(self) -> str:
        return self.raw.get("file", "")

    @property
    def page(self) -> int:
        return int(self.raw.get("page", 0))

    @property
    def section(self) -> str:
        return self.raw.get("section", "")

    @property
    def anchor(self) -> str:
        return self.raw.get("anchor", "")

    @property
    def doc_id(self) -> str:
        return self.raw.get("doc_id", "")

    @property
    def chunk_id(self) -> str:
        return self.raw.get("chunk_id", "")

    @property
    def content_hash(self) -> str:
        return self.raw.get("content_hash", "")

    @property
    def preview(self) -> str:
        return self.raw.get("preview", "")

    @property
    def image_path(self) -> Optional[str]:
        return self.raw.get("image_path")

    @property
    def figure_number(self) -> Optional[int]:
        return self.raw.get("figure_number")

    @property
    def figure_label(self) -> Optional[str]:
        return self.raw.get("figure_label")

    @property
    def table_number(self) -> Optional[int]:
        return self.raw.get("table_number")

    @property
    def table_label(self) -> Optional[str]:
        return self.raw.get("table_label")

    @property
    def table_csv_path(self) -> Optional[str]:
        return self.raw.get("table_csv_path")

    @property
    def table_md_path(self) -> Optional[str]:
        return self.raw.get("table_md_path")


@trace_func
def read_snapshot(path: str) -> List[SnapshotRow]:
    rows: List[SnapshotRow] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rows.append(SnapshotRow(data))
            except Exception as e:
                # skip malformed
                continue
    return rows


@trace_func
def filter_noise(rows: List[SnapshotRow]) -> List[SnapshotRow]:
    from .text_processors import is_footer_preview, is_heading_only
    
    out: List[SnapshotRow] = []
    for r in rows:
        if is_footer_preview(r.preview):
            continue
        # drop heading-only lines
        if is_heading_only(r.preview):
            continue
        out.append(r)
    return out


@trace_func
def write_outputs(chunks: List[Dict[str, Any]], graph: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    graph_path = os.path.join(out_dir, "graph.json")
    with io.open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with io.open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
