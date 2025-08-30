#!/usr/bin/env python3
"""
Normalize logs/db_snapshot.jsonl into:
  - logs/normalized/chunks.jsonl  (Chunk JSONL for hybrid retrieval)
  - logs/normalized/graph.json    (Single JSON with nodes + edges for graph import)

This is a standalone, optional transformer. It does not modify the pipeline.
Safe defaults, deterministic IDs, minimal merging, and conservative heuristics.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

# Import from text_processors module
from .text_processors import norm_path, is_footer_preview, is_heading_only
# Import from date_parsers module
from .date_parsers import MONTHS, anchor_sort_key, infer_stage, parse_dates
# Import from date_parsers module
from .date_parsers import MONTHS, anchor_sort_key, infer_stage, parse_dates
# Import from measurement_extractors module
from .measurement_extractors import normalize_section_title, detect_measurements_sensors_speed, clean_caption, minimal_table_summary
# Import from chunk_generators module
from .chunk_generators import sha1
# Import from graph_builders module
from .graph_builders import write_outputs
# Import from io_handlers module
from .io_handlers import SnapshotRow, read_snapshot, filter_noise

from RAG.app.logger import trace_func

@trace_func
def maybe_upgrade_with_full_text(rows: List[SnapshotRow], full_path: Optional[str]) -> List[SnapshotRow]:
    """If a full snapshot exists (db_snapshot_full.jsonl), merge its text and metadata into rows.
    Match on (doc_id, chunk_id, content_hash). Missing keys are left as-is.
    """
    if not full_path or not os.path.exists(full_path):
        return rows
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    try:
        with io.open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = (str(data.get("doc_id", "")), str(data.get("chunk_id", "")), str(data.get("content_hash", "")))
                    index[key] = data
                except Exception:
                    continue
    except Exception:
        return rows
    out: List[SnapshotRow] = []
    for r in rows:
        key = (str(r.doc_id), str(r.chunk_id), str(r.content_hash))
        full = index.get(key)
        if full:
            # inject richer preview if empty
            if not r.raw.get("preview") and full.get("text"):
                r.raw["preview"] = str(full.get("text"))[:200]
            # propagate any missing fields we care about
            if not r.raw.get("file") and full.get("file"):
                r.raw["file"] = full.get("file")
            if not r.raw.get("page") and full.get("page") is not None:
                r.raw["page"] = full.get("page")
            # keep section/anchor as-is; they should already exist
        out.append(r)
    return out


@trace_func
def chunks_and_graph(rows: List[SnapshotRow]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Sort rows deterministically
    rows = sorted(rows, key=lambda r: anchor_sort_key(r.anchor, r.page))

    chunk_items: List[Dict[str, Any]] = []
    # Dedupe guard across passes: (doc_id, chunk_id, content_hash, final_text)
    seen_keys: Set[Tuple[str, str, str, str]] = set()
    # Graph accumulators
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, ntype: str, props: Optional[Dict[str, Any]] = None):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": ntype, "props": props or {}}

    def add_edge(et: str, src: str, dst: str, props: Optional[Dict[str, Any]] = None):
        edges.append({"type": et, "from": src, "to": dst, **({"props": props} if props else {})})

    # Document node (single doc assumption for snapshot)
    doc_ids = sorted({r.doc_id for r in rows if r.doc_id})
    for d in doc_ids:
        add_node(f"doc:{d}", "Document", {"title": rows[0].file if rows else d})

    # Helper to build chunk object
    def make_id(doc_id: str, base_chunk_id: str, content_hash: str, merge_span: str) -> str:
        return sha1(f"{doc_id}|{base_chunk_id}|{content_hash}|{merge_span}")

    def emit_chunk(
        *,
        row: SnapshotRow,
        text: str,
        section_title: str,
        ctype: str,
        page_start: int,
        page_end: int,
        source_id: Optional[str],
        anchor: str,
        extras: Dict[str, Any],
        merge_span: str,
        date: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        wear_stage: Optional[str] = None,
        measurement_type: Optional[List[str]] = None,
        sensor_type: Optional[List[str]] = None,
        speed: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        mt = measurement_type or []
        st = sensor_type or []
        sp = speed or []
        # Dedupe key is independent of merge_span to avoid duplicates across passes
        key = (row.doc_id, row.chunk_id, row.content_hash, (text or "").strip())
        if key in seen_keys:
            return None
        cid = make_id(row.doc_id, row.chunk_id, row.content_hash, merge_span)
        item = {
            "id": cid,
            "document_id": row.doc_id,
            "text": text.strip(),
            "section_title": section_title,
            "page_start": page_start,
            "page_end": page_end,
            "type": ctype,
            "anchor": anchor,
            "source_id": source_id,
            "date": date,
            "date_start": date_start,
            "date_end": date_end,
            "wear_stage": wear_stage,
            "sensor_type": st,
            "measurement_type": mt,
            "speed": sp,
            "extras": extras,
        }
        chunk_items.append(item)
        seen_keys.add(key)
        # Graph edges/nodes
        sec_id = f"sec:{section_title.replace(' ', '_')}"
        add_node(sec_id, "Section", {"title": section_title})
        add_edge("BELONGS_TO", f"chunk:{cid}", sec_id)
        add_edge("BELONGS_TO", sec_id, f"doc:{row.doc_id}")
        # Stage/metric/sensor
        if wear_stage:
            add_node(f"stage:{wear_stage}", "Stage", {})
            add_edge("HAS_STAGE", f"chunk:{cid}", f"stage:{wear_stage}")
        for m in mt:
            add_node(f"metric:{m}", "Metric", {})
            add_edge("HAS_METRIC", f"chunk:{cid}", f"metric:{m}")
        for s in st:
            add_node(f"sensor:{s}", "Sensor", {})
            add_edge("HAS_SENSOR", f"chunk:{cid}", f"sensor:{s}")
        # Refers-to Figure/Table
        if source_id:
            if source_id.lower().startswith("fig:"):
                fid = source_id.split(":", 1)[1]
                fig_node = f"fig:{fid}"
                add_node(fig_node, "Figure", {"caption": text[:160]})
                add_edge("REFERS_TO", f"chunk:{cid}", fig_node)
                # place figure under Results and Analysis
                add_edge("BELONGS_TO", fig_node, sec_id)
            elif source_id.lower().startswith("tbl:"):
                tid = source_id.split(":", 1)[1]
                tbl_node = f"tbl:{tid}"
                add_node(tbl_node, "Table", {"title": text[:160]})
                add_edge("REFERS_TO", f"chunk:{cid}", tbl_node)
                add_edge("BELONGS_TO", tbl_node, sec_id)
        return item

    # Entities (document-scoped, obvious)
    doc_entities = ["INS Haifa", "MG-5025A"]

    # Track emitted IDs to avoid duplicates
    emitted_ids: set = set()

    # Pass 1: emit figures and tables, plus figure assets
    for r in rows:
        if r.section == "Figure" and r.figure_number and r.figure_label:
            # Caption chunk
            caption = clean_caption(r.figure_label)
            mtypes, stypes, speeds = detect_measurements_sensors_speed(caption)
            section_title = "Results and Analysis"
            extras = {
                "entities": doc_entities,
                "image_uri": norm_path(r.image_path),
                "source_chunk_id": r.chunk_id,
                "section_raw": r.section,
                "file_title": r.file,
            }
            item_cap = emit_chunk(
                row=r,
                text=caption,
                section_title=section_title,
                ctype="figure_caption",
                page_start=r.page,
                page_end=r.page,
                source_id=f"Fig:{r.figure_number}",
                anchor=r.anchor,
                extras=extras,
                merge_span=f"{r.anchor}#caption",
                measurement_type=mtypes,
                sensor_type=stypes,
                speed=speeds,
            )
            if item_cap:
                emitted_ids.add(item_cap["id"])
            # Asset chunk
            asset_text = f"Asset for Fig:{r.figure_number}"
            extras_asset = {
                "image_uri": norm_path(r.image_path),
                "entities": doc_entities,
                "source_chunk_id": r.chunk_id,
                "section_raw": r.section,
                "file_title": r.file,
            }
            item_asset = emit_chunk(
                row=r,
                text=asset_text,
                section_title=section_title,
                ctype="figure_asset",
                page_start=r.page,
                page_end=r.page,
                source_id=f"Fig:{r.figure_number}",
                anchor=r.anchor,
                extras=extras_asset,
                merge_span=f"{r.anchor}#asset",
            )
            if item_asset:
                emitted_ids.add(item_asset["id"])
        elif r.section == "Table" and r.table_number and r.table_label:
            # Table summary chunk
            summary = minimal_table_summary(r.table_label)
            mtypes, stypes, speeds = detect_measurements_sensors_speed(summary)
            section_title = "Investigation Findings"
            extras = {
                "entities": doc_entities,
                "table_csv": norm_path(r.table_csv_path),
                "table_md": norm_path(r.table_md_path),
                "source_chunk_id": r.chunk_id,
                "section_raw": r.section,
                "file_title": r.file,
            }
            item_tbl = emit_chunk(
                row=r,
                text=summary,
                section_title=section_title,
                ctype="table_data",
                page_start=r.page,
                page_end=r.page,
                source_id=f"Tbl:{r.table_number}",
                anchor=r.anchor,
                extras=extras,
                merge_span=r.anchor,
                measurement_type=mtypes,
                sensor_type=stypes,
                speed=speeds,
            )
            if item_tbl:
                emitted_ids.add(item_tbl["id"])

    # Pass 2: timeline events (scan narrative rows for date phrases)
    # We choose the first row that mentions each key window to avoid duplicates.
    seen_dates: set = set()
    desired_patterns = [
        ("on april 9", "date"),
        ("from april 23", "range"),
        ("from may 14", "range"),
        ("on june 15", "date"),
    ]
    for r in rows:
        if r.section in ("Figure", "Table"):
            continue
        tl = r.preview.lower()
        for pat, kind in desired_patterns:
            if pat in tl:
                ds, de, d = parse_dates(r.preview)
                key = (ds or "") + (de or "") + (d or "")
                if key and key not in seen_dates:
                    seen_dates.add(key)
                    # Build event text: use a compact rewrite
                    if d:
                        text = f"{d}: event marker derived from narrative (prototype)."
                    else:
                        text = f"{ds} to {de}: event window derived from narrative (prototype)."
                    section_title = "Failure Progression"
                    mtypes, stypes, speeds = detect_measurements_sensors_speed(r.preview)
                    stage = infer_stage(ds, de, d)
                    extras = {
                        "entities": doc_entities,
                        "source_chunk_id": r.chunk_id,
                        "section_raw": r.section,
                        "file_title": r.file,
                    }
                    item_evt = emit_chunk(
                        row=r,
                        text=text,
                        section_title=section_title,
                        ctype="timeline_event",
                        page_start=r.page,
                        page_end=r.page,
                        source_id=None,
                        anchor=r.anchor,
                        extras=extras,
                        merge_span=r.anchor,
                        date=d,
                        date_start=ds,
                        date_end=de,
                        wear_stage=stage,
                        measurement_type=mtypes,
                        sensor_type=stypes,
                        speed=speeds,
                    )
                    if item_evt:
                        emitted_ids.add(item_evt["id"])
                break

    # Pass 3: narrative text merging (Text/Analysis etc.), conservative within a page and section
    def is_date_boundary(text: str) -> bool:
        tl = text.lower()
        return any(pat in tl for pat, _ in desired_patterns)

    i = 0
    n = len(rows)
    while i < n:
        r = rows[i]
        if r.section in ("Figure", "Table"):
            i += 1
            continue
        if is_date_boundary(r.preview) or is_heading_only(r.preview):
            i += 1
            continue
        # Start a group within same page and same normalized section
        section_title = normalize_section_title(r.section, r.preview)
        page = r.page
        group: List[SnapshotRow] = [r]
        word_count = len(re.findall(r"\w+", r.preview))
        j = i + 1
        while j < n:
            rj = rows[j]
            if rj.section in ("Figure", "Table"):
                break
            if rj.page != page:
                break
            if is_date_boundary(rj.preview) or is_heading_only(rj.preview):
                break
            stj = normalize_section_title(rj.section, rj.preview)
            if stj != section_title:
                break
            # accumulate until ~300 words
            new_wc = len(re.findall(r"\w+", rj.preview))
            if word_count + new_wc > 300:
                break
            group.append(rj)
            word_count += new_wc
            j += 1
        # Build merged text (preserve bullet feel by inserting " - " between short list-like lines)
        def line_to_sentence(t: str) -> str:
            t = t.strip()
            if re.match(r"^\d+\.|^-\s+", t):
                return f"- {t.split(maxsplit=1)[-1]}"
            return t
        merged_text = " ".join(line_to_sentence(g.preview) for g in group).strip()
        # Derive fields
        ds, de, d = parse_dates(merged_text)
        stage = infer_stage(ds, de, d)
        mtypes, stypes, speeds = detect_measurements_sensors_speed(merged_text)
        first = group[0]
        last = group[-1]
        merge_span = first.anchor if first.anchor == last.anchor else f"{first.anchor}..{last.anchor}"
        extras = {
            "entities": doc_entities,
            "source_chunk_id": first.chunk_id,
            "section_raw": first.section,
            "file_title": first.file,
        }
        item_txt = emit_chunk(
            row=first,
            text=merged_text,
            section_title=section_title,
            ctype="text",
            page_start=first.page,
            page_end=last.page,
            source_id=None,
            anchor=first.anchor,
            extras=extras,
            merge_span=merge_span,
            date=d,
            date_start=ds,
            date_end=de,
            wear_stage=stage,
            measurement_type=mtypes,
            sensor_type=stypes,
            speed=speeds,
        )
        # Dedupe by id
        if item_txt and item_txt["id"] not in emitted_ids:
            emitted_ids.add(item_txt["id"])
        i = j if j > i else i + 1

    # Graph: add NEXT edges for events in chronological order
    # Collect event nodes by date or midpoint
    event_chunks = [c for c in chunk_items if c.get("type") == "timeline_event"]
    def event_key(c: Dict[str, Any]) -> str:
        if c.get("date"):
            return c["date"]
        # midpoint for range
        ds, de = c.get("date_start"), c.get("date_end")
        return (ds or "") + "~" + (de or "")

    event_chunks_sorted = sorted(event_chunks, key=event_key)
    # Add nodes for Events and NEXT edges
    prev_event_id: Optional[str] = None
    for ec in event_chunks_sorted:
        d = ec.get("date")
        if d:
            eid = f"event:{d}"
            add_node(eid, "Event", {"date": d})
            # Link event to stage/sensors/metrics as well
            if ec.get("wear_stage"):
                add_edge("HAS_STAGE", eid, f"stage:{ec['wear_stage']}")
            for m in ec.get("measurement_type", []) or []:
                add_edge("HAS_METRIC", eid, f"metric:{m}")
            for s in ec.get("sensor_type", []) or []:
                add_edge("HAS_SENSOR", eid, f"sensor:{s}")
            if prev_event_id:
                add_edge("NEXT", prev_event_id, eid)
            prev_event_id = eid
        else:
            # For ranges, create two nodes and a NEXT edge between them
            ds, de = ec.get("date_start"), ec.get("date_end")
            if ds:
                eid_s = f"event:{ds}"
                add_node(eid_s, "Event", {"date": ds})
                if prev_event_id:
                    add_edge("NEXT", prev_event_id, eid_s)
                prev_event_id = eid_s
            if de:
                eid_e = f"event:{de}"
                add_node(eid_e, "Event", {"date": de})
                if prev_event_id and prev_event_id != eid_e:
                    add_edge("NEXT", prev_event_id, eid_e)
                prev_event_id = eid_e

    graph = {"nodes": list(nodes.values()), "edges": edges}
    return chunk_items, graph



@trace_func
def main():
    parser = argparse.ArgumentParser(description="Normalize db_snapshot.jsonl → chunks.jsonl + graph.json")
    parser.add_argument("--input", default=os.path.join("logs", "db_snapshot.jsonl"))
    parser.add_argument("--full", default=os.path.join("logs", "db_snapshot_full.jsonl"), help="Optional path to non-normalized full snapshot (text+metadata)")
    parser.add_argument("--outdir", default=os.path.join("logs", "normalized"))
    args = parser.parse_args()

    rows = read_snapshot(args.input)
    rows = maybe_upgrade_with_full_text(rows, args.full)
    rows = filter_noise(rows)
    chunks, graph = chunks_and_graph(rows)
    write_outputs(chunks, graph, args.outdir)
    print(f"Wrote {len(chunks)} chunks → {os.path.join(args.outdir, 'chunks.jsonl')}\nGraph nodes={len(graph.get('nodes', []))}, edges={len(graph.get('edges', []))} → {os.path.join(args.outdir, 'graph.json')}")


if __name__ == "__main__":
    main()
