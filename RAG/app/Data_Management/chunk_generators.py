#!/usr/bin/env python3
"""
Chunk Generators
===============

Chunk creation and merging logic functions.
"""

import hashlib
from typing import Any, Dict, List, Optional, Set

from RAG.app.logger import trace_func

from .text_processors import norm_path
from .date_parsers import parse_dates, infer_stage
from .measurement_extractors import detect_measurements_sensors_speed, normalize_section_title


@trace_func
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

@trace_func
def emit_chunk(
    row,
    text: str,
    section_title: str,
    ctype: str,
    page_start: int,
    page_end: int,
    source_id: Optional[str],
    anchor: str,
    extras: Dict[str, Any],
    merge_span: Optional[int] = None,
    date: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    wear_stage: Optional[str] = None,
    measurement_type: Optional[List[str]] = None,
    sensor_type: Optional[List[str]] = None,
    speed: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Emit a chunk with all metadata."""
    if not text or not text.strip():
        return None
    
    # Generate deterministic ID
    content_hash = sha1(text.strip())
    chunk_id = f"{ctype}:{content_hash[:8]}"
    
    chunk = {
        "id": chunk_id,
        "type": ctype,
        "text": text.strip(),
        "section": section_title,
        "page_start": page_start,
        "page_end": page_end,
        "source_id": source_id,
        "anchor": anchor,
        "date": date,
        "date_start": date_start,
        "date_end": date_end,
        "wear_stage": wear_stage,
        "measurement_type": measurement_type or [],
        "sensor_type": sensor_type or [],
        "speed": speed or [],
        "merge_span": merge_span,
        **extras
    }
    
    return chunk


@trace_func
def merge_adjacent_chunks(rows: List, max_merge_span: int = 3) -> List[Dict[str, Any]]:
    """Merge adjacent text chunks with similar characteristics."""
    if not rows:
        return []
    
    chunk_items = []
    emitted_ids: Set[str] = set()
    
    i = 0
    while i < len(rows):
        first = rows[i]
        
        # Find mergeable range
        j = i + 1
        while j < len(rows):
            next_row = rows[j]
            
            # Check if mergeable
            if (next_row.page - first.page > max_merge_span or
                next_row.section != first.section or
                next_row.anchor != first.anchor):
                break
            j += 1
        
        # Merge text from i to j-1
        merged_text = " ".join(rows[k].text for k in range(i, j))
        last = rows[j-1]
        
        # Parse dates from merged text
        ds, de, d = parse_dates(merged_text)
        stage = infer_stage(ds, de, d)
        
        # Extract measurements
        mtypes, stypes, speeds = detect_measurements_sensors_speed(merged_text)
        
        # Determine section title
        section_title = normalize_section_title(first.section, merged_text)
        
        # Create extras
        extras = {}
        if first.source_id:
            extras["source_id"] = norm_path(first.source_id)
        
        merge_span = j - i
        
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
            chunk_items.append(item_txt)
        
        i = j
    
    return chunk_items

