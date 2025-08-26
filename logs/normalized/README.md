# Normalized outputs

This folder contains opt-in artifacts produced by `app/normalize_snapshot.py` from `logs/db_snapshot.jsonl`.

Artifacts:
- `chunks.jsonl` — one JSON object per line, for hybrid text retrieval (Chroma-ready).
- `graph.json` — a single JSON with nodes and edges for graph import.

Chunk schema (per line):
- id: deterministic SHA1 of (doc_id|chunk_id|content_hash|merge_span)
- document_id: source document identifier
- text: final chunk text (trimmed)
- section_title: one of [Executive Summary, System Description, Baseline Condition, Failure Progression, Investigation Findings, Results and Analysis, Recommendations, Conclusion]
- page_start, page_end: 1-based page indexes
- type: one of [text, figure_caption, figure_asset, table_data, timeline_event]
- anchor: original anchor (e.g., figure-2, table-03, p12-t3)
- source_id: "Fig:N" or "Tbl:N" when applicable
- date, date_start, date_end: extracted dates (YYYY-MM-DD)
- wear_stage: one of [baseline, mild, moderate, severe, failure]
- sensor_type: e.g., [accelerometer, camera]
- measurement_type: e.g., [RMS, FME, crest_factor]
- speed: e.g., [15, 45]
- extras: bag of useful fields (image_uri, table_csv, table_md, entities, file_title, section_raw, source_chunk_id)

Graph schema:
- nodes: array of { id, type, props }
  - types: Document, Section, Figure, Table, Stage, Metric, Sensor, Event
- edges: array of { type, from, to, props? }
  - types: BELONGS_TO, REFERS_TO, HAS_STAGE, HAS_METRIC, HAS_SENSOR, NEXT

Notes:
- Determinism: IDs are stable as long as the merge_span and inputs don’t change.
- Dedupe guard: identical (doc_id|chunk_id|content_hash) with identical text is emitted once, even across passes.
- Figures emit 2 chunks: `figure_caption` and `figure_asset` (merge_span suffix `#caption` / `#asset`).
- Tables emit a compact summary sentence by default (safe fallback when no richer summary is available).

How to run:
```
python .\app\normalize_snapshot.py --input .\logs\db_snapshot.jsonl --outdir .\logs\normalized
```
