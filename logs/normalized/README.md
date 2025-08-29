# Normalized artifacts (optional)

This folder is produced by `app/normalize_snapshot.py` from `logs/db_snapshot.jsonl` to enable stable A/B runs, fast reindexing, and graph import.

Artifacts
- `chunks.jsonl` — normalized chunks for direct indexing (used when `RAG_USE_NORMALIZED=1`).
- `graph.json` — simple node/edge JSON for previews and optional Neo4j import (`RAG_IMPORT_NORMALIZED_GRAPH=1`).

Chunk fields (typical)
- `id` (stable), `document_id`, `text`, `type` (text|figure_caption|figure_asset|table_data|timeline_event)
- `page_start`, `page_end`, `anchor`, `source_id` (e.g., Fig:2, Tbl:1)
- Optional: `date*`, `wear_stage`, `sensor_type`, `measurement_type`, `speed`, `extras`

Graph fields (typical)
- `nodes`: { id, type, props }, types like Document, Section, Figure, Table, Stage, Metric, Sensor, Event
- `edges`: { type, from, to, props? }, types like BELONGS_TO, REFERS_TO, HAS_STAGE, HAS_METRIC, HAS_SENSOR, NEXT

Generate
```powershell
python .\app\normalize_snapshot.py --input .\logs\db_snapshot.jsonl --outdir .\logs\normalized
```
