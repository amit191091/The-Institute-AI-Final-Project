# Metadata-Driven Hybrid RAG for Failure Analysis Reports

This project ingests PDF/DOC/DOCX/TXT failure-analysis documents, performs structure-aware chunking with anchors, enriches chunks with metadata, builds dense (Chroma) and sparse (BM25) indices, and provides a Hybrid Retrieval + Router (Summary/Needle/Table) QA UI with Gradio.

## Quickstart

1. Create .env with your keys (prefer Google first):

```
GOOGLE_API_KEY=your_key
# optional fallback
OPENAI_API_KEY=your_key
OPENAI_CHAT_MODEL=gpt-4o-mini
CLIENT_ID=DEMO-CLIENT
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Put your PDFs in `data/` (or use the root `Gear wear Failure.pdf`).

4. Run:

```
python main.py
```

The app builds the index and launches a Gradio UI.

## Notes
- Dense K≈10 ∪ Sparse K≈10 → filter by metadata → rerank K≈20 → pass 6–8 to LLM.
- Metadata includes ≥5 fields: filename, page, section, chunk_summary, keywords, critical_entities, optional client_id/case_id, incident fields.
- Tables are summarized with anchors retained and content available for Table-QA.

## Environment toggles

- RAG_HEADLESS=true — build indices only and skip Gradio UI (useful for servers/CI).
- RAG_PDF_HI_RES=false — skip hi_res (OCR/vision) parser and use standard PDF parsing.
- RAG_OCR_LANG=eng — language(s) for OCR when using hi_res (comma-separated like "eng+heb").
- RAG_USE_TABULA=true — try extracting tables via Java Tabula in addition to standard parsing.
- RAG_EXTRACT_IMAGES=true — extract embedded images via PyMuPDF and index as Figure contexts.
- RAG_SYNTH_TABLES=true — synthesize table-like elements from text blocks when native extractors fail.
- RAG_USE_PDFPLUMBER=true — try extracting tables via pdfplumber (pure Python; good fallback).
- RAG_USE_CAMELOT=true — try extracting tables via Camelot (needs Ghostscript/Poppler).

### Optional, normalized pipeline and graph

- RAG_USE_NORMALIZED=true — index from logs/normalized/chunks.jsonl if present (A/B without changing extractors).
- RAG_USE_NORMALIZED_GRAPH=true — make the Graph tab read logs/normalized/graph.json when present.
- RAG_IMPORT_NORMALIZED_GRAPH=true — import logs/normalized/graph.json (+ chunks.jsonl) into Neo4j with ON_PAGE edges.
- RAG_GRAPH_DB=true — enable Neo4j population from current docs (entity co-mention graph) during ingestion.

Artifacts written by the normalization tools (optional):
- logs/normalized/chunks.jsonl — deterministic chunk records with stable anchors/labels.
- logs/normalized/graph.json — compact graph (nodes/edges) suitable for import or UI preview.

### Optional, LLM-first router and CE reranker

- RAG_USE_LLM_ROUTER=true — enable LLM-powered intent parser; falls back to regex if disabled/unavailable.
- RAG_INTENT_PROVIDER=openai|google — choose provider; defaults: OpenAI if OPENAI_API_KEY set else Google.
- OPENAI_CHAT_MODEL=gpt-4o-mini — small/cheap router model; override as needed.
- GOOGLE_CHAT_MODEL=gemini-1.5-flash — small/fast router model; override as needed.
- RAG_USE_CE_RERANKER=true — enable a second-stage cross-encoder reranker after the default reranker.
- RAG_CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 — default CE model name.

Example .env additions:

```
# Normalized artifacts and graph
RAG_USE_NORMALIZED=1
RAG_USE_NORMALIZED_GRAPH=1
# RAG_IMPORT_NORMALIZED_GRAPH=1  # optional, imports into Neo4j
# RAG_GRAPH_DB=1                 # optional, co-mention graph import

# LLM intent router
RAG_USE_LLM_ROUTER=1
RAG_INTENT_PROVIDER=google
GOOGLE_CHAT_MODEL=gemini-1.5-flash
# OPENAI_CHAT_MODEL=gpt-4o-mini

# Cross-encoder reranker
RAG_USE_CE_RERANKER=true
RAG_CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Evaluation files (auto-load)

The UI auto-loads ground truths and QA if the following files exist:
- data/gear_wear_ground_truth_context_free.json or gear_wear_ground_truth_context_free.json
- data/gear_wear_qa_context_free.jsonl or gear_wear_qa_context_free.jsonl
- Also supports gear_wear_ground_truth.json and gear_wear_qa.(json|jsonl)

Supported schemas:
- context_free GT: { id -> { answer or acceptable_answers } } and QA: [ { id, question } ] — joined by id at load time.
- direct pairs: list of { question, ground_truths } or { question, answer/reference }.
- flexible nested JSON: loader scans for question + answers/reference fields.

Notes:
- For table questions (sensors/thresholds), retrieval biases prefer Table sections; CE reranker is optional and improves ordering.
- Router can be LLM-first (flagged) with a strict JSON schema; falls back to regex when disabled or no API key.
