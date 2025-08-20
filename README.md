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
