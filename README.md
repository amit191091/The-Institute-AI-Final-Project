# The Institute AI Final Project — Metadata‑Driven Hybrid RAG

End‑to‑end Retrieval‑Augmented Generation pipeline for technical reports. Ingests PDFs/DOCs/TXT, performs structure‑aware chunking, enriches metadata, builds hybrid indices (dense + BM25), and answers questions via a router that selects between Summary, Needle, and Table agents. Optional graph, normalized artifacts, LlamaIndex comparators, and automated evaluation (RAGAS + DeepEval) are included.

## Highlights

- Multi‑parser ingestion with robust table and figure handling (pdfplumber, tabula, camelot, PyMuPDF images)
- Structure‑aware chunking with anchors and stable metadata (file, page, section, anchors, labels, table/figure numbers)
- Hybrid retrieval: sparse BM25 + dense embeddings with tunable weights and optional CE reranker
- Router + agents: summary, precise extractive needle, and schema‑agnostic table QA (natural lookups)
- Gradio UI with filters, DB explorer, table/figure previews, router diagnostics, and optional graph view
- Normalized artifacts and optional Neo4j graph import; LlamaIndex export/compare for A/B
- Automated evaluation pipeline with RAGAS and optional Confident AI DeepEval

## Repository layout

- `Main.py` — entry point delegating to `app/pipeline.py`
- `app/` — core modules
	- `pipeline.py` — orchestration: ingestion → chunking → metadata → indexing → retrieval → UI/eval
	- `loaders.py`, `normalized_loader.py` — input loaders and normalized doc loader
	- `chunking.py` — structure/semantic chunking and packing controls
	- `metadata.py` — metadata enrichment (sections, anchors, dates, labels, tokens)
	- `indexing.py` — vector store creation (Chroma/docarray/FAISS), sparse BM25, KV expansion for tables
	- `retrieve.py` — hybrid retriever (Ensemble), filters, reranker, lexical scoring and boosts
	- `agents.py` — router helpers and answer agents: summary, needle (extractive), table QA
	- `table_ops.py` — natural table lookup utilities (schema‑agnostic value retrieval)
	- `agent_orchestrator.py`,`agent_tools.py` — optional tool‑driven orchestrator for traceable answers
	- `router_chain.py`,`query_intent.py` — optional LLM‑based intent routing (JSON schema); rules fallback
	- `ui_gradio.py` — UI: ask, filters, previews, evaluation runner, optional graph tab
	- `graph.py`,`graphdb.py`,`graphdb_import_normalized.py` — optional graph creation and Neo4j import
	- `llamaindex_export.py`,`llamaindex_compare.py` — optional LlamaIndex export/A/B hybrid stores
	- `prompts.py` — system/user prompts and few‑shots for agents
	- `eval_ragas.py`,`eval_deepeval.py`,`eval_factual.py` — evaluation utilities
	- `logger.py` — tracing and structured logging helpers
	- `config.py` — default knobs (K values, chunking ranges)
- `data/` — input PDFs and QA/GT datasets
- `index/` — vector store persistence (e.g., `chroma/`)
- `logs/` — snapshots, run logs, per‑question evals, optional graph HTML
	- `logs/normalized/` — normalized chunks/graph artifacts (optional)
- `scripts/` — quick probes and smoke scripts
- `tests/` — unit/e2e tests and smoke checks

## Installation

Prerequisites:
- Python 3.10+
- Optional for table extraction: Java (Tabula), Ghostscript/Poppler (Camelot)
- Optional for graph DB: Neo4j 5.x reachable via env vars

Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

Create a `.env` (Google preferred by default; OpenAI as fallback):

```dotenv
GOOGLE_API_KEY=your_key
# optional
OPENAI_API_KEY=your_key
OPENAI_CHAT_MODEL=gpt-4.1-nano
CLIENT_ID=DEMO-CLIENT
```

Place documents under `data/` (or keep root `Gear wear Failure.pdf`).

## Running

By default, `Main.py` sets headless + evaluation to avoid launching UI in CI.

- Run full pipeline with UI:

```powershell
$env:RAG_HEADLESS="0"; $env:RAG_EVAL="0"; python .\Main.py
```

- Headless with evaluation only (default):

```powershell
python .\Main.py
```

- UI‑only reusing persisted artifacts (skip ingestion):

```powershell
$env:RAG_UI_ONLY="1"; python .\Main.py
```

UI server options:
- `GRADIO_SERVER_NAME` (default 127.0.0.1)
- `GRADIO_PORT` (default 7860)
- `GRADIO_SHARE=1` to enable Gradio share

## Core pipeline

1) Ingestion: parses PDFs/DOCs/TXT and extracts text, tables (pdfplumber/tabula/camelot), and figures (PyMuPDF images) with anchors.  
2) Chunking: structure‑aware packing with semantic/multi‑split options and page/section boundaries.  
3) Metadata: attaches file/page/section, anchors, labels (Table/Figure numbers), date tokens, etc.  
4) Indexing: builds dense vector store (Chroma/docarray/FAISS) + sparse BM25; optional persist to `RAG_CHROMA_DIR`.  
5) Retrieval: hybrid ensemble with tunable weights, metadata filters, domain boosts, and optional CE reranker.  
6) Routing/Agents: LLM/router selects agent (summary/needle/table); table agent supports natural lookups and KV mini‑docs.  
7) UI & Eval: Gradio app for QA, debugging, and running RAGAS/DeepEval; optional graph rendering and Neo4j import.

## Retrieval details

- Embeddings preference: Google (`models/text-embedding-004`) → OpenAI (`text-embedding-3-small`) → FakeEmbeddings fallback (dev)
- Vector backends: `RAG_VECTOR_BACKEND=chroma|docarray|faiss` (default: chroma)
- Persistence: set `RAG_CHROMA_DIR` and optional `RAG_CHROMA_COLLECTION`; snapshots written to `logs/`
- Sparse retriever: BM25 (`k` from settings)
- Hybrid weights: `RAG_SPARSE_WEIGHT` and `RAG_DENSE_WEIGHT` (normalized)
- Second‑stage rerank (optional): `RAG_USE_CE_RERANKER=1` with `RAG_CE_MODEL`
- Precision pruning: `RAG_MIN_CTX_SCORE` to drop low‑score contexts post‑rerank

## Routing and agents

- Router: rules‑based by default; optional LLM router (`RAG_USE_LLM_ROUTER=1`) via `query_intent.py`
- Agents:
	- Summary — synthesizes concise, cited summary from multi‑context input
	- Needle — extractive Q/A with citation normalization and optional one‑sentence trimming (`RAG_TRIM_ANSWERS`)
	- Table — natural value lookup from tables/figures; also covers instrumentation/threshold inventory
- Orchestrator: optional `agent_orchestrator.run(...)` to plan/retrieve/answer with a trace payload

## Normalized artifacts and graph

- Normalized mode: `RAG_USE_NORMALIZED=1` to index from `logs/normalized/chunks.jsonl` if present
- Normalized graph: `RAG_USE_NORMALIZED_GRAPH=1`; optional import: `RAG_IMPORT_NORMALIZED_GRAPH=1`
- Neo4j population from current docs: `RAG_GRAPH_DB=1` (default on). Import helpers in `graphdb_import_normalized.py`.
- LlamaIndex export/compare: set `RAG_ENABLE_LLAMAINDEX=1` to export and A/B alternative indexes

## Evaluation

- Files auto‑discovered (if not overridden via `RAG_QA_FILE`/`RAG_GT_FILE`):
	- `data/gear_wear_qa_context_free.jsonl`
	- `data/gear_wear_ground_truth_context_free.json`
- Supports multiple QA/GT schemas; flexible loaders join by id or by matching question text.
- Metrics: RAGAS (faithfulness, relevancy, context precision/recall, table QA accuracy) + optional DeepEval.
- Outputs: `logs/eval_ragas_summary.json`, `logs/eval_ragas_per_question.jsonl`, and optional DeepEval artifacts.

## Environment variables (common)

- General: `RAG_HEADLESS`, `RAG_EVAL`, `RAG_UI_ONLY`, `RAG_CLEAN_RUN`
- Chunking: `RAG_TEXT_SPLIT_MULTI`, `RAG_SEMANTIC_CHUNKING`, `RAG_TEXT_TARGET_TOKENS`, `RAG_TEXT_MAX_TOKENS`, `RAG_TEXT_OVERLAP_SENTENCES`
- Parsing/extraction: `RAG_USE_PDFPLUMBER`, `RAG_USE_TABULA`, `RAG_USE_CAMELOT`, `RAG_EXTRACT_IMAGES`
- Retrieval: `RAG_VECTOR_BACKEND`, `RAG_CHROMA_DIR`, `RAG_SPARSE_WEIGHT`, `RAG_DENSE_WEIGHT`, `RAG_USE_CE_RERANKER`, `RAG_MIN_CTX_SCORE`
- Router/LLM: `RAG_USE_LLM_ROUTER`, `GOOGLE_CHAT_MODEL`, `OPENAI_CHAT_MODEL`, `FORCE_OPENAI_ONLY`
- Eval: `RAG_QA_FILE`, `RAG_GT_FILE`, `RAG_DEEPEVAL`, `RAGAS_LLM_PROVIDER`
- Misc: `RAG_TRIM_ANSWERS`, `RAG_EXTRACTIVE_FORCE`, `RAG_TRACE*` toggles for debug

## Logs and artifacts

- `logs/db_snapshot.jsonl` — compact DB view with stable IDs and previews
- `logs/db_snapshot_full.jsonl` — full text + metadata for UI reuse
- `logs/chroma_snapshot*.jsonl` — vector store dump of ids/metadata/previews
- `logs/graph.html` — quick graph visualization (optional)
- `logs/queries.jsonl` — per‑question run logs with routes and contexts

## Tests and smoke checks

- Run tests:

```powershell
python -m pytest -q
```

Key suites: `tests/test_all.py`, `tests/smoke_chunking_test.py`, `tests/test_extractors.py`, `tests/test_table_filter_duplicates.py`.

## Troubleshooting

- No contexts found or low recall: relax filters (drop `section`, numbers) or lower `RAG_MIN_CTX_SCORE`.
- Table extraction flaky: install Java (Tabula) and Ghostscript/Poppler (Camelot), or enable `RAG_USE_PDFPLUMBER`.
- UI doesn’t launch: ensure `RAG_HEADLESS=0`; check port (`GRADIO_PORT`), and that ingestion succeeded.
- No embeddings backend: set `GOOGLE_API_KEY` or `OPENAI_API_KEY`; dev fallback uses FakeEmbeddings if available.
- Neo4j import: confirm Neo4j is reachable and credentials are configured in environment.

## License

Provided for educational and research purposes. See repository for details.

