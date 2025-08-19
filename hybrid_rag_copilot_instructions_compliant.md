# Metadata-Driven Hybrid RAG for Failure Analysis Reports (Copilot Instructions) — **Compliant Edition**

> **Goal:** Build a **lean, readable, simple, reliable, and impressive** Python system for a **metadata-driven Hybrid RAG** that analyzes *stress/failure analysis* reports (bearings/gears) across **multiple documents**.  
> **Stack:** Python · LangChain · FAISS/Chroma (dense) · BM25 (sparse) · Optional Gradio UI  
> **Docs:** Start with **PDF**; support **DOC/DOCX/TXT**.  
> **Design Targets:** Hybrid retrieval (**Dense K~10 U Sparse K~10**) -> **Rerank K~20** -> pass **6-8** final contexts to LLM.  
> **Compliance:** This spec implements all PDF requirements (parsing/chunking, anchors, >=5 metadata fields, indexing, filters, hybrid retrieval, reranker, multi-doc, router+3 sub-agents, data criteria, evaluation metrics/thresholds).

---

## ✅ Compliance Map (What this file guarantees)

- **Parsing & Chunking**
  - Split by natural structure: **sections, titles, tables, figures, appendices**.
  - **Distillation**: each chunk holds about **5%** core information of its source segment.
  - **Anchors saved per chunk**: `PageNumber`, `SectionType`, `TableId/FigureId` (when applicable), **row/column position** for table coverage.
- **Metadata (>=5 fields per chunk)** — Always includes: `FileName`, `PageNumber`, `SectionType`, `ChunkSummary`, `Keywords`, plus conditional fields: `CriticalEntities`, `IncidentType`, `IncidentDate`, `AmountRange`, `TableId/FigureId`, `ClientId/CaseId`.
- **Indexing**
  - Tables -> **CSV/Markdown** + **short textual summary**.
  - Save anchors (`TableId`, `PageNumber`) & metadata alongside vectors for **filtering**.
- **Hybrid Retrieval**
  - **Query analysis** -> extract **keywords, entities, dates** -> build **metadata filters**.
  - Dense (**K~10**) U Sparse (**K~10**) -> **metadata filtering** -> **Reranker** (**K~20**) -> LLM (**6-8 contexts**).
- **Multi-Document Support**
  - Use `ClientId/CaseId` to link docs; support **all files** or **single doc** retrieval.
- **Agents (Router + 3 Sub-Agents)**
  - **Summary Agent**, **Needle Agent** (anchor/subsection), **Table-QA Agent** (quant/table Qs).
- **Data Criteria (ingestion checks)**
  - **Doc length >= 10 pages** (warn/fail otherwise).
  - **Chunk size**: **avg 250-500 tokens**, **max 800** for table/figure chunks.
  - **Anchors mandatory**: `PageNumber`, `SectionType`; include `TableId/FigureId` if available.
  - Recommended content structure (headers/subheaders, **Timeline** with timestamps, **Measurements** with units, **Figures** with textual description).
- **Evaluation Targets**
  - **Answer Correctness** (context-aware)
  - **Context Precision >= 0.75**, **Recall >= 0.70**, **Faithfulness >= 0.85**
  - **Table-QA Accuracy >= 0.90**

---

## 📁 Suggested Project Layout

```txt
rag_failure_reports/
  ├─ app/
  │   ├─ __init__.py
  │   ├─ config.py
  │   ├─ loaders.py          # PDF/DOCX/TXT loaders
  │   ├─ chunking.py         # structure-aware splitting + distillation + token budget
  │   ├─ metadata.py         # schema + keywords/entities/incidents + table cell anchors
  │   ├─ indexing.py         # embeddings, FAISS/Chroma, BM25
  │   ├─ retrieve.py         # query analysis + hybrid retrieval + metadata filters + reranker
  │   ├─ agents.py           # router + summary/needle/table agents
  │   ├─ prompts.py          # prompt templates
  │   ├─ validate.py         # ingestion/data criteria checks
  │   ├─ eval_ragas.py       # evaluation harness + metrics
  │   ├─ utils.py
  │   └─ ui_gradio.py        # optional UI
  ├─ tests/
  │   └─ test_ingest_retrieve_eval.py
  ├─ main.py
  ├─ requirements.txt
  └─ README.md
```

---

## 🧱 Config

```python
# app/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    DATA_DIR: Path = Path("data")
    INDEX_DIR: Path = Path("index")
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # or local
    DENSE_K: int = 10
    SPARSE_K: int = 10
    RERANK_TOP_K: int = 20
    CONTEXT_TOP_N: int = 8
    CHUNK_TOK_AVG_RANGE: tuple[int, int] = (250, 500)  # avg target
    CHUNK_TOK_MAX: int = 800                           # hard cap for tables/figures
    MIN_PAGES: int = 10                                # ingestion rule
settings = Settings()
```

---

## 📥 Loaders (PDF/DOCX/TXT)

```python
# app/loaders.py
from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text

def load_elements(path: Path):
    """Return Unstructured elements for PDF/DOCX/TXT with page metadata kept."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return partition_pdf(filename=str(path))
    if ext in (".docx", ".doc"):
        return partition_docx(filename=str(path))
    if ext in (".txt",):
        return partition_text(filename=str(path))
    raise ValueError(f"Unsupported format: {ext}")

def load_many(paths: List[Path]):
    for p in paths:
        yield p, load_elements(p)
```

---

## ✂️ Structure-Aware Chunking, Distillation & Token Budget

```python
# app/chunking.py
from typing import Dict, List, Optional, Tuple
from app.utils import simple_summarize, approx_token_len, truncate_to_tokens
from app.metadata import classify_section_type, extract_keywords

def structure_chunks(elements, file_path: str) -> List[Dict]:
    """
    Split by natural order: Title/Text, Table, Figure/Image, Appendix.
    Distill each chunk to ~5% core info from its element.
    Save anchors: PageNumber, SectionType, TableId/FigureId (if any), and row/column position for tables.
    Enforce token budgets: avg 250-500 tokens; max 800 for table/figure chunks.
    """
    chunks: List[Dict] = []
    for el in elements:
        kind = getattr(el, "category", getattr(el, "type", "Text"))
        page = (getattr(el, "metadata", None) or {}).get("page_number")
        anchor = (getattr(el, "metadata", None) or {}).get("id")
        raw_text = (getattr(el, "text", "") or "").strip()

        # Section type mapping per spec
        section_type = classify_section_type(kind, raw_text)  # -> Summary/Timeline/Table/Figure/Analysis/Conclusion/Text

        if kind.lower() == "table":
            as_text = raw_text
            distilled = simple_summarize(as_text, ratio=0.05)
            # Table cell anchors: store row/col ranges if detectable
            row_range: Optional[Tuple[int,int]] = (0, 0)   # Copilot: infer from parser when possible
            col_names: Optional[list[str]] = []            # Copilot: fill if headers detected
            content = f"[TABLE]\nSUMMARY:\n{distilled}\nRAW:\n{as_text}"
            tok = approx_token_len(content)
            if tok > 800: content = truncate_to_tokens(content, 800)
            chunks.append({
                "file_name": file_path,
                "page": page,
                "section_type": section_type or "Table",
                "anchor": anchor or None,
                "table_row_range": row_range,
                "table_col_names": col_names,
                "content": content.strip(),
                "keywords": extract_keywords(content),
            })
            continue

        if kind.lower() in ("figure", "image"):
            caption = raw_text or "Figure"
            distilled = simple_summarize(caption, ratio=0.5)
            content = f"[FIGURE]\n{distilled}"
            tok = approx_token_len(content)
            if tok > 800: content = truncate_to_tokens(content, 800)
            chunks.append({
                "file_name": file_path,
                "page": page,
                "section_type": section_type or "Figure",
                "anchor": anchor or None,
                "content": content.strip(),
                "keywords": extract_keywords(content),
            })
            continue

        # Textual sections (headers/paragraphs/timeline/analysis/conclusion)
        content = simple_summarize(raw_text, ratio=0.05)
        chunks.append({
            "file_name": file_path,
            "page": page,
            "section_type": section_type or "Text",
            "anchor": anchor or None,
            "content": truncate_to_tokens(content, 500).strip(),
            "keywords": extract_keywords(content),
        })

    return chunks
```

---

## 🧾 Metadata Schema (>=5 fields) + Entities/Incidents/Amounts

```python
# app/metadata.py
from typing import Dict, List, Optional
import re

SECTION_ENUM = {"Summary","Timeline","Table","Figure","Analysis","Conclusion","Text"}

def classify_section_type(kind: str, text: str) -> str:
    k = (kind or "").lower()
    t = (text or "").lower()
    if "timeline" in t: return "Timeline"
    if "conclusion" in t or "summary" in t: return "Conclusion" if "conclusion" in t else "Summary"
    if k == "table": return "Table"
    if k in ("figure","image"): return "Figure"
    if any(w in t for w in ("analysis","method","procedure","results","discussion")): return "Analysis"
    return "Text"

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    # Copilot: simple RAKE-like or tf-idf-ish fallback using word frequency; strip stopwords
    words = re.findall(r"[A-Za-z0-9°%]+", text)
    return [w for w in words[:top_n]]

def extract_entities(text: str) -> List[str]:
    """Light entity extraction for bearings/gears/testing context."""
    pats = [
        r"\bCASE[-_ ]?\d+\b", r"\bCLIENT[-_ ]?\w+\b", r"\bBRG[-_ ]?\w+\b",
        r"\bGEAR[-_ ]?\w+\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d+(?:\.\d+)?\s?(MPa|RPM|°C|N|kN|mm|Hz|MPH|kW)\b"
    ]
    out = []
    for p in pats: out += re.findall(p, text)
    flat = [(x if isinstance(x,str) else x[0]) for x in out]
    return sorted(set(flat))

def extract_incident(text: str) -> Dict[str, Optional[str]]:
    itype = None
    if re.search(r"\bfail(ure|ed)|fracture|fatigue|overheat|seiz(e|ure)\b", text, re.I):
        itype = "Failure"
    idate = None
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if m: idate = m.group(1)
    amount_range = None
    m2 = re.findall(r"(\d+(?:\.\d+)?)\s?(MPa|RPM|°C|N|kN|mm)", text)
    if m2:
        nums = [float(x[0]) for x in m2]
        amount_range = f"{min(nums)}-{max(nums)} {m2[0][1]}"
    return {"IncidentType": itype, "IncidentDate": idate, "AmountRange": amount_range}

def attach_metadata(chunk: Dict, client_id: str = None, case_id: str = None) -> Dict:
    ents = extract_entities(chunk["content"])
    inc = extract_incident(chunk["content"])
    metadata = {
        "file_name": chunk["file_name"],
        "page": chunk.get("page"),
        "section": chunk.get("section_type"),
        "anchor": chunk.get("anchor"),
        "table_row_range": chunk.get("table_row_range"),
        "table_col_names": chunk.get("table_col_names"),
        "client_id": client_id,
        "case_id": case_id,
        "keywords": chunk.get("keywords", []),
        "critical_entities": ents,
        "chunk_summary": chunk["content"].splitlines()[0][:200],
        "incident_type": inc["IncidentType"],
        "incident_date": inc["IncidentDate"],
        "amount_range": inc["AmountRange"],
    }
    return {"page_content": chunk["content"], "metadata": metadata}
```

---

## 🔎 Indexing: Dense + Sparse + Table Summaries

```python
# app/indexing.py
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS  # or Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever

def to_documents(records: List[dict]) -> List[Document]:
    return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in records]

def build_dense_index(docs: List[Document]):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_sparse_retriever(docs: List[Document], k: int = 10):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
```

---

## 🧠 Query Analysis -> Filters

```python
# app/retrieve.py (part 1)
import re
from typing import Dict, List, Optional
from langchain.schema import Document

def query_analyzer(q: str) -> Dict:
    """Extract keywords, case/client IDs, dates to build metadata filters."""
    filt = {}
    mcase = re.search(r"(?:case|id)[:\-\s]*([A-Za-z0-9_-]+)", q, re.I)
    if mcase: filt["case_id"] = mcase.group(1)
    mclient = re.search(r"(?:client)[:\-\s]*([A-Za-z0-9_-]+)", q, re.I)
    if mclient: filt["client_id"] = mclient.group(1)
    mdate = re.search(r"(20\d{2}-\d{2}-\d{2})", q)
    if mdate: filt["incident_date"] = mdate.group(1)
    if "table" in q.lower(): filt["section"] = "Table"
    return {"filters": filt, "keywords": re.findall(r"[A-Za-z0-9°%]+", q)[:10]}

def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
    if not filters: return docs
    def ok(meta):
        return all(meta.get(k) == v for k, v in filters.items())
    return [d for d in docs if ok(d.metadata)]
```

---

## 🧬 Hybrid Retrieval + Metadata Filtering + Reranker

```python
# app/retrieve.py (part 2)
from langchain.retrievers import EnsembleRetriever

def build_hybrid_retriever(dense_store, sparse_retriever, dense_k=10):
    dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
    return EnsembleRetriever(retrievers=[sparse_retriever, dense])

def lexical_overlap(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def rerank_candidates(query: str, candidates: List[Document], top_n: int = 8) -> List[Document]:
    """Deterministic reranker.
score = lexical overlap + entity/keyword boost; prefer shorter chunks on ties."""
    kws = set(re.findall(r"[A-Za-z0-9°%]+", query.lower()))
    scored = []
    for d in candidates:
        base = lexical_overlap(query, d.page_content)
        meta = " ".join(map(str, d.metadata.values()))
        boost = 0.2 * lexical_overlap(" ".join(kws), meta)
        score = base + boost
        scored.append((score, len(d.page_content), d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [d for _,__,d in scored[:top_n]]
```

---

## 🧠 Prompts and Agents

```python
# app/prompts.py
SUMMARY_SYSTEM = """You summarize engineering failure reports concisely and factually."""
NEEDLE_SYSTEM  = """You answer with precise values and citations from provided context only."""
TABLE_SYSTEM   = """You answer questions about tables. If unsure, say so and ask for clarification."""

SUMMARY_PROMPT = """Context:\n{context}\n\nTask: Provide a concise, technical summary answering: {question}\n"""
NEEDLE_PROMPT  = """Use only the context. Cite file_name and page if available.\nContext:\n{context}\n\nQ: {question}\nA:"""
TABLE_PROMPT   = """You are given table/figure contexts and a question.\nContext:\n{table}\n\nQ: {question}\nA:"""
```

```python
# app/agents.py
from typing import List
from langchain.schema import Document
from app.prompts import SUMMARY_SYSTEM, NEEDLE_SYSTEM, TABLE_SYSTEM, SUMMARY_PROMPT, NEEDLE_PROMPT, TABLE_PROMPT

def route_question(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ("summarize", "overview", "overall", "conclusion")):
        return "summary"
    if "table" in ql or "chart" in ql or "value" in ql or "figure" in ql:
        return "table"
    return "needle"

def render_context(docs: List[Document], max_chars: int = 8000) -> str:
    out, n = [], 0
    for d in docs:
        piece = f"[{d.metadata.get('file_name')} p{d.metadata.get('page')}] {d.page_content}".strip()
        n += len(piece)
        if n > max_chars: break
        out.append(piece)
    return "\n\n".join(out)

def answer_summary(llm, docs: List[Document], question: str) -> str:
    ctx = render_context(docs)
    return llm(SUMMARY_SYSTEM + "\n" + SUMMARY_PROMPT.format(context=ctx, question=question)).strip()

def answer_needle(llm, docs: List[Document], question: str) -> str:
    ctx = render_context(docs)
    return llm(NEEDLE_SYSTEM + "\n" + NEEDLE_PROMPT.format(context=ctx, question=question)).strip()

def answer_table(llm, docs: List[Document], question: str) -> str:
    table_docs = [d for d in docs if d.metadata.get("section") in ("Table","Figure")] or docs
    ctx = render_context(table_docs)
    return llm(TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)).strip()
```

---

## 🛡️ Ingestion Validation (Data Criteria)

```python
# app/validate.py
from typing import Iterable, Tuple

def validate_min_pages(num_pages: int, min_pages: int = 10) -> Tuple[bool, str]:
    if num_pages < min_pages:
        return False, f"Document has {num_pages} pages; requires >= {min_pages}."
    return True, "OK"

def validate_chunk_tokens(tok_counts: list[int], avg_range=(250,500), max_tok=800) -> Tuple[bool, str]:
    avg = sum(tok_counts)/max(1,len(tok_counts))
    if not (avg_range[0] <= avg <= avg_range[1]):
        return False, f"Avg tokens {avg:.1f} not in {avg_range}."
    if any(t > max_tok for t in tok_counts):
        return False, f"One or more chunks exceed max {max_tok} tokens."
    return True, "OK"
```

---

## 🧪 Evaluation (RAGAS & Targets)

```python
# app/eval_ragas.py
# pip install ragas datasets evaluate
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# Table-QA Accuracy can be computed via task-specific unit tests comparing extracted numeric answers.

def run_eval(dataset):
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

TARGETS = {
    "context_precision": 0.75,
    "context_recall": 0.70,
    "faithfulness": 0.85,
    "table_qa_accuracy": 0.90,
}
```

---

## 🧪 End-to-End Wiring

```python
# main.py
from pathlib import Path
from app.config import settings
from app.loaders import load_many
from app.chunking import structure_chunks
from app.metadata import attach_metadata
from app.indexing import to_documents, build_dense_index, build_sparse_retriever
from app.retrieve import build_hybrid_retriever, rerank_candidates, query_analyzer, apply_filters
from app.agents import route_question, answer_summary, answer_needle, answer_table

def build_pipeline(paths):
    records = []
    for path, elements in load_many(paths):
        chunks = structure_chunks(elements, str(path))
        for ch in chunks:
            rec = attach_metadata(ch)  # TODO: pass client_id/case_id if known
            records.append(rec)
    docs = to_documents(records)
    dense = build_dense_index(docs)
    sparse = build_sparse_retriever(docs)
    hybrid = build_hybrid_retriever(dense, sparse, dense_k=settings.DENSE_K)
    return docs, hybrid

def ask(docs, hybrid, llm, question: str):
    qa = query_analyzer(question)
    candidates = hybrid.get_relevant_documents(question)  # dense(10) U sparse(10)
    # metadata filtering
    filtered = apply_filters(candidates, qa["filters"])
    top_docs = rerank_candidates(question, filtered, top_n=settings.CONTEXT_TOP_N)
    route = route_question(question)
    if route == "summary":
        return answer_summary(llm, top_docs, question)
    if route == "table":
        return answer_table(llm, top_docs, question)
    return answer_needle(llm, top_docs, question)

if __name__ == "__main__":
    paths = [Path("data/report1.pdf"), Path("data/report2.docx")]
    docs, hybrid = build_pipeline(paths)

    class EchoLLM:
        def __call__(self, prompt: str) -> str:
            return "[LLM output placeholder]\n" + prompt[-400:]

    llm = EchoLLM()
    print(ask(docs, hybrid, llm, "Summarize the failure modes for bearing ABC across all tests."))
```

---

## 🧪 Tests (chunking/metadata/retrieval/validation)

```python
# tests/test_ingest_retrieve_eval.py
from app.metadata import attach_metadata, extract_entities, extract_incident
from app.validate import validate_chunk_tokens

def test_metadata_minimum_fields():
    ch = {"file_name":"x.pdf","page":2,"section_type":"Text","anchor":None,"content":"Bearing ABC failed at 120 MPa on 2023-01-10","keywords":["Bearing","ABC"]}
    rec = attach_metadata(ch, client_id="CL-1", case_id="CASE-1")
    m = rec["metadata"]
    assert m["file_name"] and m["page"] is not None and m["section"]
    assert m["chunk_summary"] and len(m["keywords"]) >= 1
    assert "critical_entities" in m

def test_chunk_token_budget():
    toks = [300, 420, 510, 700, 790]  # one near cap
    ok, _ = validate_chunk_tokens(toks, avg_range=(250,500), max_tok=800)
    assert not ok  # avg exceeds; ensure validator catches
```

---

## 🖥️ Optional: Minimal Gradio UI

```python
# app/ui_gradio.py
import gradio as gr
from app.retrieve import rerank_candidates, query_analyzer, apply_filters
from app.agents import route_question, answer_summary, answer_needle, answer_table

def build_ui(docs, hybrid, llm):
    def on_ask(q):
        qa = query_analyzer(q)
        cands = hybrid.get_relevant_documents(q)
        filtered = apply_filters(cands, qa["filters"])
        top_docs = rerank_candidates(q, filtered, top_n=8)
        r = route_question(q)
        if r == "summary": return answer_summary(llm, top_docs, q)
        if r == "table":   return answer_table(llm, top_docs, q)
        return answer_needle(llm, top_docs, q)
    return gr.Interface(fn=on_ask, inputs="text", outputs="text", title="Hybrid RAG – Failure Reports")
```

---

## ✅ Acceptance Checklist (for Copilot)

- [ ] **Parsing/Chunking**: natural structure; **~5% distillation**; anchors (`PageNumber`, `SectionType`, `TableId/FigureId`, table **row/col**).
- [ ] **Metadata (>=5 fields)**: include `FileName`, `PageNumber`, `SectionType`, `ChunkSummary`, `Keywords`; conditional `CriticalEntities`, `IncidentType`, `IncidentDate`, `AmountRange`, `TableId/FigureId`, `ClientId/CaseId`.
- [ ] **Indexing**: tables -> CSV/Markdown + summary; save anchors + metadata with vectors.
- [ ] **Hybrid Retrieval**: **query analysis -> filters**; **Dense K~10 U Sparse K~10** -> **filter** -> **Rerank K~20** -> **6-8** to LLM.
- [ ] **Multi-Doc**: retrieve across files; filter by `ClientId/CaseId`; single-doc or all files.
- [ ] **Agents**: Router -> **Summary / Needle / Table-QA**.
- [ ] **Data Criteria**: doc length >=10 pages; chunk avg **250-500** tokens; max **800** (tables/figures); anchors mandatory; structure hints kept.
- [ ] **Evaluation**: compute **Context Precision >=0.75**, **Context Recall >=0.70**, **Faithfulness >=0.85**; **Table-QA Accuracy >=0.90** (unit tests).

---

## 🔧 Notes for Copilot

- Keep modules **small**, functions **pure**, and **typed**.
- Reranker: start **deterministic**; make it **swappable**.
- Prompts: **concise**; ground answers **only** in provided context. If unsure, ask for clarification.
- Respect token budgets: **truncate** contexts before prompting.
