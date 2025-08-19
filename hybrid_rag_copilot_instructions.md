# Metadata-Driven Hybrid RAG for Failure Analysis Reports (Copilot Instructions)

> **Goal:** Guide GitHub Copilot to generate a **lean, readable, simple, reliable, and impressive** Python implementation of a **metadata-driven Hybrid RAG** (dense+sparse) for **stress/failure analysis reports** (bearings/gears).  
> **Stack:** Python · LangChain · FAISS/Chroma (dense) · BM25 (sparse) · Gradio UI  
> **Docs:** PDF first; also DOC/DOCX/TXT (reasonable report formats)

---

## 🚦 Engineering Constraints for Copilot

- Prefer **pure functions**, clear **docstrings**, and **type hints**.
- Minimal dependencies; keep modules **small and modular**.
- Chunk **by structure** (sections/tables/figures) and **distill to ~5%** core info.
- Every chunk carries **rich metadata** (anchors, page, section type, client/case IDs, etc.).
- **Hybrid retrieval**: Dense (k≈10) ∪ Sparse/BM25 (k≈10) → **Rerank** (k≈20) → Top 6–8 to LLM.
- **Agents**: Router → {Summary, Needle (exact lookup), Table-QA}.
- Write **tests** for split/distill/metadata and retriever fusion.

---

## 📁 Suggested Project Layout

```txt
rag_failure_reports/
  ├─ app/
  │   ├─ __init__.py
  │   ├─ config.py
  │   ├─ loaders.py          # PDF/DOCX/TXT loaders
  │   ├─ chunking.py         # structure-aware splitting + distillation
  │   ├─ metadata.py         # schema + entity extraction
  │   ├─ indexing.py         # embeddings, FAISS/Chroma, BM25
  │   ├─ retrieve.py         # hybrid retrieval + rerank
  │   ├─ agents.py           # router + summary/needle/table agents
  │   ├─ prompts.py          # prompt templates
  │   ├─ utils.py
  │   └─ ui_gradio.py        # optional UI
  ├─ tests/
  │   └─ test_chunking_and_retrieval.py
  ├─ main.py
  ├─ requirements.txt
  └─ README.md
```

---

## 🧱 Config (keep it simple)

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

settings = Settings()
```

---

## 📥 Loaders (PDF first, then DOCX/TXT)

```python
# app/loaders.py
from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text

def load_elements(path: Path):
    """Return Unstructured elements for PDF/DOCX/TXT."""
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

## ✂️ Structure-Aware Chunking + 5% Distillation

```python
# app/chunking.py
from typing import Dict, List
from app.utils import simple_summarize

def structure_chunks(elements, file_path: str) -> List[Dict]:
    """
    Split by natural order: Text/Title, Table, Figure/Image, etc.
    Distill each chunk to ~5% core info.
    Save anchors: page, section_type, figure/table id, etc.
    """
    chunks: List[Dict] = []
    for el in elements:
        kind = getattr(el, "category", getattr(el, "type", "Text"))
        page = (getattr(el, "metadata", None) or {}).get("page_number")
        anchor = (getattr(el, "metadata", None) or {}).get("id")

        if kind.lower() == "table":
            # Convert table to CSV-like text + short summary for semantic search
            as_text = getattr(el, "text", "") or ""
            distilled = simple_summarize(as_text, ratio=0.05)
            content = f"[TABLE]\nSUMMARY:\n{distilled}\nRAW:\n{as_text}"
            section_type = "Table"
        elif kind.lower() in ("figure", "image"):
            caption = getattr(el, "text", "") or "Figure"
            distilled = simple_summarize(caption, ratio=0.5)
            content = f"[FIGURE]\n{distilled}"
            section_type = "Figure"
        else:
            text = getattr(el, "text", "") or ""
            content = simple_summarize(text, ratio=0.05)
            section_type = "Text"

        chunks.append({
            "file_name": file_path,
            "page": page,
            "section_type": section_type,
            "anchor": anchor,
            "content": content.strip(),
        })
    return chunks
```

**Copilot, implement `simple_summarize(text: str, ratio: float) -> str` as a tiny, dependency-light extractor (e.g., pick top sentences), with a fallback to first N sentences.**

---

## 🧾 Metadata Schema + Entities

```python
# app/metadata.py
from typing import Dict, List
import re

def extract_entities(text: str) -> List[str]:
    """Very light entity extraction: bearing IDs, case IDs, test IDs, dates, units."""
    # Copilot: include regex for IDs like CASE-123, BRG-45, dates YYYY-MM-DD, numbers with units (MPa, RPM, °C)
    patterns = [
        r"\bCASE[-_ ]?\d+\b", r"\bBRG[-_ ]?\w+\b",
        r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d+(\.\d+)?\s?(MPa|RPM|°C|N|kN|mm)\b"
    ]
    found = []
    for pat in patterns:
        found += re.findall(pat, text)
    # flatten tuples and dedup
    flat = {x if isinstance(x, str) else x[0] for x in found}
    return list(flat)

def attach_metadata(chunk: Dict, client_id: str = None, case_id: str = None) -> Dict:
    """Return LangChain Document-like dict with page_content + metadata."""
    ents = extract_entities(chunk["content"])
    return {
        "page_content": chunk["content"],
        "metadata": {
            "file_name": chunk["file_name"],
            "page": chunk.get("page"),
            "section": chunk.get("section_type"),
            "anchor": chunk.get("anchor"),
            "client_id": client_id,
            "case_id": case_id,
            "entities": ents,
            "chunk_summary": chunk["content"].splitlines()[0][:200],
        }
    }
```

---

## 🔎 Indexing: Dense (FAISS/Chroma) + Sparse (BM25)

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
    embeddings = OpenAIEmbeddings()  # Copilot: make model configurable
    return FAISS.from_documents(docs, embeddings)

def build_sparse_retriever(docs: List[Document], k: int = 10):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
```

---

## 🧬 Hybrid Retrieval + Reranking

```python
# app/retrieve.py
from typing import List
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever

def build_hybrid_retriever(dense_store, sparse_retriever, dense_k=10, sparse_k=10):
    dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
    # BM25 retriever already holds k
    return EnsembleRetriever(retrievers=[sparse_retriever, dense])

def rerank_candidates(query: str, candidates: List[Document], top_n: int = 8) -> List[Document]:
    """
    Copilot: Implement a simple reranker:
      1) lexical overlap score + entity overlap boost
      2) break ties by shorter chunks
      3) keep top_n
    Keep this deterministic and light.
    """
    # Placeholder baseline: keep first top_n
    return candidates[:top_n]
```

---

## 🗂️ Multi-Document Support (client_id / case_id)

- When ingesting docs, pass `client_id`/`case_id` to `attach_metadata`.
- At query time, detect hints like “case 123” and **filter**:
  - Dense: `vectorstore.similarity_search(query, k, filter={"case_id": "123"})`
  - Sparse: pre-filter docs by metadata (or keep one BM25 per case if preferred).

**Copilot, add a small `query_analyzer(query) -> dict`** that extracts `client_id`/`case_id` if present and returns a `filters` dict.

---

## 🧠 Prompts and Agents

```python
# app/prompts.py
SUMMARY_SYSTEM = """You summarize engineering failure reports concisely and factually."""
NEEDLE_SYSTEM  = """You answer with precise values and citations from provided context only."""
TABLE_SYSTEM   = """You answer questions about tables. If unsure, say so and ask for clarification."""

SUMMARY_PROMPT = """Context:
{context}

Task: Provide a concise, technical summary answering: {question}
"""
NEEDLE_PROMPT  = """Use only the context. Cite file_name and page if available.
Context:
{context}

Q: {question}
A:"""
TABLE_PROMPT   = """You are given a table (as CSV or text) and a question.
Table:
{table}

Q: {question}
A:"""
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
    if "table" in ql or "chart" in ql or "value" in ql:
        return "table"
    return "needle"

def render_context(docs: List[Document], max_chars: int = 8000) -> str:
    out, n = [], 0
    for d in docs:
        piece = f"[{d.metadata.get('file_name')} p{d.metadata.get('page')}] {d.page_content}".strip()
        n += len(piece)
        if n > max_chars: break
        out.append(piece)
    return "

".join(out)

def answer_summary(llm, docs: List[Document], question: str) -> str:
    ctx = render_context(docs)
    return llm(SUMMARY_SYSTEM + "
" + SUMMARY_PROMPT.format(context=ctx, question=question)).strip()

def answer_needle(llm, docs: List[Document], question: str) -> str:
    ctx = render_context(docs)
    return llm(NEEDLE_SYSTEM + "
" + NEEDLE_PROMPT.format(context=ctx, question=question)).strip()

def answer_table(llm, docs: List[Document], question: str) -> str:
    # Prefer table-only docs if available
    table_docs = [d for d in docs if d.metadata.get("section") == "Table"] or docs
    ctx = render_context(table_docs)
    return llm(TABLE_SYSTEM + "
" + TABLE_PROMPT.format(table=ctx, question=question)).strip()
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
from app.retrieve import build_hybrid_retriever, rerank_candidates
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
    hybrid = build_hybrid_retriever(dense, sparse)
    return hybrid

def ask(hybrid, llm, question: str):
    candidates = hybrid.get_relevant_documents(question)  # ≈ dense(10) ∪ sparse(10) fused
    top_docs = rerank_candidates(question, candidates, top_n=8)
    route = route_question(question)
    if route == "summary":
        return answer_summary(llm, top_docs, question)
    if route == "table":
        return answer_table(llm, top_docs, question)
    return answer_needle(llm, top_docs, question)

if __name__ == "__main__":
    paths = [Path("data/report1.pdf"), Path("data/report2.docx")]
    hybrid = build_pipeline(paths)

    # Copilot: inject a minimal LLM wrapper with `__call__(prompt:str)->str`
    class EchoLLM:
        def __call__(self, prompt: str) -> str:
            return "[LLM output placeholder]\n" + prompt[-400:]

    llm = EchoLLM()
    print(ask(hybrid, llm, "Summarize the failure modes for bearing ABC across all tests."))
```

---

## 🧪 Tests (critical parts)

```python
# tests/test_chunking_and_retrieval.py
from app.chunking import structure_chunks
from app.metadata import attach_metadata, extract_entities

def test_structure_and_metadata():
    fake_elements = [{"text":"Bearing ABC failed at 120 MPa on 2023-01-10", "type":"Text", "metadata":{"page_number": 2}}]
    # Copilot: adapt to unstructured element stub
    chunks = [{"file_name":"x.pdf", "page":2, "section_type":"Text", "anchor":None, "content":"Bearing ABC failed at 120 MPa on 2023-01-10"}]
    rec = attach_metadata(chunks[0])
    ents = rec["metadata"]["entities"]
    assert any("MPa" in e or "2023-01-10" in e for e in ents)
```

---

## 🖥️ Optional: Minimal Gradio UI

```python
# app/ui_gradio.py
import gradio as gr
from app.agents import route_question, answer_summary, answer_needle, answer_table
from app.retrieve import rerank_candidates

def build_ui(hybrid, llm):
    def on_ask(q):
        cands = hybrid.get_relevant_documents(q)
        top_docs = rerank_candidates(q, cands, top_n=8)
        r = route_question(q)
        if r == "summary": return answer_summary(llm, top_docs, q)
        if r == "table":   return answer_table(llm, top_docs, q)
        return answer_needle(llm, top_docs, q)
    return gr.Interface(fn=on_ask, inputs="text", outputs="text", title="Hybrid RAG – Failure Reports")

# In main.py, after building `hybrid` and `llm`:
# from app.ui_gradio import build_ui
# ui = build_ui(hybrid, llm)
# ui.launch()
```

---

## ✅ Acceptance Checklist (for Copilot)

- [ ] Parse PDF/DOCX/TXT into structured elements.
- [ ] Chunk by **natural structure**; **distill to ~5%** core.
- [ ] Save anchors: **page**, **section_type**, **anchor id**.
- [ ] Add metadata: **file_name**, **page**, **section**, **anchor**, **chunk_summary**, **entities**, **client_id/case_id**.
- [ ] Index: dense (FAISS/Chroma) + sparse (BM25).
- [ ] Hybrid retrieval: dense(k≈10) ∪ sparse(k≈10) → **rerank** → top 6–8.
- [ ] Agents: Router → **Summary / Needle / Table-QA**.
- [ ] Provide **tests** for chunking, metadata, retrieval fusion.
- [ ] Minimal **Gradio UI** (optional) for demo.

---

## 🔧 Notes for Copilot

- Keep code **short and clear**; prefer **no magic** and **few globals**.
- Reranker: start **deterministic** (lexical/entity overlap). Make it swappable.
- Prompts: **small and strict**. Always ground answers in provided context.
- If unsure, **say so**; suggest which file/page might hold the answer.

---

**End of Copilot Instruction MD**
