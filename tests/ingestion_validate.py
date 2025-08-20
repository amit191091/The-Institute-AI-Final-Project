import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.loaders import load_many
from app.chunking import structure_chunks
from app.metadata import attach_metadata


def validate_pdf(path: Path):
    els = None
    for p, e in load_many([path]):
        els = e
    assert els is not None
    cats = [str(getattr(e, "category", "")).lower() for e in els]
    assert "text" in set(cats)
    chunks = structure_chunks(els, str(path))
    docs = [attach_metadata(c) for c in chunks]
    # Ensure figures or tables present when toggles enabled
    if os.getenv("RAG_EXTRACT_IMAGES", "").lower() in ("1", "true", "yes"):
        assert any((d["metadata"] or {}).get("section") == "Figure" for d in docs), "No figures found"
    if any(os.getenv(v, "").lower() in ("1", "true", "yes") for v in ["RAG_USE_TABULA", "RAG_USE_PDFPLUMBER", "RAG_USE_CAMELOT", "RAG_SYNTH_TABLES"]):
        assert any((d["metadata"] or {}).get("section") == "Table" for d in docs), "No tables found"


if __name__ == "__main__":
    p = Path("Gear wear Failure.pdf")
    assert p.exists(), "PDF not found"
    validate_pdf(p)
    print("Ingestion validation: PASS")
