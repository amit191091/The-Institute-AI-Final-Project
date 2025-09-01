#!/usr/bin/env python3

from app.chunking import structure_chunks
from app.loaders import load_elements
from app.metadata import attach_metadata
from app.indexing import to_documents
from pathlib import Path

# Debug figure content in documents
doc_path = Path('Gear wear Failure.pdf')
elements = load_elements(doc_path)
chunks = structure_chunks(elements, str(doc_path))

print("Debug: Figure chunks content...")
print("="*60)

# Convert to documents
records = []
for ch in chunks:
    rec = attach_metadata(ch)
    records.append(rec)
docs = to_documents(records)

# Find figure documents
figure_docs = [d for d in docs if d.metadata.get("section_type") == "Figure"]
print(f"Found {len(figure_docs)} figure documents")

for i, doc in enumerate(figure_docs):
    print(f"\nFigure document {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print("-"*40)
