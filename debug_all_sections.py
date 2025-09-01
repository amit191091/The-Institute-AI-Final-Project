#!/usr/bin/env python3

from app.chunking import structure_chunks
from app.loaders import load_elements
from app.metadata import attach_metadata
from app.indexing import to_documents
from pathlib import Path

# Debug all section types
doc_path = Path('Gear wear Failure.pdf')
elements = load_elements(doc_path)
chunks = structure_chunks(elements, str(doc_path))

print("Debug: All section types and figure-related content...")
print("="*60)

# Convert to documents
records = []
for ch in chunks:
    rec = attach_metadata(ch)
    records.append(rec)
docs = to_documents(records)

# Check all section types
section_types = {}
for doc in docs:
    section_type = doc.metadata.get("section_type", "Unknown")
    section_types[section_type] = section_types.get(section_type, 0) + 1

print("Section types found:")
for section_type, count in section_types.items():
    print(f"  {section_type}: {count}")

print("\nLooking for any content with 'Figure' keyword...")
for i, doc in enumerate(docs):
    content = doc.page_content or ""
    if "figure" in content.lower() or "fig" in content.lower():
        print(f"\nDocument {i+1} (section_type: {doc.metadata.get('section_type')}):")
        print(f"Content: {content[:300]}...")
        print(f"Metadata keys: {list(doc.metadata.keys())}")
        print("-"*40)
