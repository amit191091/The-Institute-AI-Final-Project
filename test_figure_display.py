#!/usr/bin/env python3

from app.agents import route_question_ex, answer_figure_display
from app.chunking import structure_chunks
from app.loaders import load_elements
from app.metadata import attach_metadata
from app.indexing import to_documents
from pathlib import Path

# Test the new figure display routing and agent
print("Testing enhanced figure display system...")
print("="*60)

# Test routing
question = "show me figure 3"
route, trace = route_question_ex(question)
print(f"Question: '{question}'")
print(f"Route: {route}")
print(f"Trace: {trace}")
print()

# Test with document processing
doc_path = Path('Gear wear Failure.pdf')
elements = load_elements(doc_path)
chunks = structure_chunks(elements, str(doc_path))

# Convert to documents
records = []
for ch in chunks:
    rec = attach_metadata(ch)
    records.append(rec)
docs = to_documents(records)

# Test the figure display agent
print("Testing answer_figure_display...")
print("-"*40)

class MockLLM:
    def __call__(self, prompt):
        return "[Mock LLM response]"

mock_llm = MockLLM()
answer = answer_figure_display(mock_llm, docs, question)
print(f"Answer: {answer}")
print("="*60)
