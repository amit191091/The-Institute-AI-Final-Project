#!/usr/bin/env python3

from app.pipeline import ask
from pathlib import Path

# Test the enhanced figure retrieval with an FFT-related query
doc_path = Path('Gear wear Failure.pdf')

print("Testing FFT figure query with enhanced contextual information...")
print("="*80)

response = ask(
    question="show me the figure that is fft related and explain what it shows",
    source_document_id=str(doc_path),
    conversation_context={}
)

print("Response:")
print(response)
print("="*80)
