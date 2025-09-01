#!/usr/bin/env python3

from app.chunking import structure_chunks
from app.loaders import load_elements
from pathlib import Path

# Load the document and check figure descriptions
doc_path = Path('Gear wear Failure.pdf')
elements = load_elements(doc_path)
chunks = structure_chunks(elements, str(doc_path))

# Look specifically for Figure 3 and related content
print('Checking Figure 3 and FFT-related content...')
for chunk in chunks:
    content = chunk.get('content', '')
    if 'Figure 3' in content or ('fft' in content.lower() and chunk.get('section_type') == 'Figure'):
        print(f'Found Figure 3 chunk (type: {chunk.get("section_type", "unknown")}):')
        print(content)
        print('=' * 60)
        
# Also check for any text chunks that mention FFT with more context
print('\nLooking for detailed FFT descriptions in text chunks...')
for chunk in chunks:
    content = chunk.get('content', '')
    if 'fft' in content.lower() and len(content) > 100:
        print(f'FFT context chunk (type: {chunk.get("section_type", "unknown")}):')
        print(content[:500] + '...' if len(content) > 500 else content)
        print('-' * 40)
