#!/usr/bin/env python3

from app.chunking import structure_chunks
from app.loaders import load_elements
from pathlib import Path

# Load the document and test enhanced figure processing
doc_path = Path('Gear wear Failure.pdf')
elements = load_elements(doc_path)
chunks = structure_chunks(elements, str(doc_path))

# Look specifically for Figure 3 with enhanced context
print('Checking enhanced Figure 3 content...')
for chunk in chunks:
    content = chunk.get('content', '')
    if 'Figure 3' in content and chunk.get('section_type') == 'Figure':
        print(f'Enhanced Figure 3 chunk:')
        print(content)
        print('=' * 80)
        break
