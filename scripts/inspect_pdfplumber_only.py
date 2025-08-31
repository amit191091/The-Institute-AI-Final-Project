import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for 'app' package imports
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.loaders import load_elements

def main():
    # Force only pdfplumber path
    os.environ['RAG_USE_CAMELOT'] = '0'
    os.environ['RAG_USE_TABULA'] = '0'
    os.environ['RAG_USE_LLAMAPARSE'] = '0'
    os.environ['RAG_USE_CLEAN_TABLES'] = '0'
    os.environ['RAG_SYNTH_TABLES'] = '0'
    os.environ['RAG_EXTRACT_IMAGES'] = '0'
    os.environ['RAG_PDF_HI_RES'] = '0'
    os.environ['RAG_USE_PDFPLUMBER'] = '1'
    os.environ['RAG_TABLES_PER_PAGE'] = '10'

    pdf = Path('Gear wear Failure.pdf')
    els = load_elements(pdf)
    tables = [e for e in els if str(getattr(e, 'category', '')).lower() == 'table']
    print(f"Total elements: {len(els)} | tables: {len(tables)}")

    def head_row(text: str) -> str:
        lines = [ln for ln in (text or '').splitlines() if ln.strip()]
        return lines[0] if lines else ''

    def dims(text: str) -> str:
        lines = [ln for ln in (text or '').splitlines() if ln.strip()]
        if not lines:
            return '0x0'
        if lines[0].lstrip().startswith('|') and '|' in lines[0]:
            cols = max(2, lines[0].count('|') - 1)
            return f"{len(lines)}x{cols}"
        if ',' in lines[0]:
            cols = lines[0].count(',') + 1
            return f"{len(lines)}x{cols}"
        return f"{len(lines)}x?"

    for i, e in enumerate(tables, start=1):
        md = getattr(e, 'metadata', None)
        label = getattr(md, 'table_label', None)
        src = getattr(md, 'extractor', None)
        page = getattr(md, 'page_number', None)
        text = getattr(e, 'text', '') or ''
        print(f"[{i}] page={page} src={src} label={label} dims={dims(text)}")
        print('HEAD>', head_row(text))

if __name__ == '__main__':
    main()
