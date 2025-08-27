import os
from pathlib import Path


def main():
    # Force extractor preferences for this diagnostic run
    os.environ['RAG_PDF_HI_RES'] = '0'
    os.environ['RAG_USE_PDFPLUMBER'] = '1'
    os.environ['RAG_USE_TABULA'] = '1'
    os.environ['RAG_USE_CAMELOT'] = '0'
    os.environ['RAG_SYNTH_TABLES'] = '0'
    os.environ['RAG_EXTRACT_IMAGES'] = '0'

    from app.loaders import load_elements

    pdf_path = Path('Gear wear Failure.pdf')
    elements = load_elements(pdf_path)

    rows = []
    for e in elements:
        if str(getattr(e, 'category', '')).lower() == 'table':
            md = getattr(e, 'metadata', None)
            num = getattr(md, 'table_number', None)
            extractor = getattr(md, 'extractor', None)
            label = getattr(md, 'table_label', None)
            anchor = getattr(md, 'table_anchor', None)

            # Try to preview the first header line from the markdown text
            header_preview = None
            text = getattr(e, 'text', '') or ''
            if '|' in text:
                for line in text.splitlines():
                    s = line.strip()
                    if '|' in s and not s.startswith('['):
                        header_preview = s
                        break

            rows.append((num, extractor, label, anchor, header_preview))

    rows.sort(key=lambda x: (x[0] if isinstance(x[0], int) else 9999))
    for num, extractor, label, anchor, header in rows:
        print(f"Table {num}: extractor={extractor} label={label} anchor={anchor}")
        if header:
            print(f"  header: {header}")


if __name__ == '__main__':
    main()
