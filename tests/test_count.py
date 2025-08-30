import pdfplumber
from pathlib import Path

pdf_path = Path('Gear wear Failure.pdf')
print(f'Processing {pdf_path}')

with pdfplumber.open(pdf_path) as pdf:
    total_tables = 0
    for page_num, page in enumerate(pdf.pages, 1):
        tables = page.extract_tables()
        total_tables += len(tables)
        if tables:
            print(f'Page {page_num}: {len(tables)} tables')

print(f'Total tables found: {total_tables}')
