#!/usr/bin/env python3
"""Test extractors individually to debug table extraction issues."""

from pathlib import Path

def test_extractors():
    """Quick test of all extractors."""
    pdf_path = Path("Gear wear Failure.pdf")
    if not pdf_path.exists():
        print(f"FAIL: PDF not found: {pdf_path}")
        return
    
    print("=" * 60)
    print("TESTING PDF TABLE EXTRACTORS")
    print("=" * 60)
    
    # Test pdfplumber
    print("\n1. TESTING PDFPLUMBER:")
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total = 0
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if tables:
                    print(f"  Page {page_num}: {len(tables)} tables")
                    total += len(tables)
            print(f"  RESULT: {total} tables total")
    except ImportError:
        print("  FAIL: pdfplumber not installed")
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Test camelot
    print("\n2. TESTING CAMELOT:")
    try:
        import camelot
        tables = camelot.read_pdf(str(pdf_path), pages="all")
        print(f"  RESULT: {len(tables)} tables found")
        for i, table in enumerate(tables[:3]):  # Show first 3
            print(f"    Table {i+1}: Page {table.page}, Shape {table.df.shape}")
    except ImportError:
        print("  FAIL: camelot not installed")
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Test tabula
    print("\n3. TESTING TABULA:")
    try:
        import tabula
        dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
        print(f"  RESULT: {len(dfs)} tables found")
        for i, df in enumerate(dfs[:3]):  # Show first 3
            print(f"    Table {i+1}: Shape {df.shape}")
    except ImportError:
        print("  FAIL: tabula not installed")
    except Exception as e:
        print(f"  FAIL: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_extractors()
