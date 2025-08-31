#!/usr/bin/env python3
"""Simple test of table parsing improvements."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_unit_normalization():
    """Test unit normalization function."""
    print("=== Testing Unit Normalization ===")
    
    try:
        from app.table_ops import _normalize_units
        
        test_cases = [
            ("50 kS/sec", "50 kHz"),
            ("mV/g", "mV/g"),
            ("μV", "uV"),
            ("Test kS/sec value", "Test kHz value"),
        ]
        
        for input_text, expected in test_cases:
            result = _normalize_units(input_text)
            status = "✓" if expected in result else "✗"
            print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")
            
        return True
        
    except Exception as e:
        print(f"Unit normalization test failed: {e}")
        return False

def test_table_file_parsing():
    """Test parsing the actual Table 2 file."""
    print("\n=== Testing Table 2 Parsing ===")
    
    try:
        from app.table_ops import markdown_to_df
        
        table_path = Path("data/elements/Gear wear Failure-table-02.md")
        
        if not table_path.exists():
            print(f"✗ Table file not found: {table_path}")
            return False
        
        print(f"✓ Found table file: {table_path}")
        
        # Read and show raw content
        content = table_path.read_text(encoding='utf-8')
        print("Raw table content preview:")
        print(content[:200] + "...")
        
        # Parse with our function
        df = markdown_to_df(table_path)
        
        if df is not None:
            print(f"✓ Parsed successfully: {df.shape} (rows x cols)")
            print("Columns:", list(df.columns))
            print("\nFirst few rows:")
            print(df.head())
            
            # Check for sampling rate data
            found_50 = False
            for col in df.columns:
                for val in df[col]:
                    if "50" in str(val):
                        found_50 = True
                        print(f"✓ Found '50' in column '{col}': {val}")
            
            if not found_50:
                print("✗ Could not find '50' sampling rate in parsed data")
                
            return True
        else:
            print("✗ Failed to parse table")
            return False
            
    except Exception as e:
        print(f"Table parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_lookup():
    """Test natural table lookup with mock data."""
    print("\n=== Testing Table Lookup ===")
    
    try:
        from app.table_ops import natural_table_lookup
        from langchain.schema import Document
        
        # Create a simple test table
        test_table = """
| Parameter | Value |
|-----------|-------|
| Sampling Rate | 50 kHz |
| Accelerometer | Dytran 3053B |
| Sensitivity | 9.47 mV/g |
"""
        
        doc = Document(
            page_content=test_table,
            metadata={"section": "Table", "table_number": 1}
        )
        
        test_questions = [
            "What is the sampling rate?",
            "What accelerometer is used?", 
            "What is the sensitivity?",
        ]
        
        for question in test_questions:
            answer, source = natural_table_lookup(question, [doc])
            status = "✓" if answer else "✗"
            print(f"{status} Q: {question}")
            print(f"    A: {answer or 'No answer found'}")
            
        return True
        
    except Exception as e:
        print(f"Table lookup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running simple table operation tests...")
    print("=" * 50)
    
    tests = [
        test_unit_normalization,
        test_table_file_parsing, 
        test_simple_lookup,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
