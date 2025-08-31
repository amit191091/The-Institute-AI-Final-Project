#!/usr/bin/env python3
"""Basic test to validate our improvements work."""

import sys
import os
sys.path.append('.')

print("Testing table parsing improvements...")

try:
    from app.table_ops import natural_table_lookup, build_tables_from_docs
    import pandas as pd
    
    # Create a mock document with table content
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {"type": "table"}
    
    # Test our unit normalization with a realistic Table 2 scenario
    table_content = """
| Sensor | Direction and Position | Brand | Sensitivity [mV/g] | Sampling Rate [kS/sec] |
|--------|------------------------|-------|-------------------|----------------------|
| Accelerometer | Vertical - Mid of shaft | Dytran 3053B | 100 | 50 |
| Tachometer | Horizontal - End of shaft | Honeywell 3010AN | N/A | N/A |

Additional Info: Tachometer – 30 teeth
"""
    
    docs = [MockDoc(table_content)]
    
    # Test queries that were failing before our improvements
    test_queries = [
        "What was the sampling rate?",
        "Name the accelerometer model", 
        "Which brand was used for the tachometer?",
        "How many teeth did the gear have?"
    ]
    
    print("📊 Testing table queries:")
    for query in test_queries:
        result, source = natural_table_lookup(query, docs)
        print(f"  Q: '{query}'")
        print(f"  A: '{result}'")
        print()
    
    print("✅ Table parsing test completed")
    
except Exception as e:
    print(f"❌ Error in table parsing test: {e}")
    import traceback
    traceback.print_exc()

# Test the unit normalization function directly
try:
    from app.table_ops import _normalize_units
    
    test_cases = [
        "50 kS/sec",
        "100 mV/g", 
        "μV",
        "200 Hz"
    ]
    
    print("🔧 Testing unit normalization:")
    for test in test_cases:
        normalized = _normalize_units(test)
        print(f"  '{test}' → '{normalized}'")
    
    print("✅ Unit normalization test completed")
    
except Exception as e:
    print(f"❌ Error in unit normalization test: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Test suite completed!")
