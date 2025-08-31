#!/usr/bin/env python3
"""
Test All Tables
==============

Test script to verify the table cleaner works with all three table types.
"""

import sys
from pathlib import Path

# Add RAG to path
sys.path.insert(0, str(Path(__file__).parent / "RAG"))

def test_all_tables():
    """Test with all three table types from the Gear wear Failure report."""
    
    # Test 1: Wear depth table (mangled version)
    mangled_wear_table = """| Case |  | Wear depth |  |  | Wear depth |  | Wear depth |  | Wear depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | Case |  | Case |  | Case |  |
|  |  | [μm] |  |  | [μm] |  | [μm] |  | [μm] |
|  |  |  |  |  |  |  |  |  |  |
| Healthy | 0 |  |  | W9 | 285 | W18 | 442 | W27 | 686 |
| W1 | 38 |  |  | W10 | 300 | W19 | 465 | W28 | 721 |
| W2 | 77 |  |  | W11 | 314 | W20 | 488 | W29 | 757 |
| W3 | 115 |  |  | W12 | 330 | W21 | 512 | W30 | 795 |
| W4 | 152 |  |  | W13 | 347 | W22 | 538 | W31 | 834 |
| W5 | 166 |  |  | W14 | 364 | W23 | 565 | W32 | 876 |
| W6 | 185 |  |  | W15 | 382 | W24 | 593 | W33 | 920 |
| W7 | 259 |  |  | W16 | 401 | W25 | 623 | W34 | 966 |
| W8 | 272 |  |  | W17 | 421 | W26 | 654 | W35 | 1000 |"""
    
    # Test 2: Sensor table (mangled version)
    mangled_sensor_table = """| Sensor |  | Direction and |  | Brand |  | Sensitivity |  | R | Sampling |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | Position |  |  |  | [mV/g] |  |  | ate [kS/sec] |  |"""
    
    # Test 3: Equipment table (mangled version)
    mangled_equipment_table = """|  | Feature |  | Value / Type |
| --- | --- | --- | --- |
| --- | --- | --- | --- |
| Model |  | MG-5025A |  |
| Gears type |  | Spur |  |
| Module |  | 3 mm |  |
| Lubricant |  | 2640 semi-synthetic (15W/40) |  |"""
    
    print("🧪 Testing All Table Types...")
    
    try:
        from RAG.app.loader_modules.table_cleaner import clean_table_structure, is_table_clean
        
        # Test 1: Wear depth table
        print("\n📋 Test 1: Wear Depth Table")
        print("Original (mangled):")
        print(mangled_wear_table[:200] + "...")
        cleaned_wear = clean_table_structure(mangled_wear_table)
        print("\nCleaned:")
        print(cleaned_wear)
        
        # Test 2: Sensor table
        print("\n📋 Test 2: Sensor Table")
        print("Original (mangled):")
        print(mangled_sensor_table)
        cleaned_sensor = clean_table_structure(mangled_sensor_table)
        print("\nCleaned:")
        print(cleaned_sensor)
        
        # Test 3: Equipment table
        print("\n📋 Test 3: Equipment Table")
        print("Original (mangled):")
        print(mangled_equipment_table)
        cleaned_equipment = clean_table_structure(mangled_equipment_table)
        print("\nCleaned:")
        print(cleaned_equipment)
        
        print("\n✅ All table tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_tables()
