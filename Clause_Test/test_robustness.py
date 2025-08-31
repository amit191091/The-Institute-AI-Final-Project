#!/usr/bin/env python3
"""Test script for robustness improvements"""

import sys
sys.path.append('.')

from app.agents import _extract_simple_entities
from langchain.schema import Document

print("=== Testing Robustness Improvements ===\n")

# Test 1: Vessel name extraction  
print("1. Vessel Name Extraction Test")
docs = [Document(
    page_content='Naval Vessel INS Haifa propulsion train was monitored during testing', 
    metadata={'source': 'Gear wear Failure.pdf', 'page': '10'}
)]
result = _extract_simple_entities('Which vessel\'s propulsion train was monitored?', docs)
print(f"Question: Which vessel's propulsion train was monitored?")
print(f"Expected: INS Haifa")
print(f"Extracted: {result}")
print()

# Test 2: Percentage extraction
print("2. Percentage Extraction Test")
docs2 = [Document(
    page_content='RMS levels elevated by roughly 10–15% above the April 9 reference, with increases stable across consecutive runs', 
    metadata={'source': 'Gear wear Failure.pdf', 'page': '4'}
)]
result2 = _extract_simple_entities('By approximately how much did RMS rise above April 9 levels at 45 RPS during moderate wear?', docs2)
print(f"Question: By approximately how much did RMS rise above April 9 levels at 45 RPS during moderate wear?")
print(f"Expected: About 10–15%")  
print(f"Extracted: {result2}")
print()

# Test 3: High frequency behavior
print("3. High-Frequency Behavior Test")
docs3 = [Document(
    page_content='At 15 RPS, spectra carried the same tendencies with more high-frequency smearing, reflecting lubrication sensitivity rather than discrete failure', 
    metadata={'source': 'Gear wear Failure.pdf', 'page': '3'}
)]
result3 = _extract_simple_entities('At 15 RPS during early wear, what high-frequency behavior was observed?', docs3)
print(f"Question: At 15 RPS during early wear, what high-frequency behavior was observed?")
print(f"Expected: More high-frequency smearing (haze)")
print(f"Extracted: {result3}")
print()

print("=== Robustness Test Complete ===")
