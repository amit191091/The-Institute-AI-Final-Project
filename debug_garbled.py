#!/usr/bin/env python3
"""
Debug the garbled OCR detection
"""

import re
from app.figure_enhancer import detect_garbled_ocr

# Test garbled text
garbled_text = "=. = y Hig aL FAs oh 1 as r ! a: allt) § ari, A et 1 ive ¢ Sas: & o SE Se ee ii Ce sos ree i | ' aa fl eae ps B, Remeny! [He]"

def debug_garbled_detection(text):
    print(f"Testing text: '{text[:50]}...'")
    print(f"Text length: {len(text)}")
    
    # Test each pattern individually
    problematic_patterns = [
        (r'[^\w\s\[\]().,;:!?%-]', 'Unusual characters'),
        (r'\b[a-zA-Z]{1}\b', 'Single letter words'),
        (r'[A-Z]{4,}', 'Long sequences of capitals'),
        (r'[^a-zA-Z\s]{3,}', 'Sequences of non-letters'),
        (r'\s{3,}', 'Multiple spaces'),
    ]
    
    total_issues = 0
    for pattern, description in problematic_patterns:
        matches = re.findall(pattern, text)
        issues = len(matches)
        total_issues += issues
        print(f"{description}: {issues} matches")
        if matches and issues < 10:  # Show first few matches for small counts
            print(f"  Examples: {matches[:5]}")
    
    print(f"Total issues: {total_issues}")
    issue_ratio = total_issues / max(len(text), 1)
    print(f"Issue ratio: {issue_ratio:.3f}")
    print(f"Threshold: 0.2")
    print(f"Is garbled: {issue_ratio > 0.2}")
    
    # Also test the actual function
    print(f"Function result: {detect_garbled_ocr(text)}")

if __name__ == "__main__":
    debug_garbled_detection(garbled_text)
