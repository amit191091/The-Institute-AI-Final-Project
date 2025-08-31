#!/usr/bin/env python3
"""
Test the enhanced figure processing
"""

from app.figure_enhancer import extract_figure_descriptions_from_chunks, enhance_figure_content

# Mock some text chunks like what we found in the logs
chunks = [
    {'content': 'Figure 1: Face view of the drive gear tooth at all health status. Figure 2: RMS level against wear depth at 15 [RPS] (above) and at 45 [RPS] (below).'},
    {'content': 'Figure 3: Fast Fourier Transform (FFT) spectrogram at 15 [RPS] (above) and 45 [RPS] (below). The white dashed line marks the separation between healthy and faulty cases. Figure 4: Normal'},
    {'content': 'Some other text without figures'},
]

def test_figure_enhancement():
    print("=== Testing Enhanced Figure Processing ===\n")
    
    # Extract descriptions
    descriptions = extract_figure_descriptions_from_chunks(chunks)
    print('Extracted descriptions:')
    for fig_num, desc in descriptions.items():
        print(f'  Figure {fig_num}: {desc}')
    
    # Test enhancement with garbled OCR
    garbled_content = '''Figure 3: OCR Text: =. = y Hig aL FAs oh 1 as r ! a: allt) § ari, A et 1 ive ¢ Sas: & o SE Se ee ii Ce sos ree i | ' aa fl eae ps B, Remeny! [He]
Context: Some context text
Page: 13
Figure 3'''

    print(f'\nOriginal content (garbled):')
    print(garbled_content[:150] + '...')
    
    enhanced = enhance_figure_content(garbled_content, figure_number=3, all_chunks=chunks)
    print(f'\nEnhanced content:')
    print(enhanced)
    
    # Debug: Check if OCR pattern is found
    import re
    ocr_match = re.search(r'OCR Text:\s*(.+?)(?:\nContext|$)', garbled_content, re.IGNORECASE | re.DOTALL)
    print(f"\nDEBUG: OCR pattern found: {ocr_match is not None}")
    if ocr_match:
        print(f"DEBUG: OCR text extracted: '{ocr_match.group(1)[:50]}...'")
    
    # Debug: Check garbled detection
    if ocr_match:
        from app.figure_enhancer import detect_garbled_ocr
        is_garbled = detect_garbled_ocr(ocr_match.group(1))
        print(f"DEBUG: Is garbled: {is_garbled}")
    
    print(f"\n=== Test Results ===")
    if "Fast Fourier Transform" in enhanced:
        print("✅ SUCCESS: Garbled OCR text was replaced with proper description!")
    else:
        print("❌ FAILED: OCR text was not enhanced")

if __name__ == "__main__":
    test_figure_enhancement()
