#!/usr/bin/env python3
"""Test script to verify database snapshot improvements."""

import os
import json
from pathlib import Path

# Set environment flags
os.environ['RAG_PDF_HI_RES'] = '0'
os.environ['RAG_USE_PDFPLUMBER'] = '1'
os.environ['RAG_USE_TABULA'] = '1'
os.environ['RAG_USE_CAMELOT'] = '0'
os.environ['RAG_SYNTH_TABLES'] = '0'
os.environ['RAG_EXTRACT_IMAGES'] = '1'

def test_snapshot_improvements():
    """Test the improved database snapshot generation."""
    print("Testing database snapshot improvements...")
    
    # Load elements
    from app.loaders import load_elements
    pdf_path = Path('Gear wear Failure.pdf')
    
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return
    
    print(f"Loading elements from {pdf_path}...")
    elements = load_elements(pdf_path)
    print(f"Loaded {len(elements)} elements")
    
    # Check for caption elements
    table_captions = [e for e in elements if str(getattr(e, 'category', '')).lower() == 'tablecaption']
    figure_captions = [e for e in elements if str(getattr(e, 'category', '')).lower() == 'figurecaption']
    tables = [e for e in elements if str(getattr(e, 'category', '')).lower() == 'table']
    figures = [e for e in elements if str(getattr(e, 'category', '')).lower() == 'figure']
    
    print(f"Found {len(table_captions)} table captions, {len(figure_captions)} figure captions")
    print(f"Found {len(tables)} tables, {len(figures)} figures")
    
    # Print sample captions
    for i, cap in enumerate(table_captions[:3]):
        md = getattr(cap, 'metadata', None)
        if md:
            print(f"Table Caption {i+1}: {getattr(cap, 'text', '')}")
            print(f"  - Number: {getattr(md, 'table_number', None)}")
            print(f"  - Anchor: {getattr(md, 'table_anchor', None)}")
    
    for i, cap in enumerate(figure_captions[:3]):
        md = getattr(cap, 'metadata', None)
        if md:
            print(f"Figure Caption {i+1}: {getattr(cap, 'text', '')}")
            print(f"  - Number: {getattr(md, 'figure_number', None)}")
            print(f"  - Anchor: {getattr(md, 'figure_anchor', None)}")
    
    # Test ground truth loading
    print("\nTesting ground truth file loading...")
    gt_path = Path('data') / 'gear_wear_ground_truth_context_free.json'
    qa_path = Path('data') / 'gear_wear_qa_context_free.jsonl'
    
    if gt_path.exists():
        print(f"Ground truth file exists: {gt_path}")
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        print(f"Ground truth items: {len(gt_data)}")
        if gt_data:
            print(f"Sample GT item: {gt_data[0]}")
    else:
        print(f"Ground truth file not found: {gt_path}")
    
    if qa_path.exists():
        print(f"QA file exists: {qa_path}")
        with open(qa_path, 'r') as f:
            qa_data = [json.loads(line) for line in f if line.strip()]
        print(f"QA items: {len(qa_data)}")
        if qa_data:
            print(f"Sample QA item: {qa_data[0]}")
    else:
        print(f"QA file not found: {qa_path}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_snapshot_improvements()
