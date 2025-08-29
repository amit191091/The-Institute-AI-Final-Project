#!/usr/bin/env python3
"""
Test Tool Implementations
========================

Demonstrates all the tool implementations with the exact signature format.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the tool implementations from MCP_Tools
from MCP_Tools.tool_implementations import rag, vision, vib, timeline

def test_rag_tools():
    """Test RAG tools."""
    print("🔍 Testing RAG Tools...")
    
    # Test indexing
    print("\n1. Testing rag.index()...")
    result = rag.index("RAG/data", clear=False)
    print(f"   Result: {result}")
    
    # Test querying
    print("\n2. Testing rag.query()...")
    result = rag.query("What is gear wear?", top_k=5)
    print(f"   Result: {result}")
    
    # Test evaluation
    print("\n3. Testing rag.evaluate()...")
    result = rag.evaluate("RAG/data/gear_wear_qa.jsonl")
    print(f"   Result: {result}")

def test_vision_tools():
    """Test Vision tools."""
    print("\n🖼️ Testing Vision Tools...")
    
    # Find a test image
    image_path = None
    for ext in ['*.jpg', '*.png', '*.bmp']:
        images = list(Path("Pictures and Vibrations database/Picture/database").glob(ext))
        if images:
            image_path = str(images[0])
            break
    
    if image_path:
        # Test alignment
        print(f"\n1. Testing vision.align() with {image_path}...")
        result = vision.align(image_path)
        print(f"   Result: {result}")
        
        # Test measurement
        print(f"\n2. Testing vision.measure() with {image_path}...")
        result = vision.measure(image_path)
        print(f"   Result: {result}")
    else:
        print("   No test images found in database directory")

def test_vibration_tools():
    """Test Vibration tools."""
    print("\n📊 Testing Vibration Tools...")
    
    # Test RMS file
    rms_file = "Pictures and Vibrations database/Vibration/database/RMS15.csv"
    if Path(rms_file).exists():
        print(f"\n1. Testing vib.features() with RMS file...")
        result = vib.features(rms_file, fs=50000)
        print(f"   Result: {result}")
    
    # Test FME file
    fme_file = "Pictures and Vibrations database/Vibration/database/FME Values.csv"
    if Path(fme_file).exists():
        print(f"\n2. Testing vib.features() with FME file...")
        result = vib.features(fme_file, fs=50000)
        print(f"   Result: {result}")
    
    # Test time series file
    time_file = "Pictures and Vibrations database/Vibration/database/Vibration signals high speed.csv"
    if Path(time_file).exists():
        print(f"\n3. Testing vib.features() with time series file...")
        result = vib.features(time_file, fs=50000)
        print(f"   Result: {result}")

def test_timeline_tools():
    """Test Timeline tools."""
    print("\n📅 Testing Timeline Tools...")
    
    # Test with PDF
    pdf_file = "Gear wear Failure.pdf"
    if Path(pdf_file).exists():
        print(f"\n1. Testing timeline.summarize() with PDF (mapreduce mode)...")
        result = timeline.summarize(pdf_file, mode="mapreduce")
        print(f"   Result: {result}")
        
        print(f"\n2. Testing timeline.summarize() with PDF (refine mode)...")
        result = timeline.summarize(pdf_file, mode="refine")
        print(f"   Result: {result}")
    
    # Test with Word document
    doc_file = "Gear wear Failure.docx"
    if Path(doc_file).exists():
        print(f"\n3. Testing timeline.summarize() with Word document...")
        result = timeline.summarize(doc_file, mode="mapreduce")
        print(f"   Result: {result}")

def main():
    """Run all tool tests."""
    print("🚀 Testing All Tool Implementations")
    print("=" * 50)
    
    try:
        test_rag_tools()
    except Exception as e:
        print(f"❌ RAG tools test failed: {e}")
    
    try:
        test_vision_tools()
    except Exception as e:
        print(f"❌ Vision tools test failed: {e}")
    
    try:
        test_vibration_tools()
    except Exception as e:
        print(f"❌ Vibration tools test failed: {e}")
    
    try:
        test_timeline_tools()
    except Exception as e:
        print(f"❌ Timeline tools test failed: {e}")
    
    print("\n✅ Tool testing completed!")

if __name__ == "__main__":
    main()
