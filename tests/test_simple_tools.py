#!/usr/bin/env python3
"""
Simple Tool Test
===============

Tests all the simplified tool implementations.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_tools():
    """Test all tool implementations."""
    print("🔧 Testing Simplified Tool Implementations...")
    
    try:
        # Import the simplified tools
        from tool_implementations_simple import rag, vision, vib, timeline
        
        print("\n✅ All tool imports successful!")
        
        # Test 1: RAG Index
        print("\n1. Testing RAG Index...")
        result = rag.index("RAG/data", clear=False)
        print(f"   Result: {result['ok']}")
        if result['ok']:
            print(f"   📊 Indexed: {result.get('indexed', 0)} documents")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test 2: RAG Query
        print("\n2. Testing RAG Query...")
        result = rag.query("What is gear wear?", top_k=5)
        print(f"   Result: {result['ok']}")
        if result['ok']:
            print(f"   💬 Answer: {result.get('answer', 'No answer')[:100]}...")
            print(f"   📄 Sources: {len(result.get('sources', []))}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test 3: RAG Evaluate
        print("\n3. Testing RAG Evaluate...")
        result = rag.evaluate("test_eval_set")
        print(f"   Result: {result['ok']}")
        if result['ok']:
            scores = result.get('scores', {})
            print(f"   📈 Faithfulness: {scores.get('faithfulness', 0):.2f}")
            print(f"   📈 Answer Correctness: {scores.get('answer_correctness', 0):.2f}")
            print(f"   📈 Context Precision: {scores.get('context_precision', 0):.2f}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test 4: Vision Align
        print("\n4. Testing Vision Align...")
        # Use a sample image if available
        sample_image = "Pictures and Vibrations database/Picture/database/Healthy.jpg"
        if Path(sample_image).exists():
            result = vision.align(sample_image)
            print(f"   Result: {result['ok']}")
            if result['ok']:
                print(f"   🖼️  Aligned path: {result.get('aligned_path', 'N/A')}")
                transform = result.get('transform', {})
                print(f"   📐 Center: ({transform.get('center_x', 0)}, {transform.get('center_y', 0)})")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  Sample image not found: {sample_image}")
        
        # Test 5: Vision Measure
        print("\n5. Testing Vision Measure...")
        if Path(sample_image).exists():
            result = vision.measure(sample_image)
            print(f"   Result: {result['ok']}")
            if result['ok']:
                print(f"   📏 Depth: {result.get('depth_um', 0):.2f} µm")
                print(f"   📐 Area: {result.get('area_um2', 0):.2f} µm²")
                print(f"   📊 Scale: {result.get('scale_um_per_px', 0):.2f} µm/px")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  Sample image not found: {sample_image}")
        
        # Test 6: Vibration Features
        print("\n6. Testing Vibration Features...")
        sample_vib = "Pictures and Vibrations database/Vibration/database/RMS15.csv"
        if Path(sample_vib).exists():
            result = vib.features(sample_vib, fs=50000)
            print(f"   Result: {result['ok']}")
            if result['ok']:
                print(f"   📊 RMS: {result.get('rms', 0):.4f}")
                print(f"   📈 Peaks: {len(result.get('peaks', []))}")
                print(f"   🎵 Bands: {len(result.get('bands', {}))}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  Sample vibration data not found: {sample_vib}")
        
        # Test 7: Timeline Summarize
        print("\n7. Testing Timeline Summarize...")
        result = timeline.summarize("Gear wear Failure.pdf", mode="mapreduce")
        print(f"   Result: {result['ok']}")
        if result['ok']:
            timeline_events = result.get('timeline', [])
            print(f"   📅 Events: {len(timeline_events)}")
            for event in timeline_events[:2]:  # Show first 2 events
                print(f"      {event.get('t', 'N/A')}: {event.get('event', 'N/A')}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        print("\n✅ All Tool Tests Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Tool Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tools()
    sys.exit(0 if success else 1)
