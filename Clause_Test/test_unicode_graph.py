#!/usr/bin/env python3
"""Test script for Unicode character handling in graph rendering."""

import networkx as nx
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.graph import render_graph_html, _sanitize_text

def test_sanitization():
    """Test the text sanitization function with problematic Unicode characters."""
    print("Testing text sanitization...")
    
    test_cases = [
        "Temperature: 25°C",
        "Micro value: 15µm",
        "Less than or equal: x ≤ 10",
        "Greater than or equal: y ≥ 5",
        "Smart quotes: 'hello' and 'world'",
        "Dashes: en–dash and em—dash",
        "Ellipsis: waiting...",
        "Mixed: 25°C, 15µm, x ≤ 10, 'smart quotes', en–dash",
        "",  # Empty string
        None,  # None value
    ]
    
    for test_case in test_cases:
        sanitized = _sanitize_text(test_case)
        print(f"Original: {repr(test_case)}")
        print(f"Sanitized: {repr(sanitized)}")
        
        # Verify it's ASCII-safe
        try:
            sanitized.encode('ascii')
            print("✓ ASCII-safe")
        except UnicodeEncodeError as e:
            print(f"✗ Still has encoding issues: {e}")
        print("-" * 50)

def test_graph_with_unicode():
    """Test graph rendering with Unicode characters in node labels."""
    print("\nTesting graph rendering with Unicode characters...")
    
    # Create a test graph with problematic Unicode characters
    G = nx.Graph()
    
    # Add nodes with Unicode characters that typically cause issues
    unicode_nodes = [
        ("temp_node", {"type": "entity", "label": "Temperature: 25°C"}),
        ("micro_node", {"type": "entity", "label": "Micro value: 15µm"}),
        ("leq_node", {"type": "entity", "label": "Pressure ≤ 100"}),
        ("geq_node", {"type": "entity", "label": "Speed ≥ 1000"}),
        ("quote_node", {"type": "entity", "label": 'Status: "operational"'}),
        ("dash_node", {"type": "entity", "label": "Range: 1–10 units"}),
        ("ellipsis_node", {"type": "entity", "label": "Loading..."}),
        ("chunk1", {"type": "chunk", "label": "Document chunk with °C and µm"}),
        ("chunk2", {"type": "chunk", "label": 'Another chunk with "quotes" and – dashes'}),
    ]
    
    for node_id, data in unicode_nodes:
        G.add_node(node_id, **data)
    
    # Add some edges
    edges = [
        ("temp_node", "chunk1"),
        ("micro_node", "chunk1"),
        ("leq_node", "chunk2"),
        ("geq_node", "chunk2"),
        ("quote_node", "chunk2"),
        ("dash_node", "chunk1"),
        ("ellipsis_node", "chunk2"),
    ]
    
    for edge in edges:
        G.add_edge(*edge)
    
    # Test rendering
    output_file = "test_unicode_graph.html"
    
    try:
        result_path = render_graph_html(G, output_file)
        print(f"✓ Graph rendered successfully to: {result_path}")
        
        # Check if file was created and has content
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✓ Output file created with size: {file_size} bytes")
            
            # Try to read the file to verify encoding
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"✓ File content readable (UTF-8): {len(content)} characters")
                
                # Check if problematic characters were handled
                problematic_chars = ['°', 'µ', '≤', '≥', '"', '"', '–', '—']
                found_problematic = [char for char in problematic_chars if char in content]
                
                if found_problematic:
                    print(f"⚠ Found problematic characters in output: {found_problematic}")
                else:
                    print("✓ No problematic Unicode characters found in output")
                    
            except UnicodeDecodeError as e:
                print(f"✗ File encoding issue: {e}")
                
        else:
            print("✗ Output file was not created")
            
    except Exception as e:
        print(f"✗ Graph rendering failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("=" * 60)
    print("UNICODE CHARACTER HANDLING TESTS")
    print("=" * 60)
    
    test_sanitization()
    test_graph_with_unicode()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
