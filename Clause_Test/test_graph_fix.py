#!/usr/bin/env python3
"""Test the fixed graph rendering functionality."""

import sys
import os
sys.path.append('.')

try:
    from app.graph import build_graph, render_graph_html
    import networkx as nx
    
    print("🔧 Testing fixed graph rendering")
    print("=" * 40)
    
    # Create a simple test graph
    G = nx.Graph()
    G.add_node("Test Node 1", type="entity", label="Test Entity")
    G.add_node("Test Node 2", type="chunk", label="Test Chunk")
    G.add_edge("Test Node 1", "Test Node 2")
    
    print(f"Created test graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Test rendering
    output_path = "logs/test_graph.html"
    print(f"Attempting to render graph to: {output_path}")
    
    try:
        result = render_graph_html(G, output_path)
        if result and os.path.exists(output_path):
            print(f"✅ Graph rendering successful! File created: {output_path}")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            print(f"   File size: {file_size} bytes")
            
            if file_size > 0:
                print("✅ Graph file is not empty")
            else:
                print("⚠️  Graph file is empty")
        else:
            print(f"❌ Graph rendering failed - no file created")
            
    except Exception as e:
        print(f"❌ Graph rendering error: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Error importing graph modules: {e}")
    import traceback
    traceback.print_exc()
