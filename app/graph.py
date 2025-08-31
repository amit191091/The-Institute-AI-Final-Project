# from __future__ import annotations

# from typing import List, Tuple, Sequence, Protocol, runtime_checkable
# import re
# import networkx as nx
# import json
# from app.logger import trace_func

# @runtime_checkable
# class DocLike(Protocol):
#     page_content: str
#     metadata: dict

# @trace_func
# def _extract_entities(text: str) -> List[str]:
#     """Entity regex: W-cases, dates, times, key nouns (RPS, RMS, FME)."""
#     t = text or ""
#     ents = set()
#     for m in re.findall(r"\bW\d{1,3}\b", t):
#         ents.add(m)
#     # Month names like "May 14" or abbreviations
#     for m in re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b", t, re.I):
#         ents.add(m)
#     # Numeric dates like 2025-05-14 or 14/05/2025
#     for m in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", t):
#         ents.add(m)
#     for m in re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t):
#         ents.add(m)
#     # Times like 13:51–14:17
#     for m in re.findall(r"\b\d{1,2}:\d{2}(?:\s*[–-]\s*\d{1,2}:\d{2})?\b", t):
#         ents.add(m)
#     for kw in ("RMS", "FME", "GMF", "RPS", "crest factor", "wear depth"):
#         if re.search(rf"\b{re.escape(kw)}\b", t, re.I):
#             ents.add(kw)
#     return list(ents)

# @trace_func
# def build_graph(docs: Sequence[DocLike]) -> nx.Graph:
#     """Build a simple undirected graph of entities co-mentioned within the same chunk.
#     Nodes: entities and file/page anchors. Edges: co-mentions.
#     """
#     G = nx.Graph()
#     for d in docs:
#         md = d.metadata or {}
#         node_doc = f"{md.get('file_name','doc')}#p{md.get('page')}:{md.get('section','')}/{md.get('anchor','')}"
#         G.add_node(node_doc, type="chunk", label=node_doc)
#         ents = _extract_entities(d.page_content)
#         for e in ents:
#             if not G.has_node(e):
#                 G.add_node(e, type="entity", label=e)
#             G.add_edge(node_doc, e)
#         # connect entities co-mentioned in same chunk
#         for i in range(len(ents)):
#             for j in range(i + 1, len(ents)):
#                 a, b = ents[i], ents[j]
#                 if a != b:
#                     G.add_edge(a, b)
#     return G

# # @trace_func
# # def render_graph_html(G: nx.Graph, out_path: str, height: str = "600px") -> str:
# #     """Render the graph to an interactive HTML. Prefer pyvis, fallback to a minimal vis-network HTML.

# #     This avoids failures when Jinja2 templates are unavailable in some environments.
# #     """
# #     # First, try PyVis rendering
# #     try:
# #         from pyvis.network import Network  # type: ignore
# #         net = Network(height=height, width="100%", directed=False, notebook=False, cdn_resources="in_line")
# #         for n, data in G.nodes(data=True):
# #             net.add_node(n, label=data.get("label", str(n)), color="#6aa84f" if data.get("type") == "entity" else "#3c78d8")
# #         for u, v in G.edges():
# #             net.add_edge(str(u), str(v))
# #         # set_options expects a JSON string, not a JS object literal
# #         net.set_options(
# #             '{"nodes": {"shape": "dot", "size": 12}, '
# #             '"physics": {"stabilization": true}, '
# #             '"interaction": {"hover": true, "tooltipDelay": 150}}'
# #         )
# #         try:
# #             net.write_html(out_path)  # avoid opening browser
# #         except Exception:
# #             net.show(out_path)
# #         return out_path
# #     except Exception:
# #         pass

# #     # Fallback: write a minimal vis-network HTML manually
# #     nodes = []
# #     for n, data in G.nodes(data=True):
# #         nodes.append({
# #             "id": str(n),
# #             "label": data.get("label", str(n)),
# #             "color": "#6aa84f" if data.get("type") == "entity" else "#3c78d8",
# #         })
# #     edges = [{"from": str(u), "to": str(v)} for u, v in G.edges()]
# #     html = f"""<!doctype html>
# # <html>
# # <head>
# #   <meta charset=\"utf-8\" />
# #   <title>Knowledge Graph</title>
# #   <script src=\"https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js\"></script>
# #   <style>
# #     #kg {{ width: 100%; height: {height}; border: 1px solid #ccc; }}
# #   </style>
# #   </head>
# # <body>
# #   <div id=\"kg\"></div>
# #   <script>
# #     const nodes = new vis.DataSet({json.dumps(nodes)});
# #     const edges = new vis.DataSet({json.dumps(edges)});
# #     const container = document.getElementById('kg');
# #     const data = {{ nodes, edges }};
# #     const options = {{ nodes: {{ shape: 'dot', size: 12 }}, physics: {{ stabilization: true }}, interaction: {{ hover: true, tooltipDelay: 150 }} }};
# #     new vis.Network(container, data, options);
# #   </script>
# # </body>
# # </html>
# # """
# #     with open(out_path, "w", encoding="utf-8") as f:
# #         f.write(html)
# #     return out_path
# @trace_func
# def render_graph_html(G: nx.Graph, out_path: str, height: str = "800px") -> str:
#     """Render the graph to an interactive HTML with improved layout and clarity.
    
#     Uses enhanced physics settings, better node styling, and filtering to create
#     a much more readable and organized graph visualization.
#     """
#     # First, try PyVis rendering with improved settings
#     try:
#         from pyvis.network import Network  # type: ignore
        
#         # Filter graph to reduce clutter - keep only nodes with degree > 1
#         # and limit total nodes for better performance
#         filtered_G = G.copy()
#         nodes_to_remove = [n for n in filtered_G.nodes() if filtered_G.degree(n) <= 1]
#         filtered_G.remove_nodes_from(nodes_to_remove)
        
#         # If still too many nodes, keep only the most connected ones
#         if len(filtered_G.nodes()) > 100:
#             # Sort by degree and keep top nodes
#             node_degrees = [(n, filtered_G.degree(n)) for n in filtered_G.nodes()]
#             node_degrees.sort(key=lambda x: x[1], reverse=True)
#             nodes_to_keep = [n for n, _ in node_degrees[:100]]
#             nodes_to_remove = [n for n in filtered_G.nodes() if n not in nodes_to_keep]
#             filtered_G.remove_nodes_from(nodes_to_remove)
        
#         net = Network(height=height, width="100%", directed=False, notebook=False, cdn_resources="in_line")
        
#         # Add nodes with enhanced styling
#         for n, data in filtered_G.nodes(data=True):
#             node_type = data.get("type", "unknown")
#             degree = filtered_G.degree(n)
            
#             # Use enhanced size calculation
#             if node_type == "entity":
#                 size = min(12 + degree * 4, 35)  # Entities get bigger
#             else:
#                 size = min(10 + degree * 2.5, 25)   # Chunks get smaller
            
#             # Use enhanced color coding
#             if node_type == "entity":
#                 if any(keyword in str(n).lower() for keyword in ["w1", "w2", "w3", "w4", "w5"]):
#                     color = "#E74C3C"  # Bright red for W-cases
#                 elif any(keyword in str(n).lower() for keyword in ["rms", "fme", "gmf", "rps"]):
#                     color = "#3498DB"  # Blue for technical terms
#                 elif any(keyword in str(n).lower() for keyword in ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]):
#                     color = "#F39C12"  # Orange for dates
#                 else:
#                     color = "#27AE60"  # Green for other entities
#             elif node_type == "chunk":
#                 color = "#9B59B6"  # Purple for chunks
#             else:
#                 color = "#95A5A6"  # Gray for unknown
            
#             # Add node with enhanced properties
#             net.add_node(
#                 n, 
#                 label=data.get("label", str(n))[:35],  # Slightly longer labels
#                 color=color,
#                 size=size,
#                 font={
#                     "size": max(12, size - 3), 
#                     "face": "Arial", 
#                     "color": "#000000",
#                     "strokeWidth": 2,
#                     "strokeColor": "#FFFFFF"
#                 },
#                 borderWidth=3 if degree > 8 else 2,
#                 borderColor="#000000",
#                 shadow=True,
#                 mass=degree + 1  # Heavier nodes for better physics
#             )
        
#         # Add edges with improved styling
#         for u, v in filtered_G.edges():
#             net.add_edge(
#                 str(u), 
#                 str(v),
#                 width=1,
#                 color={"color": "#848484", "opacity": 0.6},
#                 smooth={"type": "continuous", "forceDirection": "none"}
#             )
        
#         # Enhanced physics and layout settings with maximum spacing
#         physics_options = {
#             "forceAtlas2Based": {
#                 "gravitationalConstant": -150,  # Much stronger repulsion
#                 "centralGravity": 0.003,       # Very weak central attraction
#                 "springLength": 250,           # Much longer spring length
#                 "springConstant": 0.03,        # Softer springs
#                 "damping": 0.5,                # More damping for stability
#                 "avoidOverlap": 0.95           # Maximum overlap avoidance
#             },
#             "stabilization": {
#                 "enabled": True,
#                 "iterations": 2000,            # More iterations for better layout
#                 "updateInterval": 50,
#                 "fit": True
#             },
#             "minVelocity": 0.75,
#             "maxVelocity": 30
#         }
        
#         # Enhanced interaction settings
#         interaction_options = {
#             "hover": True,
#             "tooltipDelay": 200,
#             "hideEdgesOnDrag": True,
#             "navigationButtons": True,
#             "keyboard": True,
#             "zoomView": True,
#             "dragView": True
#         }
        
#         # Enhanced node settings
#         node_options = {
#             "shape": "dot",
#             "shadow": True,
#             "borderWidth": 2,
#             "borderColor": "#000000"
#         }
        
#         # Enhanced edge settings
#         edge_options = {
#             "smooth": {"type": "continuous", "forceDirection": "none"},
#             "color": {"color": "#848484", "opacity": 0.6},
#             "width": 1
#         }
        
#         # Combine all options
#         all_options = {
#             "nodes": node_options,
#             "edges": edge_options,
#             "physics": physics_options,
#             "interaction": interaction_options
#         }
        
#         net.set_options(json.dumps(all_options))
        
#         try:
#             net.write_html(out_path)  # avoid opening browser
#         except Exception:
#             net.show(out_path)
#         return out_path
        
#     except Exception as e:
#         print(f"PyVis rendering failed: {e}")
#         pass

#     # Fallback: write an improved vis-network HTML manually
#     # Filter graph for fallback too
#     filtered_G = G.copy()
#     nodes_to_remove = [n for n in filtered_G.nodes() if filtered_G.degree(n) <= 1]
#     filtered_G.remove_nodes_from(nodes_to_remove)
    
#     if len(filtered_G.nodes()) > 100:
#         node_degrees = [(n, filtered_G.degree(n)) for n in filtered_G.nodes()]
#         node_degrees.sort(key=lambda x: x[1], reverse=True)
#         nodes_to_keep = [n for n, _ in node_degrees[:100]]
#         nodes_to_remove = [n for n in filtered_G.nodes() if n not in nodes_to_keep]
#         filtered_G.remove_nodes_from(nodes_to_remove)
    
#     nodes = []
#     for n, data in filtered_G.nodes(data=True):
#         node_type = data.get("type", "unknown")
#         degree = filtered_G.degree(n)
        
#         # Use enhanced size calculation
#         if node_type == "entity":
#             size = min(12 + degree * 4, 35)  # Entities get bigger
#         else:
#             size = min(10 + degree * 2.5, 25)   # Chunks get smaller
        
#         # Use enhanced color coding
#         if node_type == "entity":
#             if any(keyword in str(n).lower() for keyword in ["w1", "w2", "w3", "w4", "w5"]):
#                 color = "#E74C3C"  # Bright red for W-cases
#             elif any(keyword in str(n).lower() for keyword in ["rms", "fme", "gmf", "rps"]):
#                 color = "#3498DB"  # Blue for technical terms
#             elif any(keyword in str(n).lower() for keyword in ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]):
#                 color = "#F39C12"  # Orange for dates
#             else:
#                 color = "#27AE60"  # Green for other entities
#         elif node_type == "chunk":
#             color = "#9B59B6"  # Purple for chunks
#         else:
#             color = "#95A5A6"  # Gray for unknown
            
#         nodes.append({
#             "id": str(n),
#             "label": data.get("label", str(n))[:35],
#             "color": color,
#             "size": size,
#             "font": {
#                 "size": max(12, size - 3), 
#                 "face": "Arial", 
#                 "color": "#000000",
#                 "strokeWidth": 2,
#                 "strokeColor": "#FFFFFF"
#             },
#             "borderWidth": 3 if degree > 8 else 2,
#             "borderColor": "#000000",
#             "shadow": True,
#             "mass": degree + 1
#         })
    
#     edges = [{"from": str(u), "to": str(v), "width": 1, "color": {"color": "#848484", "opacity": 0.6}} for u, v in filtered_G.edges()]
    
#     # Enhanced fallback options
#     fallback_options = {
#         "nodes": {
#             "shape": "dot",
#             "shadow": True,
#             "borderWidth": 2,
#             "borderColor": "#000000"
#         },
#         "edges": {
#             "smooth": {"type": "continuous", "forceDirection": "none"},
#             "color": {"color": "#848484", "opacity": 0.6},
#             "width": 1
#         },
#         "physics": {
#             "forceAtlas2Based": {
#                 "gravitationalConstant": -100,
#                 "centralGravity": 0.005,
#                 "springLength": 150,
#                 "springConstant": 0.05,
#                 "damping": 0.3,
#                 "avoidOverlap": 0.8
#             },
#             "stabilization": {
#                 "enabled": True,
#                 "iterations": 2000,
#                 "updateInterval": 50,
#                 "fit": True
#             },
#             "minVelocity": 0.75,
#             "maxVelocity": 30
#         },
#         "interaction": {
#             "hover": True,
#             "tooltipDelay": 200,
#             "hideEdgesOnDrag": True,
#             "navigationButtons": True,
#             "keyboard": True,
#             "zoomView": True,
#             "dragView": True
#         }
#     }
    
#     html = f"""<!doctype html>
# <html>
# <head>
#   <meta charset=\"utf-8\" />
#   <title>Knowledge Graph - Enhanced Layout</title>
#   <script src=\"https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js\"></script>
#   <style>
#     body {{ margin: 0; padding: 10px; font-family: Arial, sans-serif; }}
#     #kg {{ width: 100%; height: {height}; border: 2px solid #ddd; border-radius: 8px; }}
#     .controls {{ margin-bottom: 10px; }}
#     .info {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
#   </style>
# </head>
# <body>
#      <div class="info">
#      <strong>Enhanced Knowledge Graph</strong><br>
#      • Nodes sized by connectivity (more connections = bigger)<br>
#      • <span style="color:#E74C3C">Red nodes</span>: W-cases (W1, W2, W3, etc.)<br>
#      • <span style="color:#3498DB">Blue nodes</span>: Technical terms (RMS, FME, GMF, RPS)<br>
#      • <span style="color:#F39C12">Orange nodes</span>: Dates and times<br>
#      • <span style="color:#27AE60">Green nodes</span>: Other entities<br>
#      • <span style="color:#9B59B6">Purple nodes</span>: Document chunks<br>
#      • Drag nodes to reposition • Use mouse wheel to zoom • Right-click for options
#    </div>
#   <div id=\"kg\"></div>
#   <script>
#     const nodes = new vis.DataSet({json.dumps(nodes)});
#     const edges = new vis.DataSet({json.dumps(edges)});
#     const container = document.getElementById('kg');
#     const data = {{ nodes, edges }};
#     const options = {json.dumps(fallback_options)};
#     const network = new vis.Network(container, data, options);
    
#     // Add some helpful event listeners
#     network.on("stabilizationProgress", function(params) {{
#         console.log('Stabilization progress:', params.iterations + '/' + params.total);
#     }});
    
#     network.on("stabilizationIterationsDone", function() {{
#         console.log('Graph layout stabilized');
#     }});
#   </script>
# </body>
# </html>
# """
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     return out_path


# @trace_func
# def render_graph_html_with_layout(G: nx.Graph, out_path: str, layout_type: str = "force", height: str = "800px") -> str:
#     """Render graph with different layout options for better clarity.
    
#     Args:
#         G: NetworkX graph
#         out_path: Output HTML file path
#         layout_type: One of "force", "hierarchical", "circular", "grid"
#         height: Graph height in CSS units
#     """
#     # Use the clean graph function first
#     if len(G.nodes()) > 100:
#         G = create_clean_graph([], max_nodes=60)  # This is a placeholder - in practice you'd pass the docs
    
#     # Layout-specific physics settings with enhanced spacing
#     if layout_type == "force":
#         physics_options = {
#             "forceAtlas2Based": {
#                 "gravitationalConstant": -150,  # Much stronger repulsion
#                 "centralGravity": 0.005,        # Weaker central attraction
#                 "springLength": 250,            # Longer spring length
#                 "springConstant": 0.03,         # Softer springs
#                 "damping": 0.5,                 # More damping for stability
#                 "avoidOverlap": 0.95            # Maximum overlap avoidance
#             }
#         }
#     elif layout_type == "hierarchical":
#         physics_options = {
#             "hierarchicalRepulsion": {
#                 "nodeDistance": 200,            # Increased distance
#                 "centralGravity": 0.0,
#                 "springLength": 150,            # Longer springs
#                 "springConstant": 0.008,        # Softer springs
#                 "damping": 0.1,
#                 "avoidOverlap": 0.8             # Strong overlap avoidance
#             }
#         }
#     elif layout_type == "circular":
#         physics_options = {
#             "circular": {
#                 "enabled": True,
#                 "centralGravity": 0.0,
#                 "springLength": 150,            # Longer springs
#                 "springConstant": 0.008,        # Softer springs
#                 "damping": 0.1,
#                 "avoidOverlap": 0.8             # Strong overlap avoidance
#             }
#         }
#     elif layout_type == "spread":  # New layout for maximum spacing
#         physics_options = {
#             "forceAtlas2Based": {
#                 "gravitationalConstant": -200,  # Maximum repulsion
#                 "centralGravity": 0.001,        # Minimal central attraction
#                 "springLength": 300,            # Very long springs
#                 "springConstant": 0.02,         # Very soft springs
#                 "damping": 0.6,                 # High damping
#                 "avoidOverlap": 1.0             # Maximum overlap avoidance
#             }
#         }
#     elif layout_type == "cluster":  # New layout optimized for W-cases
#         physics_options = {
#             "forceAtlas2Based": {
#                 "gravitationalConstant": -180,  # Strong repulsion for clusters
#                 "centralGravity": 0.002,        # Very weak central attraction
#                 "springLength": 280,            # Long springs
#                 "springConstant": 0.025,        # Soft springs
#                 "damping": 0.55,                # High damping
#                 "avoidOverlap": 0.98            # Near maximum overlap avoidance
#             }
#         }
#     else:  # grid or default
#         physics_options = {
#             "forceAtlas2Based": {
#                 "gravitationalConstant": -120,
#                 "centralGravity": 0.005,
#                 "springLength": 200,
#                 "springConstant": 0.04,
#                 "damping": 0.4,
#                 "avoidOverlap": 0.9
#             }
#         }
    
#     # Add stabilization settings
#     physics_options["stabilization"] = {
#         "enabled": True,
#         "iterations": 2000,
#         "updateInterval": 50,
#         "fit": True
#     }
    
#     # Enhanced node styling based on type and importance
#     for n, data in G.nodes(data=True):
#         node_type = data.get("type", "unknown")
#         degree = G.degree(n)
        
#         # Size based on degree and type with better scaling
#         if node_type == "entity":
#             size = min(12 + degree * 4, 35)  # Entities get bigger
#         else:
#             size = min(10 + degree * 2.5, 25)   # Chunks get smaller
        
#         # Enhanced color coding with better contrast
#         if node_type == "entity":
#             if any(keyword in str(n).lower() for keyword in ["w1", "w2", "w3", "w4", "w5"]):
#                 color = "#E74C3C"  # Bright red for W-cases
#             elif any(keyword in str(n).lower() for keyword in ["rms", "fme", "gmf", "rps"]):
#                 color = "#3498DB"  # Blue for technical terms
#             elif any(keyword in str(n).lower() for keyword in ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]):
#                 color = "#F39C12"  # Orange for dates
#             else:
#                 color = "#27AE60"  # Green for other entities
#         elif node_type == "chunk":
#             color = "#9B59B6"  # Purple for chunks
#         else:
#             color = "#95A5A6"  # Gray for unknown
        
#         # Update node data with enhanced properties
#         data["size"] = size
#         data["color"] = color
#         data["font_size"] = max(12, size - 3)
#         data["font_weight"] = "bold" if degree > 5 else "normal"
#         data["border_width"] = 3 if degree > 8 else 2
    
#     # Now call the main rendering function with enhanced options
#     return render_graph_html(G, out_path, height)

from __future__ import annotations

from typing import List, Tuple, Sequence, Protocol, runtime_checkable
import re
import networkx as nx
import json
from app.logger import trace_func

@runtime_checkable
class DocLike(Protocol):
    page_content: str
    metadata: dict

@trace_func
def _extract_entities(text: str) -> List[str]:
    """Entity regex: W-cases, dates, times, key nouns (RPS, RMS, FME)."""
    t = text or ""
    ents = set()
    for m in re.findall(r"\bW\d{1,3}\b", t):
        ents.add(m)
    # Month names like "May 14" or abbreviations
    for m in re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b", t, re.I):
        ents.add(m)
    # Numeric dates like 2025-05-14 or 14/05/2025
    for m in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", t):
        ents.add(m)
    for m in re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t):
        ents.add(m)
    # Times like 13:51–14:17
    for m in re.findall(r"\b\d{1,2}:\d{2}(?:\s*[–-]\s*\d{1,2}:\d{2})?\b", t):
        ents.add(m)
    for kw in ("RMS", "FME", "GMF", "RPS", "crest factor", "wear depth"):
        if re.search(rf"\b{re.escape(kw)}\b", t, re.I):
            ents.add(kw)
    return list(ents)

@trace_func
def build_graph(docs: Sequence[DocLike]) -> nx.Graph:
    """Build a simple undirected graph of entities co-mentioned within the same chunk.
    Nodes: entities and file/page anchors. Edges: co-mentions.
    """
    G = nx.Graph()
    for d in docs:
        md = d.metadata or {}
        node_doc = f"{md.get('file_name','doc')}#p{md.get('page')}:{md.get('section','')}/{md.get('anchor','')}"
        G.add_node(node_doc, type="chunk", label=node_doc)
        ents = _extract_entities(d.page_content)
        for e in ents:
            if not G.has_node(e):
                G.add_node(e, type="entity", label=e)
            G.add_edge(node_doc, e)
        # connect entities co-mentioned in same chunk
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if a != b:
                    G.add_edge(a, b)
    return G

@trace_func
def create_clean_graph(docs: Sequence[DocLike], max_nodes: int = 60) -> nx.Graph:
    """Create a cleaner, more readable graph by filtering and organizing nodes.
    
    This function creates a more manageable graph by:
    1. Filtering out low-connectivity nodes
    2. Limiting total nodes for better performance
    3. Organizing nodes by importance (degree centrality)
    """
    G = nx.Graph()
    
    # First pass: build the full graph
    for d in docs:
        md = d.metadata or {}
        node_doc = f"{md.get('file_name','doc')}#p{md.get('page')}:{md.get('section','')}/{md.get('anchor','')}"
        G.add_node(node_doc, type="chunk", label=node_doc)
        ents = _extract_entities(d.page_content)
        for e in ents:
            if not G.has_node(e):
                G.add_node(e, type="entity", label=e)
            G.add_edge(node_doc, e)
        # connect entities co-mentioned in same chunk
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if a != b:
                    G.add_edge(a, b)
    
    # Second pass: clean and filter the graph
    if len(G.nodes()) > max_nodes:
        # Calculate degree centrality for all nodes
        degree_centrality = nx.degree_centrality(G)
        
        # Sort nodes by degree centrality (importance)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Keep only the most important nodes
        nodes_to_keep = [node for node, _ in sorted_nodes[:max_nodes]]
        
        # Create a subgraph with only the important nodes
        G = G.subgraph(nodes_to_keep).copy()
    
    # Remove isolated nodes (degree 0)
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated_nodes)
    
    return G


def _sanitize_text(text: str) -> str:
    """Sanitize text for safe encoding, removing problematic Unicode characters."""
    if not text:
        return ""
    
    # Convert to string and handle None
    text = str(text)
    
    # Remove or replace problematic Unicode characters
    import unicodedata
    
    # Normalize to decomposed form, then remove combining characters
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Replace common problematic characters
    replacements = {
        '"': '"',  # Smart quotes
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '…': '...',  # Ellipsis
        '°': 'deg',  # Degree symbol
        'µ': 'u',   # Micro symbol
        '≤': '<=',  # Less than or equal
        '≥': '>=',  # Greater than or equal
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Keep only ASCII printable characters and basic Unicode
    text = ''.join(c for c in text if ord(c) < 256 and (c.isprintable() or c.isspace()))
    
    # Limit length to prevent extremely long labels
    if len(text) > 50:
        text = text[:47] + "..."
    
    return text.strip()


@trace_func
def render_graph_html(G: nx.Graph, out_path: str, height: str = "800px") -> str:
    """Render the graph to an interactive HTML with improved layout and clarity.
    
    Uses enhanced physics settings, better node styling, and filtering to create
    a much more readable and organized graph visualization.
    """
    # First, try PyVis rendering with improved settings
    try:
        from pyvis.network import Network  # type: ignore
        
        # Filter graph to reduce clutter - keep only nodes with degree > 1
        # and limit total nodes for better performance
        filtered_G = G.copy()
        nodes_to_remove = [n for n in filtered_G.nodes() if filtered_G.degree(n) <= 1]
        filtered_G.remove_nodes_from(nodes_to_remove)
        
        # If still too many nodes, keep only the most connected ones
        if len(filtered_G.nodes()) > 100:
            # Sort by degree and keep top nodes
            node_degrees = [(n, filtered_G.degree(n)) for n in filtered_G.nodes()]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            nodes_to_keep = [n for n, _ in node_degrees[:100]]
            nodes_to_remove = [n for n in filtered_G.nodes() if n not in nodes_to_keep]
            filtered_G.remove_nodes_from(nodes_to_remove)
        
        # Create Network object with error handling
        net = Network(height=height, width="100%", directed=False, notebook=False, cdn_resources="in_line")
        
        # Verify network was created successfully
        if net is None:
            raise Exception("Failed to create PyVis Network object")
        
        # Check if we have any nodes to render
        if len(filtered_G.nodes()) == 0:
            print("Warning: No nodes to render in graph")
            # Create a minimal fallback
            net.add_node("empty", label="No data to display")
        else:
            # Add nodes with enhanced styling
            for n, data in filtered_G.nodes(data=True):
                node_type = data.get("type", "unknown")
                degree = filtered_G.degree(n)
                
                # Use enhanced size calculation
                if node_type == "entity":
                    size = min(12 + degree * 4, 35)  # Entities get bigger
                else:
                    size = min(10 + degree * 2.5, 25)   # Chunks get smaller
                
                # Use enhanced color coding
                if node_type == "entity":
                    if any(keyword in str(n).lower() for keyword in ["w1", "w2", "w3", "w4", "w5"]):
                        color = "#E74C3C"  # Bright red for W-cases
                    elif any(keyword in str(n).lower() for keyword in ["rms", "fme", "gmf", "rps"]):
                        color = "#3498DB"  # Blue for technical terms
                    elif any(keyword in str(n).lower() for keyword in ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]):
                        color = "#F39C12"  # Orange for dates
                    else:
                        color = "#27AE60"  # Green for other entities
                elif node_type == "chunk":
                    color = "#9B59B6"  # Purple for chunks
                else:
                    color = "#95A5A6"  # Gray for unknown
                
                # Add node with enhanced properties
                # Sanitize label to prevent encoding issues
                original_label = data.get("label", str(n))
                sanitized_label = _sanitize_text(original_label)
                
                net.add_node(
                    n, 
                    label=sanitized_label,  # Use sanitized label
                    color=color,
                    size=size,
                    font={
                        "size": max(12, size - 3), 
                        "face": "Arial", 
                        "color": "#000000",
                        "strokeWidth": 2,
                        "strokeColor": "#FFFFFF"
                    },
                    borderWidth=3 if degree > 8 else 2,
                    borderColor="#000000",
                    shadow=True,
                    mass=degree + 1  # Heavier nodes for better physics
                )
            
            # Add edges with improved styling
            for u, v in filtered_G.edges():
                net.add_edge(
                    str(u), 
                    str(v),
                    width=1,
                    color={"color": "#848484", "opacity": 0.6},
                    smooth={"type": "continuous", "forceDirection": "none"}
                )
        
        # Enhanced physics and layout settings with maximum spacing
        physics_options = {
            "forceAtlas2Based": {
                "gravitationalConstant": -150,  # Much stronger repulsion
                "centralGravity": 0.003,       # Very weak central attraction
                "springLength": 250,           # Much longer spring length
                "springConstant": 0.03,        # Softer springs
                "damping": 0.5,                # More damping for stability
                "avoidOverlap": 0.95           # Maximum overlap avoidance
            },
            "stabilization": {
                "enabled": True,
                "iterations": 2000,            # More iterations for better layout
                "updateInterval": 50,
                "fit": True
            },
            "minVelocity": 0.75,
            "maxVelocity": 30
        }
        
        # Enhanced interaction settings
        interaction_options = {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": True,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
            "dragView": True
        }
        
        # Enhanced node settings
        node_options = {
            "shape": "dot",
            "shadow": True,
            "borderWidth": 2,
            "borderColor": "#000000"
        }
        
        # Enhanced edge settings
        edge_options = {
            "smooth": {"type": "continuous", "forceDirection": "none"},
            "color": {"color": "#848484", "opacity": 0.6},
            "width": 1
        }
        
        # Combine all options
        all_options = {
            "nodes": node_options,
            "edges": edge_options,
            "physics": physics_options,
            "interaction": interaction_options
        }
        
        net.set_options(json.dumps(all_options))
        
        try:
            # Try writing with explicit UTF-8 encoding to handle Unicode characters
            net.write_html(out_path, encoding='utf-8')
        except UnicodeEncodeError as encoding_error:
            print(f"PyVis encoding error: {encoding_error}")
            print("Attempting to sanitize node labels and retry...")
            # Sanitize node labels to remove problematic Unicode characters
            try:
                for node_id in net.get_nodes():
                    node = net.get_node(node_id)
                    if 'label' in node:
                        # Use our comprehensive sanitization function
                        node['label'] = _sanitize_text(node['label'])
                net.write_html(out_path, encoding='utf-8')
            except Exception as retry_error:
                print(f"PyVis retry after sanitization failed: {retry_error}")
                raise Exception(f"PyVis encoding issue could not be resolved: {encoding_error}")
        except Exception as write_error:
            print(f"PyVis write_html failed: {write_error}")
            # Don't try net.show() as it's unreliable, go straight to fallback
            raise Exception(f"PyVis write_html failed: {write_error}")
        
        return out_path
        
    except Exception as e:
        print(f"PyVis rendering failed: {e}")
        pass

    # Fallback: write an improved vis-network HTML manually
    # Filter graph for fallback too
    filtered_G = G.copy()
    nodes_to_remove = [n for n in filtered_G.nodes() if filtered_G.degree(n) <= 1]
    filtered_G.remove_nodes_from(nodes_to_remove)
    
    if len(filtered_G.nodes()) > 100:
        node_degrees = [(n, filtered_G.degree(n)) for n in filtered_G.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = [n for n, _ in node_degrees[:100]]
        nodes_to_remove = [n for n in filtered_G.nodes() if n not in nodes_to_keep]
        filtered_G.remove_nodes_from(nodes_to_remove)
    
    nodes = []
    for n, data in filtered_G.nodes(data=True):
        node_type = data.get("type", "unknown")
        degree = filtered_G.degree(n)
        
        # Use enhanced size calculation
        if node_type == "entity":
            size = min(12 + degree * 4, 35)  # Entities get bigger
        else:
            size = min(10 + degree * 2.5, 25)   # Chunks get smaller
        
        # Use enhanced color coding
        if node_type == "entity":
            if any(keyword in str(n).lower() for keyword in ["w1", "w2", "w3", "w4", "w5"]):
                color = "#E74C3C"  # Bright red for W-cases
            elif any(keyword in str(n).lower() for keyword in ["rms", "fme", "gmf", "rps"]):
                color = "#3498DB"  # Blue for technical terms
            elif any(keyword in str(n).lower() for keyword in ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]):
                color = "#F39C12"  # Orange for dates
            else:
                color = "#27AE60"  # Green for other entities
        elif node_type == "chunk":
            color = "#9B59B6"  # Purple for chunks
        else:
            color = "#95A5A6"  # Gray for unknown
            
        nodes.append({
            "id": str(n),
            "label": _sanitize_text(data.get("label", str(n))),  # Use comprehensive sanitization
            "color": color,
            "size": size,
            "font": {
                "size": max(12, size - 3), 
                "face": "Arial", 
                "color": "#000000",
                "strokeWidth": 2,
                "strokeColor": "#FFFFFF"
            },
            "borderWidth": 3 if degree > 8 else 2,
            "borderColor": "#000000",
            "shadow": True,
            "mass": degree + 1
        })
    
    edges = [{"from": str(u), "to": str(v), "width": 1, "color": {"color": "#848484", "opacity": 0.6}} for u, v in filtered_G.edges()]
    
    # Enhanced fallback options
    fallback_options = {
        "nodes": {
            "shape": "dot",
            "shadow": True,
            "borderWidth": 2,
            "borderColor": "#000000"
        },
        "edges": {
            "smooth": {"type": "continuous", "forceDirection": "none"},
            "color": {"color": "#848484", "opacity": 0.6},
            "width": 1
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.005,
                "springLength": 150,
                "springConstant": 0.05,
                "damping": 0.3,
                "avoidOverlap": 0.8
            },
            "stabilization": {
                "enabled": True,
                "iterations": 2000,
                "updateInterval": 50,
                "fit": True
            },
            "minVelocity": 0.75,
            "maxVelocity": 30
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": True,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
            "dragView": True
        }
    }
    
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Knowledge Graph - Enhanced Layout</title>
  <script src=\"https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js\"></script>
  <style>
    body {{ margin: 0; padding: 10px; font-family: Arial, sans-serif; }}
    #kg {{ width: 100%; height: {height}; border: 2px solid #ddd; border-radius: 8px; }}
    .controls {{ margin-bottom: 10px; }}
    .info {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
  </style>
</head>
<body>
     <div class="info">
     <strong>Enhanced Knowledge Graph</strong><br>
     • Nodes sized by connectivity (more connections = bigger)<br>
     • <span style="color:#E74C3C">Red nodes</span>: W-cases (W1, W2, W3, etc.)<br>
     • <span style="color:#3498DB">Blue nodes</span>: Technical terms (RMS, FME, GMF, RPS)<br>
     • <span style="color:#F39C12">Orange nodes</span>: Dates and times<br>
     • <span style="color:#27AE60">Green nodes</span>: Other entities<br>
     • <span style="color:#9B59B6">Purple nodes</span>: Document chunks<br>
     • Drag nodes to reposition • Use mouse wheel to zoom • Right-click for options
   </div>
  <div id=\"kg\"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});
    const container = document.getElementById('kg');
    const data = {{ nodes, edges }};
    const options = {json.dumps(fallback_options)};
    const network = new vis.Network(container, data, options);
    
    // Add some helpful event listeners
    network.on("stabilizationProgress", function(params) {{
        console.log('Stabilization progress:', params.iterations + '/' + params.total);
    }});
    
    network.on("stabilizationIterationsDone", function() {{
        console.log('Graph layout stabilized');
    }});
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


@trace_func
def render_graph_html_with_layout(G: nx.Graph, out_path: str, layout_type: str = "force", height: str = "800px") -> str:
    """Render graph with different layout options for better clarity.
    
    Args:
        G: NetworkX graph
        out_path: Output HTML file path
        layout_type: One of "force", "hierarchical", "circular", "grid"
        height: Graph height in CSS units
    """
    # Use the clean graph function first
    if len(G.nodes()) > 100:
        G = create_clean_graph([], max_nodes=60)  # This is a placeholder - in practice you'd pass the docs
    
    # Layout-specific physics settings with enhanced spacing
    if layout_type == "force":
        physics_options = {
            "forceAtlas2Based": {
                "gravitationalConstant": -150,  # Much stronger repulsion
                "centralGravity": 0.005,        # Weaker central attraction
                "springLength": 250,            # Longer spring length
                "springConstant": 0.03,         # Softer springs
                "damping": 0.5,                 # More damping for stability
                "avoidOverlap": 0.95            # Maximum overlap avoidance
            }
        }
    elif layout_type == "hierarchical":
        physics_options = {
            "hierarchicalRepulsion": {
                "nodeDistance": 200,            # Increased distance
                "centralGravity": 0.0,
                "springLength": 150,            # Longer springs
                "springConstant": 0.008,        # Softer springs
                "damping": 0.1,
                "avoidOverlap": 0.8             # Strong overlap avoidance
            }
        }
    elif layout_type == "circular":
        physics_options = {
            "circular": {
                "enabled": True,
                "centralGravity": 0.0,
                "springLength": 150,            # Longer springs
                "springConstant": 0.008,        # Softer springs
                "damping": 0.1,
                "avoidOverlap": 0.8             # Strong overlap avoidance
            }
        }
    elif layout_type == "spread":  # New layout for maximum spacing
        physics_options = {
            "forceAtlas2Based": {
                "gravitationalConstant": -200,  # Maximum repulsion
                "centralGravity": 0.001,        # Minimal central attraction
                "springLength": 300,            # Very long springs
                "springConstant": 0.02,         # Very soft springs
                "damping": 0.6,                 # High damping
                "avoidOverlap": 1.0             # Maximum overlap avoidance
            }
        }
    elif layout_type == "cluster":  # New layout optimized for W-cases
        physics_options = {
            "forceAtlas2Based": {
                "gravitationalConstant": -180,  # Strong repulsion for clusters
                "centralGravity": 0.002,        # Very weak central attraction
                "springLength": 280,            # Long springs
                "springConstant": 0.025,        # Soft springs
                "damping": 0.55,                # High damping
                "avoidOverlap": 0.98            # Near maximum overlap avoidance
            }
        }
    else:  # grid or default
        physics_options = {
            "forceAtlas2Based": {
                "gravitationalConstant": -120,
                "centralGravity": 0.005,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.4,
                "avoidOverlap": 0.9
            }
        }
    
    # Add stabilization settings
    physics_options["stabilization"] = {
        "enabled": True,
        "iterations": 2000,
        "updateInterval": 50,
        "fit": True
    }
    
    # Enhanced node styling based on type and importance
    for n, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        degree = G.degree(n)
        
        # Size based on degree and type with better scaling
        if node_type == "entity":
            size = min(12 + degree * 4, 35)  # Entities get bigger
        else:
            size = min(10 + degree * 2.5, 25)   # Chunks get smaller
        
        # Enhanced color coding with better contrast
        if node_type == "entity":
            if any(keyword in str(n).lower() for keyword in ["w1", "w2", "w3", "w4", "w5"]):
                color = "#E74C3C"  # Bright red for W-cases
            elif any(keyword in str(n).lower() for keyword in ["rms", "fme", "gmf", "rps"]):
                color = "#3498DB"  # Blue for technical terms
            elif any(keyword in str(n).lower() for keyword in ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]):
                color = "#F39C12"  # Orange for dates
            else:
                color = "#27AE60"  # Green for other entities
        elif node_type == "chunk":
            color = "#9B59B6"  # Purple for chunks
        else:
            color = "#95A5A6"  # Gray for unknown
        
        # Update node data with enhanced properties
        data["size"] = size
        data["color"] = color
        data["font_size"] = max(12, size - 3)
        data["font_weight"] = "bold" if degree > 5 else "normal"
        data["border_width"] = 3 if degree > 8 else 2
    
    # Now call the main rendering function with enhanced options
    return render_graph_html(G, out_path, height)