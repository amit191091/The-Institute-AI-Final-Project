from __future__ import annotations

from typing import List, Tuple, Sequence, Protocol, runtime_checkable
import re
import networkx as nx
import json

from RAG.app.logger import trace_func

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
    # Times like 13:51Γאף14:17
    for m in re.findall(r"\b\d{1,2}:\d{2}(?:\s*[Γאף-]\s*\d{1,2}:\d{2})?\b", t):
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
def render_graph_html(G: nx.Graph, out_path: str, height: str = "600px") -> str:
    """Render the graph to an interactive HTML. Prefer pyvis, fallback to a minimal vis-network HTML.

    This avoids failures when Jinja2 templates are unavailable in some environments.
    """
    # First, try PyVis rendering
    try:
        from pyvis.network import Network  # type: ignore
        net = Network(height=height, width="100%", directed=False, notebook=False, cdn_resources="in_line")
        for n, data in G.nodes(data=True):
            net.add_node(n, label=data.get("label", str(n)), color="#6aa84f" if data.get("type") == "entity" else "#3c78d8")
        for u, v in G.edges():
            net.add_edge(str(u), str(v))
        # set_options expects a JSON string, not a JS object literal
        net.set_options(
            '{"nodes": {"shape": "dot", "size": 12}, '
            '"physics": {"stabilization": true}, '
            '"interaction": {"hover": true, "tooltipDelay": 150}}'
        )
        try:
            net.write_html(out_path)  # avoid opening browser
        except Exception:
            net.show(out_path)
        return out_path
    except Exception:
        pass

    # Fallback: write a minimal vis-network HTML manually
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({
            "id": str(n),
            "label": data.get("label", str(n)),
            "color": "#6aa84f" if data.get("type") == "entity" else "#3c78d8",
        })
    edges = [{"from": str(u), "to": str(v)} for u, v in G.edges()]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Knowledge Graph</title>
  <script src=\"https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js\"></script>
  <style>
    #kg {{ width: 100%; height: {height}; border: 1px solid #ccc; }}
  </style>
  </head>
<body>
  <div id=\"kg\"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});
    const container = document.getElementById('kg');
    const data = {{ nodes, edges }};
    const options = {{ nodes: {{ shape: 'dot', size: 12 }}, physics: {{ stabilization: true }}, interaction: {{ hover: true, tooltipDelay: 150 }} }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
