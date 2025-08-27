"""
Graph and database event handlers for the Gradio interface.
"""
import json
import re
from pathlib import Path
import gradio as gr

from RAG.app.graphdb import run_cypher as _run_cypher
from RAG.app.normalized_loader import load_normalized_docs

try:
    from RAG.app.graph import build_graph, render_graph_html
except Exception:
    build_graph = None  # type: ignore
    render_graph_html = None  # type: ignore


def _gen_graph(source_choice: str, docs):
    """Generate graph from various sources."""
    try:
        if render_graph_html is None:
            return gr.update(value=""), "(graph module not available; install dependencies: networkx, pyvis)"
        # Select source
        if source_choice == "Docs co-mention (default)":
            if build_graph is None:
                return gr.update(value=""), "(build_graph not available)"
            G = build_graph(docs)
        elif source_choice == "Normalized graph.json":
            G = _build_graph_from_normalized_json()
        elif source_choice == "Normalized chunks":
            G = _build_graph_from_normalized_chunks()
        elif source_choice == "Neo4j (live)":
            G = _build_graph_from_neo4j()
        else:
            if build_graph is None:
                return gr.update(value=""), "(unknown source)"
            G = build_graph(docs)
        from RAG.app.config import settings
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        out = settings.LOGS_DIR/"graph.html"
        render_graph_html(G, str(out))
        html_data = out.read_text(encoding="utf-8")
        # Best-effort inline via iframe srcdoc; also provide a link for full view
        import html as _html
        iframe = f"<p><a href='file:///{out.resolve().as_posix()}' target='_blank'>Open graph.html in browser</a></p>" \
                 f"<iframe style='width:100%;height:650px;border:1px solid #ddd' srcdoc=\"{_html.escape(html_data)}\"></iframe>"
        return gr.update(value=iframe), "Graph updated."
    except Exception as e:
        return gr.update(value=""), f"(failed to build graph: {e})"


def _build_graph_from_normalized_json():
    """Build graph from normalized JSON file."""
    from pathlib import Path as _P
    import json as _json
    import networkx as _nx
    from RAG.app.config import settings
    p = settings.LOGS_DIR/"normalized"/"graph.json"
    if not p.exists():
        raise RuntimeError("RAG/logs/normalized/graph.json not found")
    data = _json.loads(p.read_text(encoding="utf-8"))
    G2 = _nx.Graph()
    def _node_label(n):
        t = n.get("type")
        pid = str(n.get("id"))
        props = n.get("props") or {}
        if t == "Figure":
            return props.get("caption") or pid
        if t == "Table":
            return props.get("title") or pid
        if t == "Section":
            return props.get("title") or pid
        if t == "Event":
            return props.get("date") or pid
        return pid
    for n in (data.get("nodes") or []):
        nid = str(n.get("id"))
        G2.add_node(nid, type=n.get("type"), label=_node_label(n))
    for e in (data.get("edges") or []):
        u = str(e.get("from")); v = str(e.get("to"))
        if u and v:
            G2.add_edge(u, v, type=e.get("type"))
    return G2

def _build_graph_from_normalized_chunks():
    """Build graph from normalized chunks."""
    import networkx as _nx
    from RAG.app.config import settings
    p = settings.LOGS_DIR/"normalized"/"chunks.jsonl"
    ndocs = load_normalized_docs(p)
    if not ndocs:
        raise RuntimeError("RAG/logs/normalized/chunks.jsonl not found or empty")
    G3 = _nx.Graph()
    for d in ndocs:
        md = d.metadata or {}
        cid = str(md.get("chunk_id") or md.get("anchor") or id(d))
        clabel = f"chunk:{cid}"
        G3.add_node(clabel, type="Chunk", label=clabel)
        # Page linkage
        file = md.get("file_name"); page = md.get("page")
        if file and page is not None:
            pid = f"{file}#p{page}"
            G3.add_node(pid, type="Page", label=pid)
            G3.add_edge(clabel, pid, type="ON_PAGE")
        # Table/Figure linkage
        if md.get("table_number"):
            tnode = f"tbl:{int(md['table_number'])}"
            G3.add_node(tnode, type="Table", label=md.get("table_label") or tnode)
            G3.add_edge(clabel, tnode, type="REFERS_TO")
        if md.get("figure_number"):
            fnode = f"fig:{int(md['figure_number'])}"
            G3.add_node(fnode, type="Figure", label=md.get("figure_label") or fnode)
            G3.add_edge(clabel, fnode, type="REFERS_TO")
    return G3

def _build_graph_from_neo4j():
    """Build graph from Neo4j database."""
    import networkx as _nx
    rows = _run_cypher("MATCH (a)-[r]->(b) RETURN a, type(r) as t, b LIMIT 200")  # type: ignore
    G4 = _nx.Graph()
    def _nid(x):
        try:
            props = x.get("properties") or {}
            labels = x.get("labels") or []
            label = labels[0] if labels else "Node"
            # Prefer stable id keys
            for k in ("id", "name", "title", "date"):
                if k in props and props[k]:
                    return f"{label}:{props[k]}"
            return f"{label}:{props}"  # fallback
        except Exception:
            return str(x)
    for r in rows or []:
        a = r.get("a") or r.get("A") or r.get("n")
        b = r.get("b") or r.get("B") or r.get("m")
        t = r.get("t") or "REL"
        if not a or not b:
            continue
        na = _nid(a); nb = _nid(b)
        G4.add_node(na, type="Neo4j")
        G4.add_node(nb, type="Neo4j")
        G4.add_edge(na, nb, type=str(t))
    return G4

def _run_cypher_ui(q: str = ""):
    """Run Cypher query in UI."""
    try:
        query = (q or "").strip()
        if not query:
            return {"error": "empty query", "hint": "Example: MATCH (n) RETURN n LIMIT 10"}
        rows = _run_cypher(query)  # type: ignore
        # Return rows directly; gr.JSON can render lists of dicts
        return rows if rows else {"rows": [], "note": "Query ran, no results returned."}
    except Exception as e:
        # Surface config issues clearly (e.g., missing Neo4j env vars)
        return {"error": str(e)}
