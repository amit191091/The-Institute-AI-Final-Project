from __future__ import annotations
from app.logger import trace_func
"""
Neo4j graph database integration: build a persistent graph from docs and run Cypher queries.

Env vars required:
  NEO4J_URI=bolt://localhost:7687 (or neo4j+s://... for Aura)
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=...

Enable via RAG_GRAPH_DB=true to populate after ingestion.
"""

import os
from typing import Any, Dict, Iterable, List, Tuple, Sequence

# Load .env lazily (non-fatal if missing)
try:
    from dotenv import dotenv_values, find_dotenv  # type: ignore
    try:
        env_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
        if env_path:
            for k, v in (dotenv_values(env_path) or {}).items():
                if v is not None and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass
except Exception:
    pass


try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore

from app.graph import _extract_entities, DocLike  # reuse same entity logic
try:  # neo4j v5
    from neo4j.graph import Node as _Neo4jNode, Relationship as _Neo4jRel, Path as _Neo4jPath  # type: ignore
except Exception:  # pragma: no cover
    _Neo4jNode = None  # type: ignore
    _Neo4jRel = None  # type: ignore
    _Neo4jPath = None  # type: ignore

@trace_func
def _get_driver():
    uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    if GraphDatabase is None:
        return None
    if not (uri and user and pwd):
        return None
    return GraphDatabase.driver(uri, auth=(user, pwd))

@trace_func
def _chunk_node_id(md: dict) -> str:
    return f"{md.get('file_name','doc')}#p{md.get('page')}:{md.get('section','')}/{md.get('anchor','')}"

@trace_func
def build_graph_db(docs: Sequence[DocLike]) -> int:
    """Create/merge Chunk, Entity, Table and TableCell nodes in Neo4j.

    Nodes:
      (:Chunk {id, file, page:int, section, anchor})
      (:Entity {name, type})
      (:Table {doc_id, number:int, file, page, label, anchor})
      (:TableCell {id, doc_id, number:int, key, value, value_num:float, unit, file, page})

    Rels:
      (:Chunk)-[:MENTIONS]->(:Entity)
      (:Chunk)-[:HAS_TABLE]->(:Table)
      (:Table)-[:HAS_CELL]->(:TableCell)
      (:Chunk)-[:HAS_CAPTION_TEXT]->(:Chunk)  // figure/table -> caption text chunk

    Returns: rough count of upserts performed.
    """
    drv = _get_driver()
    if drv is None:
        return 0
    count = 0
    db = os.getenv("NEO4J_DATABASE") or "neo4j"
    constraint_cypher = [
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT table_key IF NOT EXISTS FOR (t:Table) REQUIRE (t.doc_id, t.number) IS NODE KEY",
        "CREATE CONSTRAINT tablecell_id IF NOT EXISTS FOR (x:TableCell) REQUIRE x.id IS UNIQUE",
    ]

    # Pre-scan text chunks for caption target anchors
    text_anchor_index: dict[tuple[str, int | None, str], str] = {}
    for d in docs:
        md = (getattr(d, "metadata", None) or {})
        sec = md.get("section") or md.get("section_type")
        if sec == "Text":
            key = (str(md.get("doc_id")), md.get("page"), str(md.get("anchor")))
            text_anchor_index[key] = _chunk_node_id(md)

    with drv.session(database=db) as s:
        # Constraints
        for c in constraint_cypher:
            try:
                s.run(c)  # type: ignore[arg-type]
            except Exception:
                pass

        caption_edges: list[tuple[str, str]] = []
        # Pass 1: Chunks, Entities, Tables, caption edges planning
        for d in docs:
            md = (getattr(d, "metadata", None) or {})
            cid = _chunk_node_id(md)
            file = md.get("file_name"); page = md.get("page"); section = md.get("section") or md.get("section_type"); anchor = md.get("anchor")
            text = getattr(d, "page_content", "") or ""

            # Chunk
            s.run(
                "MERGE (c:Chunk {id:$id}) SET c.file=$file, c.page=toInteger($page), c.section=$section, c.anchor=$anchor",
                {"id": cid, "file": file, "page": page, "section": section, "anchor": anchor},
            )
            count += 1

            # Entities
            ents = _extract_entities(text)
            for e in ents:
                s.run("MERGE (x:Entity {name:$name}) ON CREATE SET x.type=$type", {"name": e, "type": "entity"})
                s.run("MATCH (c:Chunk {id:$id}), (x:Entity {name:$name}) MERGE (c)-[:MENTIONS]->(x)", {"id": cid, "name": e})
                count += 1

            # Tables and caption links
            if section == "Table":
                try:
                    doc_id = str(md.get("doc_id")) if md.get("doc_id") is not None else ""
                    tn_val = md.get("table_number")
                    tn = int(tn_val) if tn_val is not None else None
                except Exception:
                    tn = None
                s.run(
                    "MERGE (t:Table {doc_id:$doc, number:toInteger($num)}) SET t.file=$file, t.page=toInteger($page), t.label=$label, t.anchor=$anchor",
                    {"doc": doc_id, "num": tn, "file": file, "page": page, "label": md.get("table_label"), "anchor": anchor},
                )
                s.run(
                    "MATCH (c:Chunk {id:$cid}), (t:Table {doc_id:$doc, number:toInteger($num)}) MERGE (c)-[:HAS_TABLE]->(t)",
                    {"cid": cid, "doc": doc_id, "num": tn},
                )
                count += 2
                ta = md.get("table_associated_anchor")
                if ta is not None:
                    key = (doc_id, page, str(ta))
                    dst = text_anchor_index.get(key)
                    if dst:
                        caption_edges.append((cid, dst))
            elif section == "Figure":
                doc_id = str(md.get("doc_id")) if md.get("doc_id") is not None else ""
                fa = md.get("figure_associated_anchor")
                if fa is not None:
                    key = (doc_id, page, str(fa))
                    dst = text_anchor_index.get(key)
                    if dst:
                        caption_edges.append((cid, dst))

        # Pass 2: TableCell nodes from KV mini-docs
        for d in docs:
            md = (getattr(d, "metadata", None) or {})
            section = md.get("section") or md.get("section_type")
            if section != "TableCell":
                continue
            doc_id = str(md.get("doc_id")) if md.get("doc_id") is not None else ""
            file = md.get("file_name"); page = md.get("page")
            tn_val = md.get("table_number")
            try:
                tn = int(tn_val) if tn_val is not None else None
            except Exception:
                tn = None
            key = str(md.get("kv_key") or "").strip()
            val_raw = str(md.get("kv_value") or "").strip()
            # numeric/unit split
            val_num = None; unit = None
            try:
                import re as _re
                m = _re.match(r"^\s*([+-]?\d+(?:[\.,]\d+)?)\s*([A-Za-z°%/]+)?\s*$", val_raw)
                if m:
                    val_num = float(m.group(1).replace(",", "."))
                    unit = m.group(2)
            except Exception:
                pass
            cell_id = f"{doc_id}#table-{tn}:{abs(hash(key+val_raw)) & 0xFFFFFFFF}"
            s.run(
                "MERGE (x:TableCell {id:$id}) SET x.doc_id=$doc, x.number=toInteger($num), x.key=$key, x.value=$val, x.value_num=$vnum, x.unit=$unit, x.file=$file, x.page=toInteger($page)",
                {"id": cell_id, "doc": doc_id, "num": tn, "key": key, "val": val_raw, "vnum": val_num, "unit": unit, "file": file, "page": page},
            )
            s.run(
                "MATCH (t:Table {doc_id:$doc, number:toInteger($num)}), (x:TableCell {id:$id}) MERGE (t)-[:HAS_CELL]->(x)",
                {"doc": doc_id, "num": tn, "id": cell_id},
            )
            count += 2
            ents = _extract_entities(f"{key}: {val_raw}")
            for e in ents:
                s.run("MERGE (z:Entity {name:$name}) ON CREATE SET z.type=$type", {"name": e, "type": "entity"})
                s.run("MATCH (x:TableCell {id:$id}), (z:Entity {name:$name}) MERGE (x)-[:MENTIONS]->(z)", {"id": cell_id, "name": e})
                count += 1

        # Pass 3: Create caption edges
        for src, dst in caption_edges:
            s.run(
                "MATCH (a:Chunk {id:$a}), (b:Chunk {id:$b}) MERGE (a)-[:HAS_CAPTION_TEXT]->(b)",
                {"a": src, "b": dst},
            )
            count += 1
    return count


@trace_func
def query_table_cells(number: int, key: str, doc_id: str | None = None, limit: int = 5) -> list[dict]:
    """Find table cells by table number and a key substring (case-insensitive).

    Returns list of dicts with value, unit, file, page, doc_id, key.
    """
    drv = _get_driver()
    if drv is None:
        return []
    db = os.getenv("NEO4J_DATABASE") or "neo4j"
    cy = (
        "MATCH (t:Table {number:toInteger($num)})-[:HAS_CELL]->(x:TableCell) "
        "WHERE toLower(x.key) CONTAINS toLower($key) "
        "RETURN x.value AS value, x.unit AS unit, x.file AS file, x.page AS page, t.doc_id AS doc_id, x.key AS key "
        "LIMIT toInteger($lim)"
    )
    cy_doc = (
        "MATCH (t:Table {doc_id:$doc, number:toInteger($num)})-[:HAS_CELL]->(x:TableCell) "
        "WHERE toLower(x.key) CONTAINS toLower($key) "
        "RETURN x.value AS value, x.unit AS unit, x.file AS file, x.page AS page, t.doc_id AS doc_id, x.key AS key "
        "LIMIT toInteger($lim)"
    )
    with drv.session(database=db) as s:
        res = s.run(cy_doc if doc_id else cy, {"doc": doc_id, "num": number, "key": key, "lim": limit})
        return [r.data() for r in res]


@trace_func
def get_table_caption_text_chunk_ids(number: int, doc_id: str | None = None, limit: int = 3) -> list[str]:
    """Return caption-linked text Chunk ids for a given table.

    These are the targets of HAS_CAPTION_TEXT from the table's Chunk.
    """
    drv = _get_driver()
    if drv is None:
        return []
    db = os.getenv("NEO4J_DATABASE") or "neo4j"
    base = (
        "MATCH (c:Chunk)-[:HAS_TABLE]->(t:Table {number:toInteger($num)})-[:HAS_CELL]->(:TableCell) "
        "WITH DISTINCT c, t "
        "MATCH (c)-[:HAS_CAPTION_TEXT]->(cap:Chunk) "
        "RETURN DISTINCT cap.id AS id LIMIT toInteger($lim)"
    )
    with_doc = (
        "MATCH (c:Chunk)-[:HAS_TABLE]->(t:Table {doc_id:$doc, number:toInteger($num)})-[:HAS_CELL]->(:TableCell) "
        "WITH DISTINCT c, t "
        "MATCH (c)-[:HAS_CAPTION_TEXT]->(cap:Chunk) "
        "RETURN DISTINCT cap.id AS id LIMIT toInteger($lim)"
    )
    with drv.session(database=db) as s:
        res = s.run(with_doc if doc_id else base, {"doc": doc_id, "num": number, "lim": limit})
        return [r["id"] for r in res]


from typing import LiteralString
@trace_func
def run_cypher(query: LiteralString, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    # Build clearer diagnostics if driver cannot be created
    drv = _get_driver()
    if drv is None:
        missing = []
        if GraphDatabase is None:
            missing.append("python package 'neo4j' not installed")
        # Check both primary keys and aliases for clarity
        if not (os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")):
            missing.append("missing NEO4J_URI/NEO4J_URL")
        if not (os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")):
            missing.append("missing NEO4J_USER/NEO4J_USERNAME")
        if not (os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")):
            missing.append("missing NEO4J_PASSWORD/NEO4J_PASS")
        hint = "; ".join(missing) if missing else "unknown cause"
        raise RuntimeError(f"Neo4j driver not configured ({hint}); set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
    def _coerce_graph_value(v: Any) -> Any:
        # Convert Neo4j values to JSON-friendly structures
        try:
            if _Neo4jNode is not None and isinstance(v, _Neo4jNode):
                return {
                    "type": "Node",
                    "element_id": getattr(v, "element_id", None),
                    "labels": list(getattr(v, "labels", []) or []),
                    "properties": dict(v),
                }
            if _Neo4jRel is not None and isinstance(v, _Neo4jRel):
                return {
                    "type": "Relationship",
                    "element_id": getattr(v, "element_id", None),
                    "rel_type": getattr(v, "type", None),
                    "start": getattr(v, "start_node", None) and getattr(getattr(v, "start_node"), "element_id", None),
                    "end": getattr(v, "end_node", None) and getattr(getattr(v, "end_node"), "element_id", None),
                    "properties": dict(v),
                }
            if _Neo4jPath is not None and isinstance(v, _Neo4jPath):
                return {
                    "type": "Path",
                    "nodes": [_coerce_graph_value(n) for n in getattr(v, "nodes", [])],
                    "relationships": [_coerce_graph_value(r) for r in getattr(v, "relationships", [])],
                }
        except Exception:
            pass
        if isinstance(v, list):
            return [_coerce_graph_value(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_coerce_graph_value(x) for x in v)
        if isinstance(v, dict):
            return {k: _coerce_graph_value(val) for k, val in v.items()}
        return v

    db = os.getenv("NEO4J_DATABASE") or "neo4j"
    with drv.session(database=db) as s:
        res = s.run(query, parameters or {})
        rows: List[Dict[str, Any]] = []
        for r in res:
            try:
                items = r.items()
            except Exception:
                # Older drivers
                items = [(k, r[k]) for k in r.keys()]
            row = {k: _coerce_graph_value(v) for k, v in items}
            rows.append(row)
        return rows
