from __future__ import annotations

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

try:
    # Ensure .env is loaded even if caller didn't load it yet (won't override existing env)
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

from RAG.app.pipeline_modules.graph import _extract_entities, DocLike  # reuse same entity logic
from RAG.app.logger import trace_func
try:  # neo4j v5
    from neo4j.graph import Node as _Neo4jNode, Relationship as _Neo4jRel, Path as _Neo4jPath  # type: ignore
except Exception:  # pragma: no cover
    _Neo4jNode = None  # type: ignore
    _Neo4jRel = None  # type: ignore
    _Neo4jPath = None  # type: ignore


@trace_func
def _get_driver():
    # Support common aliases from different guides
    uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    if GraphDatabase is None:
        # neo4j package not installed
        return None
    if not (uri and user and pwd):
        return None
    # Use a short connection timeout to avoid long hangs if DB is unreachable
    timeout_s = None
    try:
        timeout_s = float(os.getenv("NEO4J_CONN_TIMEOUT", "5"))
    except Exception:
        timeout_s = None
    try:
        if timeout_s is not None:
            return GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=timeout_s)  # type: ignore[call-arg]
        return GraphDatabase.driver(uri, auth=(user, pwd))
    except TypeError:
        # Older drivers may not support connection_timeout kwarg
        return GraphDatabase.driver(uri, auth=(user, pwd))


@trace_func
def _chunk_node_id(md: dict) -> str:
    return f"{md.get('file_name','doc')}#p{md.get('page')}:{md.get('section','')}/{md.get('anchor','')}"


@trace_func
def build_graph_db(docs: Sequence[DocLike]) -> int:
    """Create/merge chunk and entity nodes, with relationships in Neo4j.

    Schema:
      (:Chunk {id, file, page:int, section, anchor})
      (:Entity {name, type})
      (:Chunk)-[:MENTIONS]->(:Entity)
      (:Entity)-[:CO_OCCUR]->(:Entity)

    Returns number of nodes/edges upserted (rough estimate).
    """
    drv = _get_driver()
    if drv is None:
        return 0
    count = 0
    # Create constraints (safe to retry)
    constraint_cypher = [
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    ]
    db = os.getenv("NEO4J_DATABASE") or "neo4j"
    with drv.session(database=db) as s:
        for c in constraint_cypher:
            try:
                s.run(c)  # type: ignore[arg-type]
            except Exception:
                pass
        # Upsert in small batches
        for d in docs:
            md = (getattr(d, "metadata", None) or {})
            cid = _chunk_node_id(md)
            file = md.get("file_name"); page = md.get("page"); section = md.get("section"); anchor = md.get("anchor")
            text = getattr(d, "page_content", "") or ""
            ents = _extract_entities(text)
            # MERGE chunk
            s.run(
                "MERGE (c:Chunk {id:$id}) SET c.file=$file, c.page=toInteger($page), c.section=$section, c.anchor=$anchor",
                {"id": cid, "file": file, "page": page, "section": section, "anchor": anchor},
            )
            count += 1
            # MERGE entities and relationships
            for e in ents:
                s.run(
                    "MERGE (x:Entity {name:$name}) ON CREATE SET x.type=$type",
                    {"name": e, "type": "entity"},
                )
                s.run(
                    "MATCH (c:Chunk {id:$id}), (x:Entity {name:$name}) MERGE (c)-[:MENTIONS]->(x)",
                    {"id": cid, "name": e},
                )
                count += 1
            # Co-occur edges among entities
            if len(ents) > 1:
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        s.run(
                            "MERGE (a:Entity {name:$a}) MERGE (b:Entity {name:$b}) MERGE (a)-[:CO_OCCUR]->(b)",
                            {"a": ents[i], "b": ents[j]},
                        )
                        count += 1
    return count


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
