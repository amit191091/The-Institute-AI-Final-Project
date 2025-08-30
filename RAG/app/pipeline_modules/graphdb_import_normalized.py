from __future__ import annotations

"""
Import logs/normalized/graph.json and chunks.jsonl into Neo4j with richer semantics.

Nodes:
  Document(id, title)
  Section(title)
  Figure(id, number, caption)
  Table(id, number, title)
  Stage(name)
  Metric(name)
  Sensor(name)
  Event(date)
  Page(id, file, number:int)
  Chunk(id)

Edges:
  BELONGS_TO, REFERS_TO, HAS_STAGE, HAS_METRIC, HAS_SENSOR, NEXT
  ON_PAGE (Chunk->Page, Figure->Page, Table->Page)

This importer also synthesizes Page and Chunk nodes from chunks.jsonl so Tables/Figures
are linked to their pages and result chunks.
"""

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from RAG.app.logger import trace_func

try:
    from dotenv import load_dotenv  # type: ignore
    from dotenv import dotenv_values, find_dotenv
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
    from neo4j import GraphDatabase, Query  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore
    class Query(str):  # type: ignore
        pass


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
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"nodes": [], "edges": []}


@trace_func
def _read_chunks(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


@trace_func
def import_normalized_graph(graph_path: str | os.PathLike[str], chunks_path: str | os.PathLike[str]) -> int:
    drv = _get_driver()
    if drv is None:
        return 0
    nodes_edges = _read_json(Path(graph_path))
    chunks = _read_chunks(Path(chunks_path))

    # Build helpers for page lookups
    # Map: figure_id -> page, table_id -> page, chunk_id -> page, table/figure id strings match source_id ("Fig:N"/"Tbl:N")
    fig_pages: Dict[str, int] = {}
    tbl_pages: Dict[str, int] = {}
    chunk_pages: Dict[str, int] = {}
    # Gather semantics for tables/figures from chunks
    tbl_sem: Dict[str, Dict[str, Any]] = {}
    fig_cap: Dict[str, str] = {}
    for c in chunks:
        try:
            ctype = str(c.get("type") or "")
            page = int(c.get("page_start") or 0)
        except Exception:
            page = 0
        cid = str(c.get("id") or "")
        if cid:
            chunk_pages[cid] = page
        sid = str(c.get("source_id") or "")
        if ctype == "figure_caption" and sid:
            fig_pages[sid] = page or fig_pages.get(sid, 0)
            try:
                fig_cap[sid] = str(c.get("text") or "")
            except Exception:
                pass
        if ctype == "table_data" and sid:
            tbl_pages[sid] = page or tbl_pages.get(sid, 0)
            # Accumulate semantics
            ts = tbl_sem.setdefault(sid, {})
            for k in ("wear_stage", "measurement_type", "sensor_type", "speed"):
                if c.get(k) is not None:
                    ts[k] = c.get(k)

    # Constraints
    db = os.getenv("NEO4J_DATABASE") or "neo4j"
    with drv.session(database=db) as s:
        for c in (
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT section_title IF NOT EXISTS FOR (x:Section) REQUIRE x.title IS UNIQUE",
            "CREATE CONSTRAINT figure_id IF NOT EXISTS FOR (f:Figure) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT table_id IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT stage_name IF NOT EXISTS FOR (x:Stage) REQUIRE x.name IS UNIQUE",
            "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (x:Metric) REQUIRE x.name IS UNIQUE",
            "CREATE CONSTRAINT sensor_name IF NOT EXISTS FOR (x:Sensor) REQUIRE x.name IS UNIQUE",
            "CREATE CONSTRAINT event_date IF NOT EXISTS FOR (x:Event) REQUIRE x.date IS UNIQUE",
        ):
            try:
                s.run(c)  # type: ignore[arg-type]
            except Exception:
                pass

        # Upsert base nodes from graph.json
        ncount = 0
        for n in nodes_edges.get("nodes", []) or []:
            nid = str(n.get("id"))
            ntype = str(n.get("type") or "")
            props = n.get("props") or {}
            if ntype == "Document":
                s.run("MERGE (d:Document {id:$id}) SET d.title=$title", {"id": nid, "title": props.get("title")})
            elif ntype == "Section":
                s.run("MERGE (x:Section {title:$t})", {"t": props.get("title")})
            elif ntype == "Figure":
                # Expect id like fig:N
                num = None
                m = re.search(r"(\d+)$", nid)
                if m:
                    try:
                        num = int(m.group(1))
                    except Exception:
                        num = None
                s.run(
                    "MERGE (f:Figure {id:$id}) SET f.number=$n, f.caption=$cap",
                    {"id": nid, "n": num, "cap": props.get("caption")},
                )
            elif ntype == "Table":
                num = None
                m = re.search(r"(\d+)$", nid)
                if m:
                    try:
                        num = int(m.group(1))
                    except Exception:
                        num = None
                s.run(
                    "MERGE (t:Table {id:$id}) SET t.number=$n, t.title=$title",
                    {"id": nid, "n": num, "title": props.get("title")},
                )
            elif ntype == "Stage":
                s.run("MERGE (x:Stage {name:$n})", {"n": nid.split(":",1)[1] if ":" in nid else nid})
            elif ntype == "Metric":
                s.run("MERGE (x:Metric {name:$n})", {"n": nid.split(":",1)[1] if ":" in nid else nid})
            elif ntype == "Sensor":
                s.run("MERGE (x:Sensor {name:$n})", {"n": nid.split(":",1)[1] if ":" in nid else nid})
            elif ntype == "Event":
                # id like event:YYYY-MM-DD
                date = nid.split(":", 1)[1] if ":" in nid else props.get("date")
                s.run("MERGE (e:Event {date:$d})", {"d": date})
            ncount += 1

        # Edges from graph.json; create Chunk nodes on-the-fly for chunk:* endpoints
        ecount = 0
        for e in nodes_edges.get("edges", []) or []:
            et = e.get("type")
            src = str(e.get("from"))
            dst = str(e.get("to"))
            def _label_for(node_id: str) -> Tuple[str, Dict[str, Any]]:
                if node_id.startswith("chunk:"):
                    return "Chunk", {"id": node_id.split(":",1)[1]}
                if node_id.startswith("fig:"):
                    return "Figure", {"id": node_id}
                if node_id.startswith("tbl:"):
                    return "Table", {"id": node_id}
                if node_id.startswith("sec:"):
                    return "Section", {"title": node_id.split(":",1)[1].replace("_"," ")}
                if node_id.startswith("doc:"):
                    return "Document", {"id": node_id.split(":",1)[1]}
                if node_id.startswith("stage:"):
                    return "Stage", {"name": node_id.split(":",1)[1]}
                if node_id.startswith("metric:"):
                    return "Metric", {"name": node_id.split(":",1)[1]}
                if node_id.startswith("sensor:"):
                    return "Sensor", {"name": node_id.split(":",1)[1]}
                if node_id.startswith("event:"):
                    return "Event", {"date": node_id.split(":",1)[1]}
                return "Thing", {"id": node_id}
            lsrc, psrc = _label_for(src)
            ldst, pdst = _label_for(dst)
            # MERGE endpoints (some may not exist yet, e.g., Chunk)
            if lsrc == "Chunk":
                s.run("MERGE (a:Chunk {id:$id})", {"id": psrc["id"]})
            if ldst == "Chunk":
                s.run("MERGE (b:Chunk {id:$id})", {"id": pdst["id"]})
            # Build Cypher for relation
            cypher = f"MATCH (a:{lsrc}), (b:{ldst}) WHERE "
            conds = []
            params = {}
            for k, v in psrc.items():
                conds.append(f"a.{k}=$a_{k}")
                params[f"a_{k}"] = v
            for k, v in pdst.items():
                conds.append(f"b.{k}=$b_{k}")
                params[f"b_{k}"] = v
            cypher += " AND ".join(conds) + " MERGE (a)-[:" + str(et) + "]->(b)"
            # Cast to Any to satisfy Pylance when passing dynamic Cypher strings
            s.run(cast(Any, cypher), params)
            ecount += 1

        # Synthesize Page nodes and ON_PAGE edges from chunks
        for c in chunks:
            cid = str(c.get("id") or "")
            page = c.get("page_start")
            file = (c.get("extras") or {}).get("file_title") or c.get("document_id")
            if not cid or page is None or file is None:
                continue
            pid = f"{file}#p{page}"
            s.run("MERGE (p:Page {id:$id}) SET p.file=$file, p.number=toInteger($n)", {"id": pid, "file": file, "n": int(page)})
            s.run("MATCH (c:Chunk {id:$cid}), (p:Page {id:$pid}) MERGE (c)-[:ON_PAGE]->(p)", {"cid": cid, "pid": pid})
            # Link Table/Figure for this chunk when source_id is present
            sid = str(c.get("source_id") or "")
            if sid:
                if sid.lower().startswith("fig"):
                    s.run("MATCH (f:Figure {id:$fid}), (p:Page {id:$pid}) MERGE (f)-[:ON_PAGE]->(p)", {"fid": f"fig:{sid.split(':',1)[1]}", "pid": pid})
                if sid.lower().startswith("tbl"):
                    s.run("MATCH (t:Table {id:$tid}), (p:Page {id:$pid}) MERGE (t)-[:ON_PAGE]->(p)", {"tid": f"tbl:{sid.split(':',1)[1]}", "pid": pid})
                # Attach semantics from table chunks
                if sid.lower().startswith("tbl"):
                    sem = tbl_sem.get(sid, {})
                    if sem.get("wear_stage"):
                        s.run("MATCH (t:Table {id:$tid}) MERGE (st:Stage {name:$name}) MERGE (t)-[:HAS_STAGE]->(st)", {"tid": f"tbl:{sid.split(':',1)[1]}", "name": sem["wear_stage"]})
                    for m in (sem.get("measurement_type") or []) or []:
                        s.run("MATCH (t:Table {id:$tid}) MERGE (x:Metric {name:$n}) MERGE (t)-[:HAS_METRIC]->(x)", {"tid": f"tbl:{sid.split(':',1)[1]}", "n": m})
                    for sn in (sem.get("sensor_type") or []) or []:
                        s.run("MATCH (t:Table {id:$tid}) MERGE (x:Sensor {name:$n}) MERGE (t)-[:HAS_SENSOR]->(x)", {"tid": f"tbl:{sid.split(':',1)[1]}", "n": sn})

    return 1
