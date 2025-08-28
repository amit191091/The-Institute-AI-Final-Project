#!/usr/bin/env python3
"""
Graph Builders
=============

Graph construction and edge management functions.
"""

import io
import json
import os
from typing import Any, Dict, List, Optional, Set


def add_node(nodes: Dict[str, Dict[str, Any]], node_id: str, node_type: str, properties: Dict[str, Any]) -> None:
    """Add a node to the graph."""
    if node_id not in nodes:
        nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties
        }


def add_edge(edges: List[Dict[str, Any]], edge_type: str, source_id: str, target_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
    """Add an edge to the graph."""
    edge = {
        "source": source_id,
        "target": target_id,
        "type": edge_type
    }
    if properties:
        edge["properties"] = properties
    edges.append(edge)


def build_graph_from_chunks(chunk_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build graph from chunk items."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    
    # Add nodes for each chunk
    for chunk in chunk_items:
        chunk_id = chunk["id"]
        chunk_type = chunk["type"]
        
        # Create node properties
        properties = {
            "text": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
            "section": chunk["section"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"]
        }
        
        # Add optional properties
        if chunk.get("date"):
            properties["date"] = chunk["date"]
        if chunk.get("wear_stage"):
            properties["wear_stage"] = chunk["wear_stage"]
        if chunk.get("measurement_type"):
            properties["measurement_type"] = chunk["measurement_type"]
        if chunk.get("sensor_type"):
            properties["sensor_type"] = chunk["sensor_type"]
        
        add_node(nodes, chunk_id, chunk_type, properties)
        
        # Add section relationships
        section_id = f"section:{chunk['section']}"
        add_node(nodes, section_id, "Section", {"name": chunk["section"]})
        add_edge(edges, "BELONGS_TO", chunk_id, section_id)
        
        # Add wear stage relationships
        if chunk.get("wear_stage"):
            stage_id = f"stage:{chunk['wear_stage']}"
            add_node(nodes, stage_id, "WearStage", {"name": chunk["wear_stage"]})
            add_edge(edges, "HAS_STAGE", chunk_id, stage_id)
        
        # Add measurement type relationships
        for mtype in chunk.get("measurement_type", []):
            metric_id = f"metric:{mtype}"
            add_node(nodes, metric_id, "MeasurementType", {"name": mtype})
            add_edge(edges, "HAS_METRIC", chunk_id, metric_id)
        
        # Add sensor type relationships
        for stype in chunk.get("sensor_type", []):
            sensor_id = f"sensor:{stype}"
            add_node(nodes, sensor_id, "SensorType", {"name": stype})
            add_edge(edges, "HAS_SENSOR", chunk_id, sensor_id)
    
    # Add timeline events and NEXT edges
    event_chunks = [c for c in chunk_items if c.get("date")]
    
    def event_key(c: Dict[str, Any]) -> str:
        if c.get("date"):
            return c["date"]
        # midpoint for range
        ds, de = c.get("date_start"), c.get("date_end")
        return (ds or "") + "~" + (de or "")
    
    event_chunks_sorted = sorted(event_chunks, key=event_key)
    
    # Add nodes for Events and NEXT edges
    prev_event_id: Optional[str] = None
    for ec in event_chunks_sorted:
        d = ec.get("date")
        if d:
            eid = f"event:{d}"
            add_node(nodes, eid, "Event", {"date": d})
            # Link event to stage/sensors/metrics as well
            if ec.get("wear_stage"):
                add_edge(edges, "HAS_STAGE", eid, f"stage:{ec['wear_stage']}")
            for m in ec.get("measurement_type", []) or []:
                add_edge(edges, "HAS_METRIC", eid, f"metric:{m}")
            for s in ec.get("sensor_type", []) or []:
                add_edge(edges, "HAS_SENSOR", eid, f"sensor:{s}")
            if prev_event_id:
                add_edge(edges, "NEXT", prev_event_id, eid)
            prev_event_id = eid
        else:
            # For ranges, create two nodes and a NEXT edge between them
            ds, de = ec.get("date_start"), ec.get("date_end")
            if ds:
                eid_s = f"event:{ds}"
                add_node(nodes, eid_s, "Event", {"date": ds})
                if prev_event_id:
                    add_edge(edges, "NEXT", prev_event_id, eid_s)
                prev_event_id = eid_s
            if de:
                eid_e = f"event:{de}"
                add_node(nodes, eid_e, "Event", {"date": de})
                if prev_event_id and prev_event_id != eid_e:
                    add_edge(edges, "NEXT", prev_event_id, eid_e)
                prev_event_id = eid_e
    
    return {"nodes": list(nodes.values()), "edges": edges}


def write_outputs(chunks: List[Dict[str, Any]], graph: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    graph_path = os.path.join(out_dir, "graph.json")
    with io.open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with io.open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
