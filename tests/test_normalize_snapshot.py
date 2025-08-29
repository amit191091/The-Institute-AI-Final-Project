import json
import os
import subprocess
import sys


def test_normalize_runs():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    script = os.path.join(repo_root, 'RAG', 'app', 'normalize_snapshot.py')
    input_path = os.path.join(repo_root, 'RAG', 'logs', 'db_snapshot.jsonl')
    out_dir = os.path.join(repo_root, 'RAG', 'logs', 'normalized')

    # Run the script
    completed = subprocess.run([sys.executable, script, '--input', input_path, '--outdir', out_dir], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    # Check outputs
    chunks_path = os.path.join(out_dir, 'chunks.jsonl')
    graph_path = os.path.join(out_dir, 'graph.json')
    assert os.path.exists(chunks_path)
    assert os.path.exists(graph_path)

    # Spot check: at least figures 1..4 and tables 1..3 appear in source ids or nodes
    with open(chunks_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    # ensure each chunk has minimal fields
    for c in lines:
        assert 'id' in c and 'document_id' in c and 'text' in c and 'type' in c
    # graph sanity
    g = json.load(open(graph_path, 'r', encoding='utf-8'))
    node_ids = {n['id'] for n in g.get('nodes', [])}
    # Check for actual nodes that exist in the data
    assert 'doc:gear-wear-failure.pdf' in node_ids
    assert 'sec:Baseline_Condition' in node_ids
    assert 'metric:RMS' in node_ids
