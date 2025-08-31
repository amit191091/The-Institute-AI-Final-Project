import os
from types import SimpleNamespace
from app.chunking import structure_chunks
from app.utils import approx_token_len


def _el(kind: str, text: str, page: int):
    return SimpleNamespace(category=kind, text=text, metadata=SimpleNamespace(page_number=page))


def _demo_elements():
    # 3 paragraphs with varying lengths, a table and a figure caption
    return [
        _el("Text", "Introduction\nThis is a longer paragraph about gear wear mechanisms. " * 8, 1),
        _el("Text", "Methods\nDetailed experimental setup is described here." * 6, 1),
        _el("Text", "Results\nSignificant pitting and scuffing observed in Figure 2." * 6, 1),
        _el("Image", "", 1),
        _el("Text", "Figure 2: Microscopy of worn tooth\nImage file: images/p1.png", 1),
        _el("Table", "| A | B |\n|---|---|\n| 1 | 2 |", 1),
        _el("Text", "Table 1: Demo table caption", 1),
    ]


def run_with_env(env: dict):
    # Apply env temporarily
    old = {}
    try:
        for k, v in env.items():
            old[k] = os.environ.get(k)
            os.environ[k] = str(v)
        chunks = structure_chunks(_demo_elements(), file_path="flags-demo.pdf")
        # Treat any non-table/figure as textual for averaging
        text_chunks = [
            c for c in chunks
            if (c.get("section") or c.get("section_type") or "Text") not in ("Table", "Figure")
        ]
        avg_tokens = 0.0
        if text_chunks:
            toks = [approx_token_len(c.get("content") or "") for c in text_chunks]
            avg_tokens = sum(toks) / max(1, len(toks))
        return {
            "total": len(chunks),
            "text": len(text_chunks),
            "avg_tokens": round(avg_tokens, 1),
        }
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_chunking_flags_matrix():
    base = {
        "RAG_TEXT_TARGET_TOKENS": "450",
        "RAG_TEXT_MAX_TOKENS": "800",
    }
    cases = [
        {**base, "RAG_TEXT_SPLIT_MULTI": "0", "RAG_SEMANTIC_CHUNKING": "0"},
        {**base, "RAG_TEXT_SPLIT_MULTI": "1", "RAG_SEMANTIC_CHUNKING": "0"},
        {**base, "RAG_TEXT_SPLIT_MULTI": "0", "RAG_SEMANTIC_CHUNKING": "1"},
        {**base, "RAG_TEXT_SPLIT_MULTI": "1", "RAG_SEMANTIC_CHUNKING": "1"},
        # Vary overlap and min chunk sizes
        {**base, "RAG_TEXT_SPLIT_MULTI": "1", "RAG_SEMANTIC_CHUNKING": "1", "RAG_TEXT_OVERLAP_SENTENCES": "0"},
        {**base, "RAG_TEXT_SPLIT_MULTI": "1", "RAG_SEMANTIC_CHUNKING": "1", "RAG_TEXT_OVERLAP_SENTENCES": "2"},
        {**base, "RAG_TEXT_SPLIT_MULTI": "1", "RAG_SEMANTIC_CHUNKING": "1", "RAG_MIN_CHUNK_TOKENS": "300"},
        {**base, "RAG_TEXT_SPLIT_MULTI": "1", "RAG_SEMANTIC_CHUNKING": "1", "RAG_MIN_CHUNK_TOKENS": "100"},
    ]

    results = []
    for env in cases:
        res = run_with_env(env)
        res["env"] = env
        results.append(res)

    # Persist a debug artifact for local inspection
    try:
        import json, os
        os.makedirs("logs", exist_ok=True)
        with open("logs/chunk_flags_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    # Sanity: avg tokens should be > 10 in all cases on this synthetic content
    for r in results:
        assert r["avg_tokens"] >= 10.0, f"avg tokens too small with env={r['env']} => {r}"

    # Basic sanity: when MIN_CHUNK_TOKENS is higher, average tokens should not be smaller
    hi = [r for r in results if r["env"].get("RAG_MIN_CHUNK_TOKENS") == "300"]
    lo = [r for r in results if r["env"].get("RAG_MIN_CHUNK_TOKENS") == "100"]
    if hi and lo:
        assert hi[0]["avg_tokens"] >= lo[0]["avg_tokens"]
