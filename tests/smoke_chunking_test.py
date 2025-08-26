from types import SimpleNamespace

from app.chunking import structure_chunks


def _el(kind: str, text: str, page: int):
    return SimpleNamespace(category=kind, text=text, metadata=SimpleNamespace(page_number=page))


def test_basic_anchors_and_ids():
    # Arrange synthetic elements: an image + its caption + a descriptive paragraph
    elements = [
        _el("Image", "", 1),
        _el("Text", "Figure 1: Wear pattern on gear tooth\nImage file: images/p1.png", 1),
        _el("Text", "As shown in Figure 1, the spalling initiates at the root.", 1),
        _el("Table", "| ColA | ColB |\n| --- | --- |\n| 1 | 2 |", 1),
        _el("Text", "Table 1: Simple demo table", 1),
    ]

    chunks = structure_chunks(elements, file_path="demo.pdf")
    assert any(c.get("section") == "Figure" for c in chunks)
    assert any(c.get("section") == "Table" for c in chunks)

    # Figure assertions
    figs = [c for c in chunks if c.get("section") == "Figure"]
    f = figs[0]
    assert f.get("anchor") == "figure-1"
    assert f.get("figure_label", "").lower().startswith("figure 1:"), f.get("figure_label")
    assert f.get("chunk_id") and f.get("doc_id") and f.get("content_hash")
    # Associated text should reference Figure 1 and not be the caption
    assoc = f.get("figure_associated_text_preview")
    assert assoc and "figure 1" in assoc.lower()

    # Table assertions
    tbls = [c for c in chunks if c.get("section") == "Table"]
    t = tbls[0]
    assert t.get("anchor", "").startswith("table-") or t.get("anchor", "").startswith("p1-tbl")
    assert t.get("chunk_id") and t.get("doc_id") and t.get("content_hash")
