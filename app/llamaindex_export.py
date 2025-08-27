from __future__ import annotations

"""Optional: parse PDFs with LlamaIndex and export elements under data/elements/llamaindex.
Writes:
- data/elements/llamaindex/[doc-stem]/tables/*.md and *.csv when possible
- data/elements/llamaindex/[doc-stem]/images/*.png
- data/elements/llamaindex/[doc-stem]/chunks.jsonl (text chunks with anchors)
Graceful no-op if llama_index is unavailable.
"""

from pathlib import Path
import json
import os
import shutil

def _safe_imports():
    try:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.readers.file import PDFReader
        return SimpleDirectoryReader, PDFReader
    except Exception:
        return None, None


def export_llamaindex_for(paths: list[Path], out_root: Path = Path("data")/"elements"/"llamaindex") -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    SDR, PDFReader = _safe_imports()
    if SDR is None or PDFReader is None:
        # Missing dependency; skip silently
        return 0
    count = 0
    for p in paths:
        try:
            if p.suffix.lower() != ".pdf":
                continue
            stem_norm = p.stem.replace(" ", "-").lower()
            stem_dir = out_root / stem_norm
            (stem_dir/"tables").mkdir(parents=True, exist_ok=True)
            (stem_dir/"images").mkdir(parents=True, exist_ok=True)
            chunks_path = stem_dir / "chunks.jsonl"
            # Load with PDFReader to preserve element-level structure where possible
            reader = PDFReader()
            docs = reader.load_data(file=Path(p))  # returns a list of Document
            with open(chunks_path, "w", encoding="utf-8") as f:
                for i, d in enumerate(docs):
                    meta = getattr(d, "metadata", {}) or {}
                    text = getattr(d, "text", "") or ""
                    rec = {
                        "file": str(p.name),
                        "page": meta.get("page_label") or meta.get("page") or None,
                        "type": meta.get("type") or meta.get("category") or "Text",
                        "metadata": meta,
                        "content": text,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Mirror pipeline tables/images into llamaindex folder for convenience
            try:
                img_dir = Path("data")/"images"
                if img_dir.exists():
                    for img in img_dir.glob("*.png"):
                        if img.name.lower().startswith(stem_norm):
                            shutil.copy2(img, stem_dir/"images"/img.name)
            except Exception:
                pass
            try:
                el_dir = Path("data")/"elements"
                if el_dir.exists():
                    for ext in ("*.md", "*.csv"):
                        for tb in el_dir.glob(ext):
                            name_l = tb.name.lower()
                            if name_l.startswith(stem_norm) and ("-table-" in name_l):
                                shutil.copy2(tb, stem_dir/"tables"/tb.name)
            except Exception:
                pass
            # Note: Native LlamaIndex table/image export varies; we provide mirrored artifacts + text chunks.
            count += 1
        except Exception:
            continue
    return count
