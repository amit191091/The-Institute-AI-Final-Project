from typing import Dict, List, Optional, Tuple
from RAG.app.logger import get_logger
from RAG.app.utils import simple_summarize, truncate_to_tokens, naive_markdown_table, sha1_short
from RAG.app.config import settings


def process_table_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, trace=False):
    """Process a table element and create a table chunk."""
    log = get_logger()
    
    raw_text = (getattr(el, "text", "") or "").strip()
    extractor = getattr(md, "extractor", None) if md is not None else None
    table_md_path = getattr(md, "table_md_path", None) if md is not None else None
    table_csv_path = getattr(md, "table_csv_path", None) if md is not None else None
    table_number = getattr(md, "table_number", None) if md is not None else None
    table_label = getattr(md, "table_label", None) if md is not None else None
    table_caption = getattr(md, "table_caption", None) if md is not None else None
    
    as_text = raw_text
    # Use the generated summary if available
    table_summary = getattr(md, "table_summary", None) if md is not None else None
    if table_summary:
        distilled = table_summary
    else:
        distilled = simple_summarize(as_text, ratio=0.05)
    
    row_range: Optional[Tuple[int, int]] = (0, 0)
    col_names: Optional[list[str]] = []
    md_content = naive_markdown_table(as_text)
    # If a full label exists, include at the top for clarity
    label_hdr = f"LABEL: {table_label}\n" if (table_label and str(table_label).strip()) else ""
    content = f"[TABLE]\n{label_hdr}SUMMARY:\n{distilled}\nMARKDOWN:\n{md_content or as_text}\nRAW:\n{as_text}"
    tok = len(content.split())  # Simple token approximation
    if tok > settings.chunking.CHUNK_TOK_MAX:
        content = truncate_to_tokens(content, settings.chunking.CHUNK_TOK_MAX)
    if trace:
        log.debug("CHUNK-OUT[%d]: section=Table tokens=%d anchor=%s", idx, tok, anchor)
    
    # Final fallback for table anchor if still None
    if anchor is None:
        try:
            if page is not None:
                page_ord[page] = page_ord.get(page, 0) + 1
                ordinal = page_ord[page]
                # keep a consistent table anchor scheme when number is unknown
                anchor = f"p{int(page)}-tbl{ordinal}"
            elif table_number is not None:
                anchor = f"table-{int(table_number):02d}"
        except Exception as e:
            anchor = f"tbl{idx}"
    
    # Build deterministic IDs and metadata
    # Safe preview generation - handle empty content
    lines = (content or "").splitlines()
    chunk_preview = lines[0][:200] if lines else ""
    content_hash = sha1_short(content)
    chunk_id = f"{doc_id}#p{page}:{section_type or 'Table'}/{anchor}"
    order_val = None
    try:
        order_val = page_ord.get(int(page)) if page is not None else None
    except Exception as e:
        order_val = None
    
    return {
        "file_name": file_path,
        "page": page,
        "section_type": section_type or "Table",
        "section": "Table",
        "anchor": anchor or None,
        "order": order_val,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "extractor": extractor,
        "table_number": table_number,
        "table_label": table_label,
        "table_row_range": row_range,
        "table_col_names": col_names,
        "table_md_path": table_md_path,
        "table_csv_path": table_csv_path,
        "table_caption": table_caption,
        "content": content.strip(),
        "preview": chunk_preview,
        "keywords": [],  # Will be filled by main function
    }
