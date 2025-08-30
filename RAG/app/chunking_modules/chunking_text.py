from typing import Dict, List, Optional
from RAG.app.logger import get_logger
from RAG.app.utils import simple_summarize, truncate_to_tokens, split_into_sentences, sha1_short
from RAG.app.config import settings


def process_text_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, trace=False):
    """Process a text element and create a text chunk."""
    log = get_logger()
    
    raw_text = (getattr(el, "text", "") or "").strip()
    
    # Sentence-aware chunking to target CHUNK_TOK_AVG_RANGE[0] tokens per chunk
    sentences = split_into_sentences(raw_text)
    if not sentences:
        content = simple_summarize(raw_text, ratio=0.2)
    else:
        # Greedy pack sentences until ~CHUNK_TOK_AVG_RANGE[0] tokens, cap at CHUNK_TOK_AVG_RANGE[1]
        buf: List[str] = []
        cur_tokens = 0
        target = settings.chunking.CHUNK_TOK_AVG_RANGE[0]
        max_tokens = settings.chunking.CHUNK_TOK_AVG_RANGE[1]
        for s in sentences:
            s_tok = len(s.split())  # Simple token approximation
            if cur_tokens + s_tok > max_tokens and buf:
                break
            buf.append(s)
            cur_tokens += s_tok
            if cur_tokens >= target:
                break
        content = " ".join(buf) if buf else raw_text
    
    # Fallback anchor for text content if missing
    if anchor is None:
        try:
            if page is not None:
                page_ord[page] = page_ord.get(page, 0) + 1
                anchor = f"p{int(page)}-t{page_ord[page]}"
            else:
                anchor = f"t{idx}"
        except Exception as e:
            anchor = f"t{idx}"
    
    # Build IDs and metadata hygiene fields
    content = truncate_to_tokens(content, settings.chunking.CHUNK_TOK_AVG_RANGE[1]).strip()
    # Safe preview generation - handle empty content
    lines = (content or "").splitlines()
    chunk_preview = lines[0][:200] if lines else ""
    content_hash = sha1_short(content)
    chunk_id = f"{doc_id}#p{page}:{section_type or 'Text'}/{anchor}"
    order_val = None
    try:
        order_val = page_ord.get(int(page)) if page is not None else None
    except Exception as e:
        order_val = None
    
    return {
        "file_name": file_path,
        "page": page,
        "section_type": section_type or "Text",
        "section": section_type or "Text",
        "anchor": anchor or None,
        "order": order_val,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "content": content,
        "preview": chunk_preview,
        "keywords": [],  # Will be filled by main function
    }
