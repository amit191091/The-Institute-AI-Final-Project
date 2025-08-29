from typing import Dict, List, Optional, Tuple
import re
from RAG.app.logger import get_logger
from RAG.app.utils import simple_summarize, truncate_to_tokens, sha1_short
from RAG.app.config import settings


def process_figure_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, 
                        caption_map, caption_used, figure_seq_counter, last_figure_number_assigned, trace=False):
    """Process a figure/image element and create a figure chunk."""
    log = get_logger()
    
    # Increment document-level figure order
    try:
        figure_seq_counter += 1
    except Exception:
        figure_seq_counter = figure_seq_counter + 1
    
    raw_text = (getattr(el, "text", "") or "").strip()
    extractor = getattr(md, "extractor", None) if md is not None else None
    
    caption = raw_text or "Figure"
    # Extract figure number if present in caption (e.g., "Figure 2:")
    fig_num = None
    try:
        mfn = re.search(r"\b(fig(?:\.|ure)?)\s*(\d{1,3})\b", caption, re.I)
        if mfn:
            fig_num = int(mfn.group(2))
    except Exception:
        fig_num = None
    
    # Align figure to the nearest caption BELOW it on the same page (by element index), if available
    aligned_via = None
    try:
        if page is not None and caption_map.get(page):
            used = caption_used.setdefault(page, set())
            # choose the first caption with element index greater than current figure element index
            cands = [(cidx, cnum, ctext) for (cidx, cnum, ctext) in caption_map[page] if cidx > idx and cidx not in used]
            if cands:
                cidx, cnum, ctext = sorted(cands, key=lambda t: t[0])[0]
                caption = ctext or caption
                if cnum is not None:
                    fig_num = cnum
                used.add(cidx)
                aligned_via = "after_same_page"
            else:
                # fallback: pick the next unused caption regardless of relative order (rare PDFs)
                cands2 = [(cidx, cnum, ctext) for (cidx, cnum, ctext) in caption_map[page] if cidx not in used]
                if cands2:
                    cidx, cnum, ctext = sorted(cands2, key=lambda t: t[0])[0]
                    caption = ctext or caption
                    if cnum is not None:
                        fig_num = cnum
                    used.add(cidx)
                    aligned_via = "same_page_any"
    except Exception:
        aligned_via = None
    
    # Build a clean summary derived from caption (or metadata), but drop any leading "Figure X:" label
    figure_summary = getattr(md, "figure_summary", None) if md is not None else None
    try:
        if figure_summary:
            _tmp = str(figure_summary).replace("[FIGURE]", "")
            _tmp = re.sub(r"^\s*figure\s*\d{1,3}\s*:\s*", "", _tmp, flags=re.I)
            # Remove any inline image-file hints
            _tmp = re.sub(r"(?mi)^\s*image\s*file\s*:\s*.*$", "", _tmp)
            figure_summary = _tmp.strip() or None
    except Exception:
        pass
    
    if not figure_summary:
        # Provide a brief distilled summary from the cleaned caption text (after removing any image-file hints)
        try:
            cap_for_sum = re.sub(r"(?mi)^\s*image\s*file\s*:\s*.*$", "", caption)
            figure_summary = simple_summarize(cap_for_sum, ratio=0.5)
        except Exception:
            figure_summary = simple_summarize(caption, ratio=0.5)
    
    # Prepare a clean, normalized caption reflecting the final number; keep original for traceability
    caption_original = caption
    try:
        # Strip parser tags like "[FIGURE]" and any leading "Figure X:" label; also drop any "Image file:" lines
        cap_clean = caption_original.replace("[FIGURE]", "")
        cap_clean = re.sub(r"^\s*fig(?:\.|ure)?\s*\d{1,3}\s*:\s*", "", cap_clean, flags=re.I)
        # Remove any lines that declare an image file path
        cap_clean = re.sub(r"(?mi)^\s*image\s*file\s*:\s*.*$", "", cap_clean)
        # Collapse multiple blank lines and trim
        cap_clean = "\n".join([ln for ln in (cap_clean.splitlines()) if ln.strip()]).strip()
    except Exception:
        cap_clean = caption_original
    
    # We'll compute the final figure number below; temporarily store cleaned caption
    caption = cap_clean
    
    # Try to detect image path from text or metadata
    img_path = getattr(md, "image_path", None) if md is not None else None
    if not img_path:
        try:
            m = re.search(r"Image file: (.+)$", caption)
            if m:
                img_path = m.group(1).strip()
        except Exception:
            pass
    
    # Finalize figure number: prefer caption number; if missing, infer from sequence.
    # Ensure monotonically increasing across the document.
    if fig_num is not None and fig_num > last_figure_number_assigned:
        figure_number_final = fig_num
    else:
        figure_number_final = max(last_figure_number_assigned + 1, figure_seq_counter)
    last_figure_number_assigned = figure_number_final
    
    # Normalize the visible caption to the final number with better handling of long descriptions
    try:
        if re.search(r"^\s*fig(?:\.|ure)?\s*\d{1,3}\s*:\s*", caption, re.I):
            caption_norm = re.sub(r"^\s*fig(?:\.|ure)?\s*\d{1,3}\s*:\s*", f"Figure {int(figure_number_final)}: ", caption, flags=re.I)
        else:
            caption_norm = f"Figure {int(figure_number_final)}: {caption}"
        
        # Use full caption for the label (no truncation for metadata)
        figure_label_full = caption_norm
        
        # Create a shorter summary only for the content section shown to LLM 
        # (to manage token budget), but preserve full label in metadata
        try:
            if len(caption_norm.split()) > 25:  # Only summarize if very long
                cap_for_summary = simple_summarize(caption, ratio=0.6)  # Keep more content
                figure_summary_short = f"Figure {int(figure_number_final)}: {cap_for_summary}"
            else:
                figure_summary_short = caption_norm  # Use full for shorter captions
        except Exception:
            figure_summary_short = caption_norm
    except Exception:
        caption_norm = caption
        figure_label_full = caption_norm
        figure_summary_short = caption_norm

    # Derive clean anchor: "figure-N" as requested
    try:
        custom_anchor = f"figure-{int(figure_number_final)}"
    except Exception:
        # Fallback to simple numbering
        custom_anchor = f"figure-{figure_number_final}"

    # Build content with normalized caption shown to the LLM; compute token budget after composing
    content = f"[FIGURE]\nCAPTION:\n{figure_summary_short}\nSUMMARY:\n{figure_summary}"
    tok = len(content.split())  # Simple token approximation
    if tok > settings.CHUNK_TOK_MAX:
        content = truncate_to_tokens(content, settings.CHUNK_TOK_MAX)
    if trace:
        log.debug("CHUNK-OUT[%d]: section=Figure tokens=%d img=%s anchor=%s", idx, tok, img_path, anchor)
    
    # Safe preview generation - handle empty content
    lines = (content or "").splitlines()
    chunk_preview = lines[0][:200] if lines else ""
    content_hash = sha1_short(content)
    # Ensure chunk_id uses the finalized custom anchor for figures
    chunk_id = f"{doc_id}#p{page}:{section_type or 'Figure'}/{custom_anchor}"
    order_val = None
    try:
        order_val = page_ord.get(int(page)) if page is not None else None
    except Exception:
        order_val = None
    
    # Store finalized figure chunk
    return {
        "file_name": file_path,
        "page": page,
        "section_type": section_type or "Figure",
        "section": "Figure",
        "anchor": custom_anchor,  # Use our custom figure anchor
        "order": order_val,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "extractor": extractor,
        "image_path": img_path,
        "figure_number": figure_number_final,
        "figure_order": figure_seq_counter,
        "figure_label": figure_label_full,  # Use full label without truncation
        "figure_caption_original": caption_original,
        "figure_number_source": "caption" if fig_num is not None else "inferred",
        "caption_alignment": aligned_via or "none",
        "content": content.strip(),
        "preview": chunk_preview,  # Keep preview short for overview displays
        "keywords": [],  # Will be filled by main function
    }, figure_seq_counter, last_figure_number_assigned
