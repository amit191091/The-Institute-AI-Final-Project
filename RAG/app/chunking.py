from typing import Dict, List, Optional, Tuple, Any
import os
from RAG.app.logger import get_logger, trace_func
from RAG.app.Data_Management.metadata import classify_section_type, extract_keywords
from RAG.app.chunking_modules.chunking_table import process_table_chunk
from RAG.app.chunking_modules.chunking_figure import process_figure_chunk
from RAG.app.chunking_modules.chunking_text import process_text_chunk
from RAG.app.chunking_modules.chunking_utils import (
    build_caption_map, derive_anchor, derive_doc_id, 
    associate_figures_with_captions, associate_tables_with_captions, add_keywords_to_chunks,
    _md_get
)
from RAG.app.utils import (
    approx_token_len, simple_summarize, truncate_to_tokens, naive_markdown_table, 
    split_into_sentences, split_into_paragraphs, slugify, sha1_short
)
from RAG.app.chunking_modules.advanced_chunking import (
    _get_max_tokens_for_type, _get_target_tokens_for_type, _analyze_table_markdown,
    _create_chunk_dict, _is_textual_chunk, _semantic_groups
)
from RAG.app.chunking_modules.heading_detection import (
    _is_heading_by_style, _looks_like_heading, _heading_level, _detect_heading_in_text,
    _make_section_id, _update_section_context, _current_section_context
)
from RAG.app.chunking_modules.chunking_config import (
    _get_chunking_setting, _get_chunking_int_setting, _get_chunking_float_setting,
    _get_chunking_string_setting, _get_chunking_tuple_setting
)


@trace_func
def structure_chunks(elements, file_path: str) -> List[Dict]:
    """
    Split by natural order: Title/Text, Table, Figure/Image, Appendix.
    Distill each chunk to ~5% core info from its element.
    Save anchors: PageNumber, SectionType, TableId/FigureId (if any), and row/column position for tables.
    Enforce token budgets from config:
    - Text: target ~CHUNK_TOK_AVG_RANGE[0] tokens, max CHUNK_TOK_AVG_RANGE[1]
    - Tables: max CHUNK_TOK_MAX tokens
    - Figures: max CHUNK_TOK_MAX tokens
    """
    chunks: List[Dict] = []
    log = get_logger()

    trace = os.getenv("RAG_TRACE_CHUNKING", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes")
    
    if trace:
        try:
            log.debug("Chunking: %d input elements from %s", len(elements or []), file_path)
        except Exception as e:
            pass

    # Get configuration settings
    try:
        split_multi = _get_chunking_setting("TEXT_SPLIT_MULTI", True)
        SEMANTIC = _get_chunking_setting("SEMANTIC_CHUNKING", True)
        TEXT_TARGET_TOK = _get_chunking_int_setting("TEXT_TARGET_TOKENS", 375)
        TEXT_MAX_TOK = _get_chunking_int_setting("TEXT_MAX_TOKENS", 500)
        FIGURE_TABLE_MAX_TOK = _get_chunking_int_setting("FIGURE_TABLE_MAX_TOK", 800)
        OVERLAP_N = max(0, _get_chunking_int_setting("TEXT_OVERLAP_SENTENCES", 2))
        DISTILL_RATIO = _get_chunking_float_setting("DISTILL_RATIO", 0.05)
        chunking_debug = _get_chunking_setting("CHUNKING_DEBUG", False)
        log_chunk_stats = _get_chunking_setting("LOG_CHUNK_STATS", True)
    except Exception as e:
        # Fallback to safe defaults
        split_multi = True
        TEXT_TARGET_TOK, TEXT_MAX_TOK = 375, 500
        FIGURE_TABLE_MAX_TOK = 800
        SEMANTIC, OVERLAP_N = True, 2
        DISTILL_RATIO = 0.05
        chunking_debug = False
        log_chunk_stats = True

    # Trace the effective flags for visibility
    if trace or chunking_debug:
        try:
            log.info(
                "FLAGS[chunking]: MULTI=%s SEMANTIC=%s TEXT_TARGET=%d TEXT_MAX=%d FIG_TBL_MAX=%d OVERLAP=%d DISTILL=%.2f DEBUG=%s STATS=%s",
                split_multi, SEMANTIC, TEXT_TARGET_TOK, TEXT_MAX_TOK, FIGURE_TABLE_MAX_TOK,
                OVERLAP_N, DISTILL_RATIO, chunking_debug, log_chunk_stats,
            )
        except Exception as e:
            pass
    
    # Derive doc_id once per file
    doc_id = derive_doc_id(file_path)

    # Track per-page ordinals for deterministic anchors
    page_ord: Dict[int, int] = {}

    # Section hierarchy state for advanced heading detection
    section_stack: List[Dict[str, Any]] = []
    section_counter = 0

    # Pre-scan: collect caption lines per page
    caption_map = build_caption_map(elements)
    caption_used: Dict[int, set[int]] = {}

    # Doc-level sequential order for figures
    figure_seq_counter: int = 0
    last_figure_number_assigned: int = 0

    # Track indices consumed by cross-element coalescing
    consumed: set[int] = set()


    for idx, el in enumerate(elements, start=1):
        if idx in consumed:
            continue
            
        kind = getattr(el, "category", getattr(el, "type", "Text")) or "Text"
        md = getattr(el, "metadata", None)
        page = getattr(md, "page_number", None) if md is not None else None

        
        # Extract metadata fields
        table_number = getattr(md, "table_number", None) if md is not None else None
        table_md_path = getattr(md, "table_md_path", None) if md is not None else None
        table_csv_path = getattr(md, "table_csv_path", None) if md is not None else None
        table_label = getattr(md, "table_label", None) if md is not None else None
        table_caption = getattr(md, "table_caption", None) if md is not None else None
        extractor = getattr(md, "extractor", None) if md is not None else None
        
        # Derive anchor
        anchor = derive_anchor(md, table_number, page, table_md_path, table_csv_path, idx)
        raw_text = (getattr(el, "text", "") or "").strip()
        
        # Coalesce adjacent textual elements if enabled
        if str(kind).lower() not in ("table", "figure", "image") and split_multi:
            raw_text = _coalesce_text_elements(elements, idx, raw_text, page, TEXT_TARGET_TOK, TEXT_MAX_TOK, consumed)
        
        section_type = classify_section_type(str(kind), raw_text)
        
        # Advanced heading detection and section hierarchy
        if str(kind).lower() == "text":
            h = _detect_heading_in_text(raw_text, md)
            if h:
                title, level = h
                ctx, section_counter = _update_section_context(title, level, section_stack, section_counter, doc_id)
                if trace:
                    try:
                        log.debug("HEADING[%d]: level=%d title='%s'", idx, level, title)
                    except Exception as e:
                        pass
        
        if trace:
            try:
                log.debug("CHUNK-IN[%d]: kind=%s page=%s section=%s len=%d", idx, kind, page, section_type, len(raw_text))
            except Exception as e:
                pass

        # Process different element types using specialized modules
        if str(kind).lower() == "table":

            chunk = process_table_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, trace)

            chunks.append(chunk)
            continue

        elif str(kind).lower() in ("figure", "image"):

            chunk, figure_seq_counter, last_figure_number_assigned = process_figure_chunk(
                el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, 
                caption_map, caption_used, figure_seq_counter, last_figure_number_assigned, trace
            )

            chunks.append(chunk)
            continue

        # Default: narrative text
        try:
            h = _detect_heading_in_text(raw_text, md)
            if h:
                h_title, h_level = h
                ctx, section_counter = _update_section_context(h_title, h_level, section_stack, section_counter, doc_id)
        except Exception as e:
            pass

        # Process text using the text chunking module
        
        chunk = process_text_chunk(el, md, page, section_type, anchor, doc_id, page_ord, idx, file_path, trace)

        chunks.append(chunk)


    # Post-process chunks
    chunks = _post_process_chunks(chunks, log, trace)


    # Final summary log
    try:
        sec_counts = {"Text": 0, "Table": 0, "Figure": 0}
        from statistics import fmean
        toks = []
        for ch in chunks:
            sec = (ch.get("section") or ch.get("section_type") or "Text")
            if sec not in sec_counts:
                sec_counts[sec] = 0
            sec_counts[sec] += 1
            toks.append(approx_token_len(ch.get("content") or ""))
        avg_tok = float(fmean(toks)) if toks else 0.0
        log.info(
            "Chunking summary: total=%d text=%d table=%d figure=%d other=%d avg_tokens≈%.1f",
            len(chunks), sec_counts.get("Text", 0), sec_counts.get("Table", 0), 
            sec_counts.get("Figure", 0), max(0, len(chunks) - (sec_counts.get("Text",0)+sec_counts.get("Table",0)+sec_counts.get("Figure",0))), avg_tok,
        )
    except Exception as e:
        pass

    return chunks


def _coalesce_text_elements(elements, idx, raw_text, page, TEXT_TARGET_TOK, TEXT_MAX_TOK, consumed):
    """Coalesce adjacent textual elements to hit target token count."""
    try:
        block = [raw_text] if raw_text else []
        cur_tok = approx_token_len(raw_text)
        j = idx + 1
        cap_tok = int(TEXT_MAX_TOK * 1.5)
        
        while j <= len(elements) and cur_tok < max(TEXT_TARGET_TOK, 1):
            _nel = elements[j - 1]
            _nkind = getattr(_nel, "category", getattr(_nel, "type", "Text")) or "Text"
            _nmd = getattr(_nel, "metadata", None)
            _npage = _md_get(_nmd, "page_number") if _nmd is not None else None
            
            if (str(_nkind).lower() in ("table", "figure", "image")) or _npage != page:
                break
                
            _ntext = (getattr(_nel, "text", "") or "").strip()
            _hn = _detect_heading_in_text(_ntext, _nmd)
            if _hn:
                break
                
            if _ntext:
                block.append(_ntext)
                cur_tok = approx_token_len("\n\n".join(block))
                consumed.add(j)
                if cur_tok >= cap_tok:
                    break
            j += 1
            
        # If still too small, greedily pack following text elements
        min_tok = int(os.getenv("RAG_MIN_CHUNK_TOKENS", str(max(250, TEXT_TARGET_TOK // 2))) or max(250, TEXT_TARGET_TOK // 2))
        if cur_tok < min_tok:
            j2 = j
            while j2 <= len(elements) and cur_tok < min(min_tok, TEXT_MAX_TOK):
                _nel = elements[j2 - 1]
                _nkind = getattr(_nel, "category", getattr(_nel, "type", "Text")) or "Text"
                _nmd = getattr(_nel, "metadata", None)
                _npage = _md_get(_nmd, "page_number") if _nmd is not None else None
                
                if (str(_nkind).lower() in ("table", "figure", "image")) or _npage != page:
                    break
                    
                _ntext = (getattr(_nel, "text", "") or "").strip()
                if _ntext:
                    block.append(_ntext)
                    cur_tok = approx_token_len("\n\n".join(block))
                    consumed.add(j2)
                    if cur_tok >= TEXT_MAX_TOK:
                        break
                j2 += 1
                
        if block:
            return "\n\n".join(block)
    except Exception as e:
        pass
    
    return raw_text


def _post_process_chunks(chunks, log, trace):
    """Post-process chunks to associate figures/tables with captions and merge small chunks."""
    try:

        # Associate figures with captions
        chunks = associate_figures_with_captions(chunks)

        
        # Associate tables with captions  
        chunks = associate_tables_with_captions(chunks)

        
        # Merge adjacent small textual chunks
        chunks = _merge_small_chunks(chunks)

        
    except Exception as e:
        if trace:
            try:
                log.debug(f"Post-processing failed: {e}")
            except Exception as e2:
                pass
        # Return original chunks if post-processing fails
        return chunks
    
    return chunks


def _merge_small_chunks(chunks):
    """Merge adjacent small textual chunks to reach minimum token size."""
    try:
        min_tok = int(os.getenv("RAG_MIN_CHUNK_TOKENS", "250") or 250)
        merged: List[Dict[str, Any]] = []
        
        for ch in chunks:
            if not merged:
                merged.append(ch)
                continue
                
            prev = merged[-1]
            if (_is_textual_chunk(prev) and _is_textual_chunk(ch) and 
                (prev.get("file_name") == ch.get("file_name")) and 
                (prev.get("page") == ch.get("page"))):
                
                pt = approx_token_len(prev.get("content") or "")
                ct = approx_token_len(ch.get("content") or "")
                
                if pt < min_tok or ct < min_tok:
                    combined = ((prev.get("content") or "").rstrip() + "\n\n" + (ch.get("content") or "")).strip()
                    max_tokens = _get_max_tokens_for_type(prev.get("section", "Text"))
                    
                    if approx_token_len(combined) <= max_tokens:
                        prev["content"] = combined
                        prev["preview"] = (combined.splitlines()[0] if combined else "")[:200]
                        prev["keywords"] = extract_keywords(combined)
                        continue
                        
            merged.append(ch)
            
        return merged
    except Exception as e:
        return chunks

