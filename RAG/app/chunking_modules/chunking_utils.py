from typing import Dict, List, Tuple
import re
import os
from RAG.app.logger import get_logger
from RAG.app.Data_Management.metadata import classify_section_type, extract_keywords
from RAG.app.utils import slugify


def build_caption_map(elements):
    """Pre-scan: collect caption lines per page like 'Figure N: ...' or 'Fig. N: ...' to align with image elements."""
    from typing import DefaultDict
    caption_map: DefaultDict[int, List[Tuple[int, int, str]]] = DefaultDict(list)  # page -> list of (idx, num, text)
    
    for _idx, _el in enumerate(elements, start=1):
        _kind = getattr(_el, "category", getattr(_el, "type", "Text")) or "Text"
        _md = getattr(_el, "metadata", None)
        _page = getattr(_md, "page_number", None) if _md is not None else None
        if _page is None:
            continue
        if str(_kind).lower() == "text":
            _cap = (getattr(_el, "text", "") or "").strip()
            if _cap:
                # Support Figure / Fig. / multi-line text after colon; capture full line for later normalization
                m = re.search(r"\b(fig(?:\.|ure)?)\s*(\d{1,3})\b\s*:\s*(.+)$", _cap, re.I)
                if m:
                    try:
                        num = int(m.group(2))
                        caption_map[_page].append((_idx, num, _cap))
                    except Exception:
                        pass
    
    return caption_map


def derive_anchor(el, md, page, table_number, table_md_path, table_csv_path):
    """Derive a robust, non-null anchor for an element."""
    # Priority: explicit table/figure anchors -> element id -> file-based stems -> fallback
    table_anchor = getattr(md, "table_anchor", None) if md is not None else None
    figure_anchor = getattr(md, "figure_anchor", None) if md is not None else None
    anchor = table_anchor or figure_anchor
    
    if anchor is None:
        anchor = getattr(md, "id", None) if md is not None else None
    
    # If still no anchor, derive from known file paths or numbering
    if anchor is None:
        try:
            if table_number is not None and page is not None:
                anchor = f"table-{int(table_number):02d}"
        except Exception:
            pass
    
    if anchor is None and table_md_path:
        try:
            anchor = os.path.splitext(os.path.basename(str(table_md_path)))[0]
        except Exception:
            pass
    
    if anchor is None and table_csv_path:
        try:
            anchor = os.path.splitext(os.path.basename(str(table_csv_path)))[0]
        except Exception:
            pass
    
    return anchor


def derive_doc_id(file_path):
    """Derive doc_id once per file."""
    try:
        doc_id = slugify(str(os.path.basename(file_path)))
    except Exception:
        doc_id = slugify(str(file_path))
    return doc_id


def associate_figures_with_captions(chunks):
    """Post-process: associate each figure with its caption Text chunk on the same page."""
    try:
        for i, ch in enumerate(chunks):
            if (ch.get("section") or ch.get("section_type")) != "Figure":
                continue
            pg = ch.get("page")
            fig_num = ch.get("figure_number")
            assoc_text = None
            assoc_anchor = None

            # Pass 1: prefer the caption chunk (e.g., a Text chunk starting with "Figure N:" or "Fig. N:")
            if fig_num is not None:
                cap_pat = rf"^\s*fig(?:\.|ure)?\s*{int(fig_num)}\b\s*[:\.-]"
                cap_candidate = None
                for nx in chunks:
                    if (nx.get("page") != pg) or (nx.get("section") != "Text"):
                        continue
                    text_content = (nx.get("content") or "")
                    text_preview = (nx.get("preview") or "")
                    lines = []
                    try:
                        lines = (text_content or "").splitlines()
                    except Exception:
                        lines = []
                    if any(re.search(cap_pat, ln, re.I) for ln in (lines or [])) or re.search(cap_pat, text_preview, re.I):
                        cap_candidate = nx
                        break
                if cap_candidate is not None:
                    assoc_text = cap_candidate.get("preview") or (cap_candidate.get("content") or "")[:200]
                    assoc_anchor = cap_candidate.get("anchor")
                    # If the current stored figure_label seems shorter than the caption chunk, upgrade it
                    try:
                        curr_label = ch.get("figure_label") or ""
                        if assoc_text and (len(assoc_text) > len(curr_label)) and assoc_text.lower().startswith("figure "):
                            ch["figure_label"] = assoc_text
                    except Exception:
                        pass

            # Pass 2: if no caption chunk found, fallback to earliest descriptive mention on the same page
            if assoc_text is None and fig_num is not None:
                mention_pat = rf"\bfig(?:\.|ure)?\s*{int(fig_num)}\b"
                caption_pat = rf"^\s*fig(?:\.|ure)?\s*{int(fig_num)}\b\s*[:\.-]"
                candidates = []
                for nx in chunks:
                    if (nx.get("page") != pg) or (nx.get("section") != "Text"):
                        continue
                    text_content = ((nx.get("content") or "") + " " + (nx.get("preview") or "")).strip()
                    if not text_content:
                        continue
                    if re.search(caption_pat, text_content, re.I):
                        continue
                    if re.search(mention_pat, text_content, re.I):
                        candidates.append(nx)
                if candidates:
                    def _order_key(x: dict) -> int:
                        # Prefer explicit 'order' (page ordinal), fallback to numeric suffix from anchor 'p{pg}-t{n}', else large
                        try:
                            if isinstance(x.get("order"), int):
                                return int(x.get("order") or 10**9)
                            anch = str(x.get("anchor") or "")
                            m = re.search(r"p(\d+)-t(\d+)", anch)
                            if m:
                                return int(m.group(2))
                        except Exception:
                            pass
                        return 10**9
                    cand = sorted(candidates, key=_order_key)[:1]
                    if cand:
                        cand = cand[0]
                        assoc_text = cand.get("preview") or (cand.get("content") or "")[:200]
                        assoc_anchor = cand.get("anchor")
            
            # Set the association (but don't override the figure_label)
            if assoc_text:
                ch["figure_associated_text_preview"] = assoc_text
                ch["figure_associated_anchor"] = assoc_anchor
                # If figure_summary_short equals the label (or is too generic), upgrade preview to associated text
                try:
                    prev = ch.get("preview") or ""
                    label = ch.get("figure_label") or ""
                    if not prev or prev.strip().lower() == (label or "").strip().lower():
                        ch["preview"] = assoc_text
                except Exception:
                    pass
                # Keep the original figure_label from the image caption as-is
                # Don't override it with the associated text
    except Exception:
        pass


def associate_tables_with_captions(chunks):
    """Post-process: associate each table with its caption text chunk on the same page."""
    try:
        for ch in chunks:
            if (ch.get("section") or ch.get("section_type")) != "Table":
                continue
            pg = ch.get("page")
            tn = ch.get("table_number")
            assoc_text = None
            assoc_anchor = None
            if tn is None:
                continue
            # Allow variations like "Table N:", "Table N.", "Table N -"
            pattern = rf"^\s*table\s*{int(tn)}\b\s*[:\.-]"
            for nx in chunks:
                if (nx.get("page") != pg) or (nx.get("section") != "Text"):
                    continue
                text_content = (nx.get("content") or "") + " " + (nx.get("preview") or "")
                if re.search(pattern, text_content, re.I):
                    assoc_text = nx.get("preview") or (nx.get("content") or "")[:200]
                    assoc_anchor = nx.get("anchor")
                    break
            if assoc_text:
                ch["table_associated_text_preview"] = assoc_text
                ch["table_associated_anchor"] = assoc_anchor
                # table_label already carries full caption text from loaders
    except Exception:
        pass


def add_keywords_to_chunks(chunks):
    """Add keywords to all chunks."""
    for chunk in chunks:
        content = chunk.get("content", "")
        chunk["keywords"] = extract_keywords(content)
