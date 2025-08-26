from __future__ import annotations

from typing import List, Optional, TypedDict


class ChunkDict(TypedDict, total=False):
    file_name: str
    page: Optional[int]
    section_type: str
    section: str
    anchor: Optional[str]
    order: Optional[int]
    doc_id: Optional[str]
    chunk_id: Optional[str]
    content_hash: Optional[str]
    extractor: Optional[str]
    content: str
    preview: str
    keywords: List[str]

    # Table-specific
    table_number: Optional[int]
    table_label: Optional[str]
    table_row_range: Optional[List[int]]
    table_col_names: Optional[List[str]]
    table_md_path: Optional[str]
    table_csv_path: Optional[str]
    table_caption: Optional[str]

    # Figure-specific
    image_path: Optional[str]
    figure_number: Optional[int]
    figure_order: Optional[int]
    figure_label: Optional[str]
    figure_caption_original: Optional[str]
    figure_number_source: Optional[str]
    caption_alignment: Optional[str]
    figure_associated_text_preview: Optional[str]
    figure_associated_anchor: Optional[str]


class MetadataDict(TypedDict, total=False):
    file_name: str
    page: Optional[int]
    section: str
    anchor: Optional[str]
    doc_id: Optional[str]
    chunk_id: Optional[str]
    content_hash: Optional[str]
    image_path: Optional[str]
    extractor: Optional[str]

    table_number: Optional[int]
    table_label: Optional[str]
    table_md_path: Optional[str]
    table_csv_path: Optional[str]
    table_row_range: Optional[List[int]]
    table_col_names: Optional[List[str]]

    figure_number: Optional[int]
    figure_order: Optional[int]
    figure_label: Optional[str]
    figure_caption_original: Optional[str]
    figure_number_source: Optional[str]
    caption_alignment: Optional[str]
    figure_associated_text_preview: Optional[str]
    figure_associated_anchor: Optional[str]

    client_id: Optional[str]
    case_id: Optional[str]
    keywords: List[str]
    critical_entities: List[str]
    chunk_summary: str
    incident_type: Optional[str]
    incident_date: Optional[str]
    amount_range: Optional[str]
    month_tokens: List[str]
    day_tokens: List[str]
