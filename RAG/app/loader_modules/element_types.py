#!/usr/bin/env python3
"""Element type definitions for document loading pipeline.

This module provides the core Element classes used by the pipeline prior to chunking/indexing.
These classes are extracted from app/loaders.py to maintain compatibility and functionality.

Classes:
    Element: Base class for all document elements
    Table: Convenience wrapper for table elements  
    Figure: Convenience wrapper for figure/image elements
"""

from typing import Dict, Any, Optional


class Element:
    """Minimal carrier used by the pipeline prior to chunking/indexing.

    Attributes:
        text: Raw content. For tables, prefer markdown; for images, a short tag.
        category: One of {"table", "image", "text", "figure"}.
        metadata: Dict with extractor name, page_number, and optional file paths.
    """

    def __init__(self, text: str = "", category: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.category = category
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Element(category='{self.category}', text_length={len(self.text)})"


class Table(Element):
    """Convenience wrapper for table elements."""

    def __init__(self, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(text, "table", metadata)

    def __repr__(self) -> str:
        return f"Table(text_length={len(self.text)}, extractor='{self.metadata.get('extractor', 'unknown')}')"


class Figure(Element):
    """Convenience wrapper for figure/image elements."""

    def __init__(self, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(text, "image", metadata)

    def __repr__(self) -> str:
        return f"Figure(extractor='{self.metadata.get('extractor', 'unknown')}', page={self.metadata.get('page_number', 'unknown')})"


class Text(Element):
    """Convenience wrapper for text elements."""

    def __init__(self, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(text, "text", metadata)

    def __repr__(self) -> str:
        return f"Text(text_length={len(self.text)}, page={self.metadata.get('page_number', 'unknown')})"
