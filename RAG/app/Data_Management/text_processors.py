#!/usr/bin/env python3
"""
Text Processors
==============

Text cleaning and normalization functions.
"""

import re
from typing import Optional


def norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p.replace("\\", "/")


def is_footer_preview(text: str) -> bool:
    # Matches patterns like "12 | P a g e"
    return bool(re.fullmatch(r"\s*\d+\s*\|\s*P\s*a\s*g\s*e\s*", text))


def is_heading_only(text: str) -> bool:
    # Treat very short single-line headings as noise (e.g., "Case", "Recommendations")
    # Keep if it contains colon or ends with a period (likely sentence).
    if not text:
        return True
    if ":" in text or text.strip().endswith("."):
        return False
    # 1-3 words, title-like
    words = re.findall(r"\w+", text)
    return 0 < len(words) <= 3


def clean_caption(label: str) -> str:
    t = (label or "").strip()
    # Ensure single sentence. If multiple sentences, keep first; else leave.
    parts = re.split(r"(?<=[.!?])\s+", t)
    return parts[0] if parts else t
