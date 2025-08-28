#!/usr/bin/env python3
"""
Text Processors
==============

Text extraction and processing utilities for evaluation.
"""

import re
from typing import List
from difflib import SequenceMatcher


def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text for numerical comparison."""
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(n) for n in numbers]


def extract_dates(text: str) -> List[str]:
    """Extract dates from text for date comparison."""
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
        r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text.lower()))
    return dates


def extract_technical_terms(text: str) -> set:
    """Extract technical terms, units, and specific values."""
    technical_patterns = [
        r'\b\d+\.?\d*\s*(mm|μm|rps|rpm|db|khz|mv/g|w\d+)\b',  # Measurements with units
        r'\b(figure|table)\s*\d+\b',  # Figure/table references
        r'\b(dytran|3053b|mg-5025a|ins haifa)\b',  # Equipment names
        r'\b(healthy|faulty|wear|failure|analysis)\b',  # Technical status terms
    ]
    terms = set()
    for pattern in technical_patterns:
        matches = re.findall(pattern, text.lower())
        terms.update(matches)
    return terms


def extract_citations(text: str) -> set:
    """Extract citations from text."""
    citations = re.findall(r'\[([^\]]+)\]', text)
    return set(citations)


def simple_similarity(a: str, b: str) -> float:
    """Calculate simple text similarity."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
