#!/usr/bin/env python3
"""
Measurement Extractors
=====================

Technical data extraction functions.
"""

import re
from typing import List, Tuple

from RAG.app.logger import trace_func


@trace_func
def normalize_section_title(section: str, text: str) -> str:
    # Canonical list:
    # Executive Summary|System Description|Baseline Condition|Failure Progression|Investigation Findings|Results and Analysis|Recommendations|Conclusion
    s = (section or "").strip().lower()
    text_l = (text or "").strip().lower()
    if s == "summary":
        return "Executive Summary"
    if "recommendations" in text_l:
        return "Recommendations"
    if "conclusion" in text_l or s == "conclusion":
        return "Conclusion"
    if "baseline" in text_l:
        return "Baseline Condition"
    if "investigation findings" in text_l:
        return "Investigation Findings"
    # System Description keywords commonly seen in this doc
    sys_kw = (
        "system description",
        "vessel and transmission",
        "lubrication:",
        "operating conditions",
        "instrumentation & acquisition",
    )
    if any(k in text_l for k in sys_kw):
        return "System Description"
    if "failure progression" in text_l or s == "timeline":
        return "Failure Progression"
    if s in ("figure", "table", "analysis"):
        return "Results and Analysis"
    # Heuristic: default most narrative text to Results and Analysis
    return "Results and Analysis"


@trace_func
def detect_measurements_sensors_speed(text: str) -> Tuple[List[str], List[str], List[int]]:
    tl = text.lower()
    measurements: List[str] = []
    sensors: List[str] = []
    speeds: List[int] = []
    if re.search(r"\brms\b", tl):
        measurements.append("RMS")
    if re.search(r"\bfme\b|modulation energy", tl):
        measurements.append("FME")
    if re.search(r"\bcrest\s*factor\b", tl):
        measurements.append("crest_factor")
    if re.search(r"vibration|fft|spectra|spectrogram|accelerometer", tl):
        sensors.append("accelerometer")
    if re.search(r"photo|microscope|image|figure", tl):
        sensors.append("camera")
    for sp in re.findall(r"(\d{2})\s*\[?rps\]?", tl):
        try:
            v = int(sp)
            if v in (15, 45) and v not in speeds:
                speeds.append(v)
        except Exception:
            pass
    speeds.sort()
    # Deduplicate
    measurements = sorted({m for m in measurements})
    sensors = sorted({s for s in sensors})
    return measurements, sensors, speeds


@trace_func
def clean_caption(label: str) -> str:
    t = (label or "").strip()
    # Ensure single sentence. If multiple sentences, keep first; else leave.
    parts = re.split(r"(?<=[.!?])\s+", t)
    return parts[0] if parts else t


@trace_func
def minimal_table_summary(table_label: str) -> str:
    label = (table_label or "").strip()
    # Convert "Table N: X" -> "Table N summarizes X" as a safe default.
    m = re.match(r"Table\s+(\d+)\s*:\s*(.+)", label, flags=re.IGNORECASE)
    if m:
        n, title = m.groups()
        return f"Table {n} summarizes {title.rstrip('.')}."
    return label
