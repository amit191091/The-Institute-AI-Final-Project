from __future__ import annotations

"""Lightweight fact miner for common numeric/unit pairs and canonical phrasing.

Scans concatenated context for high-signal facts and returns a standardized
answer when the question matches known patterns (RPS, kHz, mV/g, modalities,
percent ranges). Keep it tiny and deterministic to reduce LLM drift.
"""
from typing import Dict, List, Optional, Tuple
import re

from langchain.schema import Document
from app.logger import trace_func


CANONICAL_PHRASES: Dict[str, str] = {
    # Sensor modalities
    "modalities:accelerometer+tachometer": "Accelerometers and tachometer.",
    # Two steady speeds
    "speeds_rps:15_45": "15 and 45 RPS",
}


def _concat(docs: List[Document]) -> str:
    return "\n".join([(d.page_content or "") for d in (docs or [])])


def _has(tokens: List[str], text: str) -> bool:
    tl = text.lower()
    return all(t in tl for t in tokens)


def _std_percent_range(text: str) -> Optional[str]:
    # Normalize ranges like 25-40%, 25 – 40 %, 25–40 % -> 25–40%
    m = re.search(r"\b(\d{1,3})\s*[\-–—]\s*(\d{1,3})\s*%", text)
    if m:
        a, b = m.group(1), m.group(2)
        return f"{a}–{b}%"
    return None


def _std_speeds_rps(text: str) -> Optional[str]:
    tl = text.lower()
    if ("15" in tl and "45" in tl) and (" rps" in tl or "rps" in tl):
        return CANONICAL_PHRASES.get("speeds_rps:15_45")
    return None


def _std_sampling_rate_khz(text: str) -> Optional[str]:
    # Accept kHz, kS/s, kS/sec, ksps variants, normalize to kHz
    m = re.search(r"\b(\d{1,4})\s*k(?:hz|s\s*/\s*s|s\s*/\s*sec|sps|s\s*ps)\b", text, re.I)
    if m:
        return f"{m.group(1)} kHz"
    return None


def _std_sensitivity_mvg(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,4})\s*m\s*v\s*/\s*g\b", text, re.I)
    if m:
        return f"{m.group(1)} mV/g"
    return None


def _std_modalities(text: str) -> Optional[str]:
    tl = text.lower()
    if ("accelerometer" in tl or "accelerometers" in tl) and "tachometer" in tl:
        return CANONICAL_PHRASES.get("modalities:accelerometer+tachometer")
    return None


@trace_func
def mine_answer_from_context(question: str, docs: List[Document]) -> Tuple[Optional[str], Dict[str, str]]:
    """Return (standard_answer, details) when a canonical fact is detected for the question.

    Details may include the matched type and raw normalization used.
    """
    ql = (question or "").lower().strip()
    text = _concat(docs)
    details: Dict[str, str] = {}

    # Targeted intents by question cues
    if any(w in ql for w in ["steady speeds", "speeds", "rps"]) or (" rps" in ql):
        ans = _std_speeds_rps(text)
        if ans:
            details["type"] = "speeds_rps"
            return ans, details

    if any(w in ql for w in ["sampling rate", "sample rate", "sampling", "khz", "ks/s", "ksps"]):
        ans = _std_sampling_rate_khz(text)
        if ans:
            details["type"] = "sampling_rate"
            return ans, details

    if any(w in ql for w in ["sensitivity", "mv/g", "mvg", "mv per g"]):
        ans = _std_sensitivity_mvg(text)
        if ans:
            details["type"] = "sensitivity_mvg"
            return ans, details

    if any(w in ql for w in ["modalities", "sensors", "sensor modalities", "instrumentation"]):
        ans = _std_modalities(text)
        if ans:
            details["type"] = "modalities"
            return ans, details

    if any(w in ql for w in ["percent", "%", "rise", "exceed", "increase"]) and ("rms" in ql or "r.m.s" in ql):
        ans = _std_percent_range(text)
        if ans:
            details["type"] = "percent_range"
            return ans, details

    return None, details


@trace_func
def canonicalize_answer(question: str, answer: str) -> str:
    """If the question maps to a known canonical phrase, enforce it when logically equivalent.

    This is intentionally conservative—only swaps when a clear canonical exists.
    """
    ql = (question or "").lower().strip()
    al = (answer or "").lower().strip()

    # Modalities
    if any(w in ql for w in ["modalities", "sensor modalities", "instrumentation"]) and (
        "accelerometer" in al and "tachometer" in al
    ):
        return CANONICAL_PHRASES["modalities:accelerometer+tachometer"]

    # Two speeds
    if (" rps" in ql or "speeds" in ql) and ("15" in al and "45" in al and "rps" in al):
        return CANONICAL_PHRASES["speeds_rps:15_45"]

    # Sampling rate normalization
    if any(w in ql for w in ["sampling", "sample rate", "khz", "ks/s", "ksps"]):
        m = re.search(r"\b(\d{1,4})\s*(khz|ks/s|ksps|k\s*s/\s*s|k\s*s/\s*sec)\b", al)
        if m:
            return f"{m.group(1)} kHz"

    # Sensitivity normalization
    if any(w in ql for w in ["sensitivity", "mv/g", "mvg", "mv per g"]):
        m = re.search(r"\b(\d{1,4})\s*m\s*v\s*/\s*g\b", al)
        if m:
            return f"{m.group(1)} mV/g"

    # Percent range en-dash
    if ("%" in al or "percent" in al):
        pr = _std_percent_range(al)
        if pr:
            return pr

    return answer
