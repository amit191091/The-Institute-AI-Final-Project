from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("μ", "u")  # normalize micro symbol
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _norm_text(s)
    # keep alphanumerics and unit characters
    s = re.sub(r"[^a-z0-9%/.-]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def token_f1(a: str, b: str) -> float:
    A = set(_tokenize(a))
    B = set(_tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    p = inter / max(1, len(A))
    r = inter / max(1, len(B))
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def exact_match_normalized(a: str, b: str) -> bool:
    return _norm_text(a) == _norm_text(b)


def _extract_percent_ranges(s: str) -> List[Tuple[float, float]]:
    s = _norm_text(s)
    # patterns like 25-40%, 25–40%, 25 — 40 %, 25 to 40 %
    dash = r"[-\u2012-\u2015\u2212]"
    pat1 = re.compile(rf"(\d+(?:\.\d+)?)\s*(?:{dash}|to)\s*(\d+(?:\.\d+)?)\s*%")
    out = []
    for m in pat1.finditer(s):
        lo = float(m.group(1))
        hi = float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def _extract_percents(s: str) -> List[float]:
    s = _norm_text(s)
    vals = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", s):
        vals.append(float(m.group(1)))
    return vals


def _unit_norm(unit: str) -> str:
    u = (unit or "").strip().lower()
    u = u.replace("μ", "u")
    mappings = {
        "ks/sec": "khz",
        "ks/s": "khz",
        "k s/s": "khz",
        "k s/sec": "khz",
        "khz": "khz",
        "hz": "hz",
        "um": "um",
        "μm": "um",
        "rps": "rps",
        "rpm": "rpm",
        "s": "s",
        "sec": "s",
        "second": "s",
        "seconds": "s",
    }
    return mappings.get(u, u)


def _extract_numbers_with_units(s: str) -> List[Tuple[float, str]]:
    s = _norm_text(s)
    out: List[Tuple[float, str]] = []
    # simple patterns: number + optional unit token right after
    for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*)([a-z%/]+)?", s):
        val = float(m.group(1))
        unit = _unit_norm(m.group(2) or "")
        out.append((val, unit))
    return out


def _compatible(val_a: float, unit_a: str, val_b: float, unit_b: str) -> bool:
    # Treat ks/sec == khz equivalence for sampling rates
    ua, ub = _unit_norm(unit_a), _unit_norm(unit_b)
    if ua == ub:
        return val_a == val_b
    if {ua, ub} == {"khz", "ks/sec"} or {ua, ub} == {"khz", "ks/s"}:
        return val_a == val_b
    return False


def numeric_agreement(a: str, b: str) -> float:
    # Exact numeric equality for matching positions/units; 1.0 if all reference numbers matched in answer
    A = _extract_numbers_with_units(a)
    B = _extract_numbers_with_units(b)
    if not B:
        return 0.0
    matched = 0
    for vb, ub in B:
        ok = False
        for va, ua in A:
            if _compatible(va, ua, vb, ub) or (ua == ub and abs(va - vb) < 1e-9):
                ok = True
                break
        matched += 1 if ok else 0
    return matched / len(B)


def percent_range_agreement(a: str, b: str, tol: float = 0.5) -> float:
    ra = _extract_percent_ranges(a)
    rb = _extract_percent_ranges(b)
    if not rb:
        # Try single percents if range not present in ref
        pa = _extract_percents(a)
        pb = _extract_percents(b)
        if not pb:
            return 0.0
        matched = 0
        for v in pb:
            ok = any(abs(v - x) <= tol for x in pa)
            matched += 1 if ok else 0
        return matched / len(pb)
    # Compare ranges: both bounds within tol
    matched = 0
    for (lo_b, hi_b) in rb:
        ok = any(abs(lo_b - lo_a) <= tol and abs(hi_b - hi_a) <= tol for (lo_a, hi_a) in ra)
        matched += 1 if ok else 0
    return matched / len(rb)


def list_f1(a: str, b: str) -> float:
    # Treat comma/semicolon/newline as separators; compare as sets
    def split_items(s: str) -> List[str]:
        s = _norm_text(s)
        parts = re.split(r"[\n,;]+", s)
        return [p.strip() for p in parts if p.strip()]
    A, B = set(split_items(a)), set(split_items(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    p = inter / max(1, len(A))
    r = inter / max(1, len(B))
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_factual_metrics(answer: str, reference: str) -> Dict[str, Any]:
    ans = answer or ""
    ref = reference or ""
    em = exact_match_normalized(ans, ref)
    tf1 = token_f1(ans, ref)
    num = numeric_agreement(ans, ref)
    pr = percent_range_agreement(ans, ref)
    lf1 = list_f1(ans, ref)
    # Aggregate score: average of available signals (EM counts as 1/0)
    vals = [float(em), tf1, num, pr, lf1]
    factual = sum(vals) / len(vals)
    return {
        "factual_em": bool(em),
        "factual_token_f1": float(tf1),
        "factual_numeric": float(num),
        "factual_range": float(pr),
        "factual_list_f1": float(lf1),
        "factual_score": float(factual),
    }
