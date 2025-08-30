#!/usr/bin/env python3
"""
Date Parsers
===========

Date parsing and timeline logic functions.
"""

import re
from typing import Optional, Tuple

from RAG.app.logger import trace_func


MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


@trace_func
def anchor_sort_key(anchor: str, page: int) -> Tuple[int, Tuple[int, str]]:
    # Sort by page, then by anchor type+numeric
    # anchors could be p12-t3, figure-2, table-03
    m = re.search(r"(figure|table|p)(?:-|)(\d+)?(?:.*?(?:t|img)?(\d+))?", anchor)
    a_type = 3
    a_num = 0
    if m:
        t = m.group(1) or "p"
        if t == "p":
            a_type = 0
        elif t == "figure":
            a_type = 1
        elif t == "table":
            a_type = 2
        # prefer the last numeric group as the specific order
        nums = [g for g in m.groups()[1:] if g and g.isdigit()]
        if nums:
            a_num = int(nums[-1])
    return (page, (a_type, f"{a_num:06d}"))


@trace_func
def infer_stage(date_s: Optional[str], date_e: Optional[str], single_date: Optional[str]) -> Optional[str]:
    # date thresholds as per spec
    # <= 2023-04-08 baseline
    # 2023-04-09 to < 2023-04-23 mild
    # 2023-04-23 to <= 2023-05-07 moderate
    # 2023-05-14 to <= 2023-06-11 severe
    # = 2023-06-15 failure
    def in_range(d: str, start: str, end: str, inclusive_end=True) -> bool:
        if inclusive_end:
            return start <= d <= end
        return start <= d < end

    d = single_date
    if d:
        if d == "2023-06-15":
            return "failure"
        if in_range(d, "2023-05-14", "2023-06-11"):
            return "severe"
        if in_range(d, "2023-04-23", "2023-05-07"):
            return "moderate"
        if in_range(d, "2023-04-09", "2023-04-22"):
            return "mild"
        if d <= "2023-04-08":
            return "baseline"
        return None
    if date_s and date_e:
        if in_range(date_s, "2023-05-14", "2023-06-11") and in_range(date_e, "2023-05-14", "2023-06-11"):
            return "severe"
        if in_range(date_s, "2023-04-23", "2023-05-07") and in_range(date_e, "2023-04-23", "2023-05-07"):
            return "moderate"
        if date_e <= "2023-04-08":
            return "baseline"
    return None


@trace_func
def parse_dates(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Returns (date_start, date_end, date)
    # Year is assumed 2023 as per spec.
    tl = text.lower()
    # From Month DD to Month DD
    m = re.search(r"from\s+([a-z]+)\s+(\d{1,2})\s+(?:to|\u2192|\-|Γאף)\s+([a-z]+)\s+(\d{1,2})", tl)
    if m:
        m1, d1, m2, d2 = m.groups()
        mm1 = MONTHS.get(m1, None)
        mm2 = MONTHS.get(m2, None)
        if mm1 and mm2:
            return (f"2023-{mm1}-{int(d1):02d}", f"2023-{mm2}-{int(d2):02d}", None)
    # On Month DD
    m = re.search(r"on\s+([a-z]+)\s+(\d{1,2})", tl)
    if m:
        mon, dd = m.groups()
        mm = MONTHS.get(mon, None)
        if mm:
            return (None, None, f"2023-{mm}-{int(dd):02d}")
    return (None, None, None)
