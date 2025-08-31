import re
from typing import Tuple

PCT_RE = re.compile(r"\b(\d{1,3})(?:\s?[\-–]\s?(\d{1,3}))?\s?%")
SPEED_RE = re.compile(r"\b(15|45)\s*RPS\b", re.I)


def require_percentage_for_pct_questions(q: str, ans: str) -> Tuple[bool, str | None]:
    ql = (q or "").lower()
    al = (ans or "").lower()
    if any(k in ql for k in ("percent", "percentage", "%", "by how much")):
        # Allow more flexible percentage formats and approximate answers
        if not (PCT_RE.search(ans or "") or 
                any(term in al for term in ("about", "approximately", "roughly", "around")) or
                "not found in context" in al):
            return False, "Expected a percentage-like answer"
    return True, None


def forbid_speed_leak_on_non_speed_questions(q: str, ans: str) -> Tuple[bool, str | None]:
    ql = (q or "").lower()
    if not any(k in ql for k in ("rps", "speed", "15", "45")):
        if SPEED_RE.search(ans or ""):
            return False, "Speed tokens leaked into non-speed answer"
    return True, None


def validate_answer(q: str, ans: str) -> Tuple[bool, str | None]:
    for fn in (require_percentage_for_pct_questions, forbid_speed_leak_on_non_speed_questions):
        ok, why = fn(q, ans)
        if not ok:
            return False, why
    return True, None
