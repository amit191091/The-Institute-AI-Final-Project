from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional


def approx_token_len(text: str) -> int:
	# very rough approx: 1 token ~ 4 chars
	return max(1, math.ceil(len(text) / 4))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
	max_chars = max_tokens * 4
	if len(text) <= max_chars:
		return text
	return text[:max_chars]


def simple_summarize(text: str, ratio: float = 0.1, min_lines: int = 1) -> str:
	if not text:
		return ""
	lines = [l.strip() for l in text.splitlines() if l.strip()]
	if not lines:
		return text[: max(1, int(len(text) * ratio))]
	n = max(min_lines, int(len(lines) * ratio))
	return "\n".join(lines[: max(1, n)])


def naive_markdown_table(text: str) -> str:
	"""Heuristic conversion of delimited text to Markdown table."""
	lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
	if not lines:
		return ""
	rows = []
	for ln in lines:
		if "\t" in ln:
			cells = [c.strip() for c in ln.split("\t")]
		elif "," in ln:
			cells = [c.strip() for c in ln.split(",")]
		else:
			cells = [c for c in ln.split()]
		rows.append(cells)
	width = max((len(r) for r in rows), default=0)
	rows = [r + [""] * (width - len(r)) for r in rows]
	if not rows:
		return ""
	header = rows[0]
	sep = ["---" for _ in header]
	body = rows[1:] if len(rows) > 1 else []
	fmt = lambda r: "| " + " | ".join(r) + " |"
	out = [fmt(header), fmt(sep)] + [fmt(r) for r in body]
	return "\n".join(out)


def read_api_key_from_file(file_path: Path) -> Optional[str]:
    """Read API key from a local file."""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                key = f.read().strip()
                return key if key else None
        return None
    except Exception:
        return None


def get_google_api_key() -> Optional[str]:
    """Get Google API key from local file."""
    from app.config import settings
    return read_api_key_from_file(settings.GOOGLE_API_KEY_PATH)


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from local file."""
    from app.config import settings
    return read_api_key_from_file(settings.OPENAI_API_KEY_PATH)

