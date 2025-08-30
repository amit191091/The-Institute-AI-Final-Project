from __future__ import annotations

from RAG.app.logger import trace_func
import math
import hashlib
import re


@trace_func
def slugify(s: str) -> str:
	s = (s or "").strip().lower()
	s = re.sub(r"[^a-z0-9\-_.]+", "-", s)
	s = re.sub(r"-+", "-", s).strip("-")
	return s or "doc"


@trace_func
def sha1_short(text: str, n: int = 12) -> str:
	try:
		h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
		return h[: max(1, n)]
	except Exception:
		return "0" * n


@trace_func
def approx_token_len(text: str) -> int:
	# very rough approx: 1 token ~ 4 chars
	return max(1, math.ceil(len(text) / 4))


@trace_func
def truncate_to_tokens(text: str, max_tokens: int) -> str:
	max_chars = max_tokens * 4
	if len(text) <= max_chars:
		return text
	return text[:max_chars]


@trace_func
def simple_summarize(text: str, ratio: float = 0.1, min_lines: int = 1) -> str:
	if not text:
		return ""
	lines = [l.strip() for l in text.splitlines() if l.strip()]
	if not lines:
		return text[: max(1, int(len(text) * ratio))]
	n = max(min_lines, int(len(lines) * ratio))
	return "\n".join(lines[: max(1, n)])


@trace_func
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


@trace_func
def split_into_sentences(text: str) -> list[str]:
	if not text:
		return []
	# Simple sentence splitter; good enough for chunking boundaries
	parts = re.split(r"(?<=[.!?])\s+", text.strip())
	return [p.strip() for p in parts if p.strip()]


@trace_func
def split_into_paragraphs(text: str) -> list[str]:
	"""Split text into paragraphs using blank lines as separators; collapse intra-paragraph whitespace."""
	if not text:
		return []
	# Normalize line endings and split on 2+ newlines
	blocks = re.split(r"\n\s*\n+", text.strip())
	out: list[str] = []
	for b in blocks:
		# Collapse internal newlines to spaces
		b2 = re.sub(r"\s*\n\s*", " ", b.strip())
		if b2:
			out.append(b2)
	return out

