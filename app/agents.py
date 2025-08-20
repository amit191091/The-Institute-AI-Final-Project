from typing import List, Protocol

from langchain.schema import Document

from app.prompts import (
	NEEDLE_PROMPT,
	NEEDLE_SYSTEM,
	SUMMARY_PROMPT,
	SUMMARY_SYSTEM,
	TABLE_PROMPT,
	TABLE_SYSTEM,
)


class LLMCallable(Protocol):
	def __call__(self, prompt: str) -> str:  # noqa: D401
		...


def route_question(q: str) -> str:
	ql = q.lower()
	if any(w in ql for w in ("summarize", "overview", "overall", "conclusion", "brief")):
		return "summary"
	# Include figure/image cues to route to table/figure handler
	if any(w in ql for w in ("table", "chart", "value", "figure", "fig ", "image", "graph", "plot")):
		return "table"
	return "needle"


def render_context(docs: List[Document], max_chars: int = 8000) -> str:
	out, n = [], 0
	for d in docs:
		piece = f"[{d.metadata.get('file_name')} p{d.metadata.get('page')}] {d.page_content}".strip()
		n += len(piece)
		if n > max_chars:
			break
		out.append(piece)
	return "\n\n".join(out)


def answer_summary(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = SUMMARY_SYSTEM + "\n" + SUMMARY_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


def answer_needle(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = NEEDLE_SYSTEM + "\n" + NEEDLE_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


def answer_table(llm: LLMCallable, docs: List[Document], question: str) -> str:
	table_docs = [d for d in docs if d.metadata.get("section") in ("Table", "Figure")] or docs
	# Sort to push most table-like content first
	table_docs = table_docs + [d for d in docs if d not in table_docs]
	ctx = render_context(table_docs)
	prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
	return llm(prompt).strip()

