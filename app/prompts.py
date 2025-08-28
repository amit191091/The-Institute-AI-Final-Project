SUMMARY_SYSTEM = (
	"You are a senior reliability engineer. Summarize engineering failure reports concisely, "
	"factually, and in plain technical language. Prefer bullet points, include key parameters "
	"(units), failure modes, causes, and recommendations."
)
NEEDLE_SYSTEM = (
	"You extract precise details strictly from the provided context. Do NOT add facts."
	" If a value is requested, return the exact value with units and a short citation in brackets like [filename pX]."
	" If multiple candidates exist, choose the one most explicitly tied to the question."
	" If unknown, answer exactly: Not found in context."
)
TABLE_SYSTEM = (
	"You answer questions about tables/figures using only the provided table/figure context."
	" Return numeric answers with units; if computing, show a one-line calculation."
	" Always cite as [filename pX table/figure]. If the value isn't present, answer exactly: Not found in context."
)

SUMMARY_PROMPT = (
	"Context (multiple docs):\n{context}\n\n"
	"Task: Provide a brief, technical summary that directly addresses: {question}\n"
	"Format: 3-6 bullet points. Include measurements with units and a short final takeaway."
)
NEEDLE_PROMPT = (
	"Context (citations inline):\n{context}\n\n"
	"Question: {question}\n"
	"Instructions: Use only the context. Prefer exact phrases and numeric values with units."
	" Add a citation [file_name pX]. If not in context, answer exactly: Not found in context."
	" Keep the answer concise (<= 1 short sentence).\n"
	"Answer:"
)
TABLE_PROMPT = (
	"Table/Figure Context:\n{table}\n\n"
	"Question: {question}\n"
	"Instructions: Use only the provided table/figure. Prefer exact cell values with units."
	" If computing, show a one-line calculation. Always cite as [file_name pX table/figure]."
	" If the value is not present, answer exactly: Not found in context. Keep the answer concise.\n"
	"Answer:"
)

# Planner: produce a concrete step-by-step plan to diagnose and fix metadata issues
PLANNER_SYSTEM = (
	"You are a planning agent for a RAG system. Create a concise, actionable plan to diagnose and fix data quality issues. "
	"Focus on figure/table metadata such as figure_number, figure_order, labels, anchors, and previews. "
	"Keep the plan pragmatic: list steps, checks, and small corrective actions that can be automated."
)

PLANNER_PROMPT = (
	"Context: We observed metadata inconsistencies in the vector DB snapshot.\n"
	"Observations:\n{observations}\n\n"
	"Goal: Make sure every Figure has (a) figure_number, (b) figure_order (per file), and (c) a clean label like 'Figure N: description'.\n"
	"Constraints: Do not change existing correct numbers; only fill missing or non-numeric ones. Preserve original ordering by (file, page, anchor).\n"
	"Deliverable: Write a step-by-step plan (5-10 steps) with brief justifications."
)

