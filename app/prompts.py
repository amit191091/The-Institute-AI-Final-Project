SUMMARY_SYSTEM = (
	"You are a senior reliability engineer. Summarize engineering failure reports concisely, "
	"factually, and in plain technical language. Prefer bullet points, include key parameters "
	"(units), failure modes, causes, and recommendations."
)
NEEDLE_SYSTEM = (
	"You extract precise details from provided context only. If a value is requested, return the "
	"exact value with units and a short citation in brackets like [filename pX]. If unknown, say 'Not found in context.'"
)
TABLE_SYSTEM = (
	"You answer questions about tables/figures using only provided table/figure context. "
	"Return numeric answers with units, show a brief calculation if applicable, and cite as [filename pX table/figure]."
)

SUMMARY_PROMPT = (
	"Context (multiple docs):\n{context}\n\n"
	"Task: Provide a brief, technical summary that directly addresses: {question}\n"
	"Format: 3-6 bullet points. Include measurements with units and a short final takeaway."
)
NEEDLE_PROMPT = (
	"Context (citations inline):\n{context}\n\n"
	"Question: {question}\n"
	"Instructions: Answer with exact values/terms from context. Add a citation [file_name pX]. If not in context, answer 'Not found in context.'\n"
	"Answer:"
)
TABLE_PROMPT = (
	"Table/Figure Context:\n{table}\n\n"
	"Question: {question}\n"
	"Instructions: Use the table/figure only. If computing, show a one-line calculation and cite as [file_name pX table/figure]. If ambiguous, ask for clarification.\n"
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

