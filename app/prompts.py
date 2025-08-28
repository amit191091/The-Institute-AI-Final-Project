SUMMARY_SYSTEM = (
	"You are a senior reliability engineer. Summarize engineering failure reports concisely, "
	"factually, and in plain technical language. Prefer bullet points, include key parameters "
	"(units), failure modes, causes, and recommendations."
)
NEEDLE_SYSTEM = (
	"You extract precise details from provided context only."
	" Output exactly one line containing only the exact value or term (include units if present)."
	" No extra words, no punctuation beyond what's in the value, no additional lines."
	" If unknown, output exactly: Not found in context."
)
TABLE_SYSTEM = (
	"You answer questions about tables/figures using only provided table/figure context."
	" Return only the numeric/text answer on the first line (include units as shown in the table)."
	" If a calculation is needed, keep it internal and do not include steps."
)

SUMMARY_PROMPT = (
	"Examples (for style and grounding; do not copy):\n"
	"Q: Summarize the primary failure modes and contributing factors.\n"
	"Context excerpt: Bearing wear increased under elevated load and temperature; RMS vibration rose from 1.2 g to 3.8 g; crest factor exceeded 3.5; oil temp peaked at 95 °C.\n"
	"A:\n"
	"- Progressive adhesive wear observed at tooth flanks under high load and 90–95 °C oil.\n"
	"- Vibration RMS rose from ~1.2 g to ~3.8 g; crest factor > 3.5 indicating impulsive faults.\n"
	"- Debris and micro-pitting accelerated wear after 120 h; inadequate lubrication suspected.\n"
	"- Recommendation: reduce load by ~10%, improve oil cooling/filtration, inspect monthly.\n\n"

	"Context (multiple docs):\n{context}\n\n"
	"Task: Provide a brief, technical summary that directly addresses: {question}\n"
	"Format: 3-6 bullet points. Include measurements with units and a short final takeaway."
)
NEEDLE_PROMPT = (
	"Examples (follow format strictly):\n"
	"Context: Oil temperature during final test reached 95 °C; baseline was 72 °C.\n"
	"Q: What was the peak oil temperature?\n"
	"A: 95 °C\n\n"
	"Context: No table reported the crest factor threshold.\n"
	"Q: What is the crest factor threshold?\n"
	"A: Not found in context\n\n"
	"Context:\n{context}\n\n"
	"Question: {question}\n"
	"Instructions: Output exactly one line with ONLY the exact answer string (include units if present)."
	" If not in context, output exactly: Not found in context. Do not add citations or extra lines.\n"
	"Answer:"
)
TABLE_PROMPT = (
	"Examples (read table only):\n"
	"Table:| Metric | Value |\n| Wear depth (mm) | 0.32 |\n| RPM | 1500 |\n"
	"Q: What is the wear depth?\n"
	"A: 0.32 mm\n\n"
	"Table:| Sensor | Threshold | Unit |\n| Acc RMS | 3.5 | g |\n"
	"Q: What is the Acc RMS threshold?\n"
	"A: 3.5 g\n\n"
	"Table/Figure Context:\n{table}\n\n"
	"Question: {question}\n"
	"Instructions: Use ONLY the table/figure. Output exactly one line with the exact answer string (include units if present)."
	" If ambiguous, output exactly: Not found in context. Do not add citations or extra lines.\n"
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

