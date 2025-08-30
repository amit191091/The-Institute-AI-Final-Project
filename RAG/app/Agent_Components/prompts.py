# Primary prompts - Simple, reliable versions from app/ folder
# These avoid JSON parsing failures and use traditional inline citations
SUMMARY_SYSTEM = (
	"You are a senior reliability engineer. Summarize engineering failure reports concisely, "
	"factually, and in plain technical language. Prefer bullet points, include key parameters "
	"(units), failure modes, causes, and recommendations."
)
NEEDLE_SYSTEM = (
	"You extract precise details strictly from the provided context. Do NOT add facts."
	" If a value is requested, return the exact value with units and a short citation in brackets like [filename pX]."
	" If multiple candidates exist, choose the one most explicitly tied to the question."
	" For dates, include the full date with year (e.g., 'April 9, 2023' not just 'April 9')."
	" For equipment questions, look for brand names, model numbers, and specifications."
	" For measurement questions, look for exact values with units (μm, kHz, mV/g, etc.)."
	" For technical terms, look for specific terminology and definitions."
	" If unknown, answer exactly: Not found in context."
)

# JSON version (alternative) - may cause parsing failures
NEEDLE_SYSTEM_JSON = (
	"You are an EXTRACTIVE QA assistant. Answer ONLY from the provided CONTEXT.\n"
	"Rules:\n"
	"1) Answer with a single short phrase/number/date/figure label if possible.\n"
	"2) Do NOT add explanations or extra text.\n"
	"3) If not answerable from context, reply exactly: 'Not found in document context'.\n"
	"4) Use the exact wording, numbers, and units from the source you cite.\n"
	"5) Wear-depth questions: look for case IDs (W1, W2, ...), return '<number> μm'. If unit is in the column header, include it.\n"
	"6) Range questions: For queries asking about ranges (e.g., 'greater than X and less than Y'), return ALL case IDs (W1, W2, etc.) that fall within the range, separated by commas.\n"
	"7) Figure questions: return only 'Figure N'.\n"
	"8) Count questions: return only the number (e.g., '30').\n"
	"9) Speed pairs: list in ascending order (e.g., '15 RPS and 45 RPS').\n"
	"10) NO crest factor content unless present in context.\n"
	"11) OUTPUT FORMAT: Return JSON with two fields only:\n"
	"    {\"answer\":\"<short string>\", \"citations\":[{\"file\":\"<name>\",\"page\":<int>,\"anchor\":\"<id>\"}]}\n"
	"   The 'answer' must not contain citations or extra words."
)

TABLE_SYSTEM = (
	"You answer questions about tables/figures using only the provided table/figure context."
	" Return numeric answers with units; if computing, show a one-line calculation."
	" Always cite as [filename pX table/figure]. If the value isn't present, answer exactly: Not found in context."
)

# JSON version (alternative) - may cause parsing failures
TABLE_SYSTEM_JSON = (
	"You answer questions about tables/figures using ONLY the provided table/figure context.\n"
	"Rules:\n"
	"1) Return the exact value with the unit as shown by the table header/entry.\n"
	"2) Do NOT add explanations or calculations unless the question explicitly asks to compute.\n"
	"3) If not answerable from table/figure, reply exactly: 'Not found in document context'.\n"
	"4) OUTPUT FORMAT: JSON as {\"answer\":\"<short>\", \"citations\":[{\"file\":\"<name>\",\"page\":<int>,\"anchor\":\"<id>\"}]}\n"
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
	" For dates, include the full date with year (e.g., 'April 9, 2023' not just 'April 9')."
	" Add a citation [file_name pX]. If not in context, answer exactly: Not found in context."
	" Keep the answer to one short sentence (max ~20 words).\n"
	"Answer:"
)

# JSON version (alternative) - may cause parsing failures
NEEDLE_PROMPT_JSON = (
	"Context:\n{context}\n\n"
	"Question: {question}\n\n"
	"IMPORTANT: For table questions, read the ENTIRE table from top to bottom to find ALL matching entries.\n"
	"Answer in JSON exactly as:\n"
	"{\"answer\":\"<short string>\", \"citations\":[{\"file\":\"<name>\",\"page\":<int>,\"anchor\":\"<id>\"}]}\n"
)

TABLE_PROMPT = (
	"Table/Figure Context:\n{table}\n\n"
	"Question: {question}\n"
	"Instructions: Use only the provided table/figure. Prefer exact cell values with units."
	" If computing, show a one-line calculation. Always cite as [file_name pX table/figure]."
	" If the value is not present, answer exactly: Not found in context. Keep the answer to one short sentence.\n"
	"Answer:"
)

# JSON version (alternative) - may cause parsing failures
TABLE_PROMPT_JSON = (
	"Table/Figure Context:\n{table}\n\n"
	"Question: {question}\n\n"
	"Answer in JSON exactly as:\n"
	"{\"answer\":\"<short string>\", \"citations\":[{\"file\":\"<name>\",\"page\":<int>,\"anchor\":\"<id>\"}]}\n"
)

# Minimal few-shot patterns to guide extractive behavior (aligned with dataset) - copied from main app
FEWSHOT_NEEDLE = [
	{"q": "What two steady speeds were used for data acquisition (in RPS)?", "a": "45 RPS and 15 RPS [Gear wear Failure.pdf pX]."},
	{"q": "What was the sampling rate per record?", "a": "50 kHz [Gear wear Failure.pdf pX]."},
	{"q": "Which lubricant and viscosity grade were in service?", "a": "2640 semi-synthetic (15W/40) [Gear wear Failure.pdf pX]."},
	{"q": "What lubricant brand was used?", "a": "Not found in context."},
	{"q": "On what date was the first onset of wear detected by visual inspection?", "a": "April 9, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "When did the system reach the failure stage?", "a": "June 15, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "What was the duration of each time record?", "a": "60 seconds [Gear wear Failure.pdf pX]."},
	{"q": "What brand of accelerometers was used?", "a": "Dytran 3053B [Gear wear Failure.pdf pX]."},
	{"q": "What was the baseline wear depth (Healthy)?", "a": "0 μm [Gear wear Failure.pdf pX]."},
]

FEWSHOT_TABLE = [
	{"q": "What is the wear depth for case W24 (in μm)?", "a": "579 μm [Gear wear Failure.pdf pX table]."},
	{"q": "Which wear case corresponds to 466 μm?", "a": "W19 [Gear wear Failure.pdf pX table]."},
	{"q": "What is the wear depth for case W33 (in μm)?", "a": "853 μm [Gear wear Failure.pdf pX table]."},
]

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

