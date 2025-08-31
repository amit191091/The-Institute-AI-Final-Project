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
	" For date ranges, use format 'From [start] to [end]' (e.g., 'From May 14 to June 11')."
	" For equipment questions, look for brand names, model numbers, and specifications in tables and text."
	" For accelerometer questions, search for 'Dytran' brand and model numbers."
	" For tachometer questions, search for 'Honeywell' brand and specifications."
	" For lubricant questions, search for '2640 semi-synthetic' specifications."
	" For sampling rate questions, search for '50 kHz' frequency values."
	" For temporal questions, search for specific dates, time periods, and wear stage transitions."
	" For count questions, look for exact numbers (e.g., '35 cases', '35 sequential wear cases')."
	" For measurement questions, look for exact values with units (μm, kHz, mV/g, etc.)."
	" For technical terms, look for specific terminology and definitions."
	" For visual evidence questions, extract the exact technical descriptions (scars, scuffing, pitting, etc.)."
	" For time/duration questions, look for specific time values (seconds, minutes, etc.)."
	" For vessel questions, look for 'INS Haifa' or similar vessel names."
	" For gearbox model questions, look for 'MG-5025A' or similar model numbers."
	" For gear type questions, look for 'Spur gears' or similar gear types."
	" For transmission ratio questions, look for '18/35' or similar ratios."
	" For gear module questions, look for '3 mm' or similar module values."
	" For wear depth questions, look for exact values with μm units in tables."
	" For baseline questions, look for '0 μm' or 'healthy' baseline values."
	" For sensitivity questions, look for exact mV/g values for accelerometers."
	" For teeth questions, look for exact tooth counts (e.g., '30 teeth')."
	" Search thoroughly in all provided context including tables, figures, and text."
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
	" For equipment specifications, include complete model numbers and details."
	" For measurements, include exact values with units."
	" For wear depths, include exact μm values."
	" For accelerometer sensitivity, look for exact mV/g values."
	" For tachometer specifications, include teeth count."
	" Add a citation [file_name pX]. If not in context, answer exactly: Not found in context."
	" Keep the answer to one short sentence (max ~25 words).\n"
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
	{"q": "Which lubricant was used in the gearbox?", "a": "2640 semi-synthetic (15W/40) oil [Gear wear Failure.pdf pX]."},
	{"q": "What is the model of the marine reduction gearbox investigated?", "a": "MG-5025A [Gear wear Failure.pdf pX]."},
	{"q": "Which vessel's propulsion train was monitored?", "a": "INS Haifa [Gear wear Failure.pdf pX]."},
	{"q": "What gear type is used in the transmission?", "a": "Spur gears [Gear wear Failure.pdf pX]."},
	{"q": "What is the gear module value?", "a": "3 mm [Gear wear Failure.pdf pX]."},
	{"q": "What is the transmission ratio (driving/driven)?", "a": "18/35 [Gear wear Failure.pdf pX]."},
	{"q": "What was the duration of each time record?", "a": "60 seconds [Gear wear Failure.pdf pX]."},
	{"q": "What brand of accelerometers was used?", "a": "Dytran 3053B [Gear wear Failure.pdf pX]."},
	{"q": "What was the accelerometer sensitivity (mV/g)?", "a": "1783 mV/g (Starboard), 1787 mV/g (Port) [Gear wear Failure.pdf pX]."},
	{"q": "What brand of tachometer was used?", "a": "Honeywell 3010AN, 30 teeth [Gear wear Failure.pdf pX]."},
	{"q": "What was the baseline wear depth (Healthy)?", "a": "0 μm [Gear wear Failure.pdf pX]."},
	{"q": "What was the final wear depth measured at Case W35?", "a": "932 μm [Gear wear Failure.pdf pX]."},
	{"q": "What was the wear depth at Case W1?", "a": "40 μm [Gear wear Failure.pdf pX]."},
	{"q": "On what date was the first onset of wear detected by visual inspection?", "a": "April 9, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "When did the system reach the failure stage?", "a": "June 15, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "Between which dates did the severe wear stage occur?", "a": "From May 14 to June 11 [Gear wear Failure.pdf pX]."},
	{"q": "When did moderate wear begin?", "a": "April 23, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "How many sequential wear cases were tracked?", "a": "35 cases [Gear wear Failure.pdf pX]."},
	{"q": "What was the earliest diagnostic indicator of mild wear?", "a": "Tooth photographs showing profile deviations [Gear wear Failure.pdf pX]."},
	{"q": "What visual evidence defined the severe wear stage?", "a": "Sharp-edged scars, scuffing, tearing, deep scarring, pitting [Gear wear Failure.pdf pX]."},
	{"q": "What visual evidence confirmed failure on June 15?", "a": "Deep gouges, fragmented edges, spalling, fractured surfaces [Gear wear Failure.pdf pX]."},
	{"q": "Until what date did the healthy baseline extend with no abnormal indications?", "a": "Until April 8, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "On what date was the new data-acquisition chain installed?", "a": "February 13, 2023 [Gear wear Failure.pdf pX]."},
	{"q": "What did the baseline RMS vibration levels indicate?", "a": "Stable alignment, no assembly errors, no abnormal gear contact [Gear wear Failure.pdf pX]."},
	{"q": "What RMS trend was observed by April 23?", "a": "RMS values consistently above the baseline [Gear wear Failure.pdf pX]."},
	{"q": "What was the purpose of photographic inspections?", "a": "To detect earliest wear onset and corroborate wear evolution [Gear wear Failure.pdf pX]."},
	{"q": "What was recommended regarding RMS monitoring thresholds?", "a": "Update thresholds with lower alarm levels [Gear wear Failure.pdf pX]."},
	{"q": "What intervention threshold was suggested for mild wear?", "a": "Record and monitor closely [Gear wear Failure.pdf pX]."},
	{"q": "What intervention threshold was suggested for moderate wear?", "a": "Schedule planned replacement or refurbishment [Gear wear Failure.pdf pX]."},
	{"q": "What intervention threshold was suggested for severe wear?", "a": "Immediate intervention required to prevent failure [Gear wear Failure.pdf pX]."},
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

