from typing import Dict, List, Optional
import re

SECTION_ENUM = {"Summary", "Timeline", "Table", "Figure", "Analysis", "Conclusion", "Text", "TableCaption", "FigureCaption"}


def classify_section_type(kind: str, text: str) -> str:
	k = (kind or "").lower()
	t = (text or "").lower()
	if "timeline" in t:
		return "Timeline"
	if "conclusion" in t:
		return "Conclusion"
	if "summary" in t:
		return "Summary"
	if k == "table":
		return "Table"
	if k == "tablecaption":
		return "TableCaption"
	if k in ("figure", "image"):
		return "Figure"
	if k == "figurecaption":
		return "FigureCaption"
	if any(w in t for w in ("analysis", "method", "procedure", "results", "discussion")):
		return "Analysis"
	return "Text"


_STOP = set(
	"the a an and or for of in on to with without by at from into over under than as is are was were be been being this that these those it its their there here".split()
)


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
	words = re.findall(r"[A-Za-z0-9°%]+", text)
	words = [w for w in words if w.lower() not in _STOP]
	return words[:top_n]


def extract_entities(text: str) -> List[str]:
	pats = [
		r"\bCASE[-_ ]?\d+\b",
		r"\bCLIENT[-_ ]?\w+\b",
		r"\bBRG[-_ ]?\w+\b",
		r"\bGEAR[-_ ]?\w+\b",
		r"\b\d{4}-\d{2}-\d{2}\b",
		r"\b\d+(?:\.\d+)?\s?(MPa|RPM|°C|N|kN|mm|Hz|MPH|kW)\b",
	]
	out: List[str] = []
	for p in pats:
		m = re.findall(p, text)
		if not m:
			continue
		if isinstance(m[0], tuple):
			out += [t[0] for t in m if isinstance(t, tuple) and t]
		else:
			out += m
	return sorted(set(out))


def extract_date_tokens(text: str) -> Dict[str, List[str]]:
	"""Extract month/day tokens from text to help date-specific retrieval.
	Returns lowercase month names and day numbers as strings.
	"""
	if not text:
		return {"month_tokens": [], "day_tokens": []}
	months = [
		"january","february","march","april","may","june","july","august","september","october","november","december"
	]
	low = text.lower()
	month_tokens: List[str] = []
	day_tokens: List[str] = []
	# month names
	for m in months:
		if m in low:
			month_tokens.append(m)
	# patterns like "June 13" or "June 13th"
	for m in months:
		for md in re.findall(rf"{m}\\s+(\\d{{1,2}})(?:st|nd|rd|th)?", low):
			day_tokens.append(md)
	# ISO dates 2023-06-13 -> month/day
	for y, mo, da in re.findall(r"(20\\d{2})-(\\d{2})-(\\d{2})", low):
		try:
			mo_i = int(mo)
			da_i = int(da)
			if 1 <= mo_i <= 12:
				month_tokens.append(months[mo_i - 1])
			if 1 <= da_i <= 31:
				day_tokens.append(str(da_i))
		except Exception:
			pass
	# simple slashed dates like 6/13 or 13/6 (ambiguous; record day if <=31)
	for a, b in re.findall(r"\b(\d{1,2})/(\d{1,2})(?:/(?:20)?\d{2})?\b", low):
		try:
			ai = int(a); bi = int(b)
			if 1 <= ai <= 12 and 1 <= bi <= 31:
				day_tokens.append(str(bi))
			elif 1 <= bi <= 12 and 1 <= ai <= 31:
				day_tokens.append(str(ai))
		except Exception:
			pass
	# dedupe
	month_tokens = sorted(set(month_tokens))
	day_tokens = sorted(set(day_tokens))
	return {"month_tokens": month_tokens, "day_tokens": day_tokens}


def extract_incident(text: str) -> Dict[str, Optional[str]]:
	itype = None
	if re.search(r"\bfail(ure|ed)|fracture|fatigue|overheat|seiz(e|ure)\b", text, re.I):
		itype = "Failure"
	idate = None
	m = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
	if m:
		idate = m.group(1)
	amount_range = None
	m2 = re.findall(r"(\d+(?:\.\d+)?)\s?(MPa|RPM|°C|N|kN|mm)", text)
	if m2:
		nums = [float(x[0]) for x in m2]
		try:
			amount_range = f"{min(nums)}-{max(nums)} {m2[0][1]}"
		except Exception:
			amount_range = None
	return {"IncidentType": itype, "IncidentDate": idate, "AmountRange": amount_range}


def attach_metadata(chunk: Dict, client_id: str | None = None, case_id: str | None = None) -> Dict:
	ents = extract_entities(chunk["content"])
	inc = extract_incident(chunk["content"])
	date_toks = extract_date_tokens(chunk["content"]) if chunk.get("content") else {"month_tokens": [], "day_tokens": []}
	metadata = {
		"file_name": chunk["file_name"],
		"page": chunk.get("page"),
	# Prefer the explicit section when present (e.g., "Figure", "Table")
	"section": chunk.get("section") or chunk.get("section_type"),
		"anchor": chunk.get("anchor"),
	# Deterministic identifiers to aid traceability/upserts
	"doc_id": chunk.get("doc_id"),
	"chunk_id": chunk.get("chunk_id"),
	"content_hash": chunk.get("content_hash"),
		"image_path": chunk.get("image_path"),
		"extractor": chunk.get("extractor"),
		"table_number": chunk.get("table_number"),
		"table_label": chunk.get("table_label"),
		"table_md_path": chunk.get("table_md_path"),
		"table_csv_path": chunk.get("table_csv_path"),
		"table_row_range": chunk.get("table_row_range"),
		"table_col_names": chunk.get("table_col_names"),
	# Figure metadata (propagate through for UI/filters/snapshot)
	"figure_number": chunk.get("figure_number"),
	"figure_order": chunk.get("figure_order"),
	"figure_label": chunk.get("figure_label"),
	"figure_caption_original": chunk.get("figure_caption_original"),
	"figure_number_source": chunk.get("figure_number_source"),
	"caption_alignment": chunk.get("caption_alignment"),
	"figure_associated_text_preview": chunk.get("figure_associated_text_preview"),
	"figure_associated_anchor": chunk.get("figure_associated_anchor"),
		"client_id": client_id,
		"case_id": case_id,
		"keywords": chunk.get("keywords", []),
		"critical_entities": ents,
		"chunk_summary": (chunk["content"].splitlines() or [""])[0][:200],
		"incident_type": inc["IncidentType"],
		"incident_date": inc["IncidentDate"],
		"amount_range": inc["AmountRange"],
		"month_tokens": date_toks.get("month_tokens", []),
		"day_tokens": date_toks.get("day_tokens", []),
	}
	return {"page_content": chunk["content"], "metadata": metadata}

