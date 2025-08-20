from typing import Dict, List, Optional
import re

SECTION_ENUM = {"Summary", "Timeline", "Table", "Figure", "Analysis", "Conclusion", "Text"}


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
	if k in ("figure", "image"):
		return "Figure"
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
	metadata = {
		"file_name": chunk["file_name"],
		"page": chunk.get("page"),
		"section": chunk.get("section_type"),
		"anchor": chunk.get("anchor"),
		"image_path": chunk.get("image_path"),
	"table_md_path": chunk.get("table_md_path"),
	"table_csv_path": chunk.get("table_csv_path"),
		"table_row_range": chunk.get("table_row_range"),
		"table_col_names": chunk.get("table_col_names"),
		"client_id": client_id,
		"case_id": case_id,
		"keywords": chunk.get("keywords", []),
		"critical_entities": ents,
		"chunk_summary": (chunk["content"].splitlines() or [""])[0][:200],
		"incident_type": inc["IncidentType"],
		"incident_date": inc["IncidentDate"],
		"amount_range": inc["AmountRange"],
	}
	return {"page_content": chunk["content"], "metadata": metadata}

