"""
Data loading and ground truth functionality for the Gradio interface.
"""
import json
import re
from pathlib import Path
from RAG.app.logger import get_logger


def _norm_q(s: str) -> str:
    """Normalize question text for comparison."""
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(".,:;!?-Γאפ\u2013\u2014\"'()[]{}")

def _load_gt_file(path: str | dict | object):
	# Load ground truth json/jsonl mapping question -> list of ground truths
	if isinstance(path, dict):
		path = path.get("name") or path.get("path") or ""
	elif hasattr(path, "name"):
		try:
			path = getattr(path, "name")
		except Exception:
			path = str(path)
	if not path:
		return "(no GT file loaded)"
	try:
		p = Path(str(path))
		if not p.exists():
			return f"(GT file not found: {p})"
		path_str = str(p)
		rows = []
		if p.suffix.lower() == ".jsonl":
			with open(p, "r", encoding="utf-8") as f:
				for line in f:
					line=line.strip()
					if not line:
						continue
					try:
						rows.append(json.loads(line))
					except Exception:
						pass
		else:
			loaded = json.load(open(p, "r", encoding="utf-8"))
			# Handle ground_truth_dataset.json format which has a questions array
			if isinstance(loaded, dict) and "questions" in loaded:
				rows = loaded["questions"]
			else:
				rows = loaded
		m: dict[str, list[str]] = {}
		for r in rows or []:
			if not isinstance(r, dict):
				continue
			q = r.get("question") or r.get("q") or r.get("prompt") or r.get("key")
			gts = r.get("ground_truths") or r.get("ground_truth") or r.get("answers") or r.get("answer") or r.get("reference") or r.get("value")
			if not q:
				continue
			# normalize to list[str]
			vals: list[str] = []
			if isinstance(gts, str):
				vals = [gts]
			elif isinstance(gts, list):
				vals = [str(x) for x in gts]
			elif gts is not None:
				vals = [str(gts)]
			if not vals:
				continue
			key = str(q)
			m.setdefault(key, [])
			# de-duplicate while preserving order
			seen = set(m[key])
			for v in vals:
				if v not in seen:
					m[key].append(v)
					seen.add(v)
		return m, f"Loaded {len(m)} ground truths from {path_str}. Sample keys: {', '.join(list(m.keys())[:2])}"
	except Exception as e:
		return {}, f"(failed to load: {e})"

def _load_qa_file(path: str | dict | object):
	# Load QA json/jsonl mapping question -> answer
	if isinstance(path, dict):
		path = path.get("name") or path.get("path") or ""
	elif hasattr(path, "name"):
		try:
			path = getattr(path, "name")
		except Exception:
			path = str(path)
	if not path:
		return "(no QA file loaded)"
	try:
		p = Path(str(path))
		if not p.exists():
			return f"(QA file not found: {p})"
		rows = []
		if p.suffix.lower() == ".jsonl":
			with open(p, "r", encoding="utf-8") as f:
				for line in f:
					line=line.strip()
					if not line:
						continue
					try:
						rows.append(json.loads(line))
					except Exception:
						pass
		else:
			rows = json.load(open(p, "r", encoding="utf-8"))
		m = {}
		for r in rows or []:
			if not isinstance(r, dict):
				continue
			q = r.get("question") or r.get("q")
			a = r.get("answer") or r.get("reference")
			if q and a:
				m[str(q)] = str(a)
		qa_map["__loaded__"] = True
		qa_map["map"] = m
		qa_map["norm"] = { _norm_q(k): v for k, v in m.items() }
		return f"Loaded {len(m)} QA pairs from {p}"
	except Exception as e:
		return f"(failed to load QA: {e})"

def load_ground_truth_and_qa():
    """Auto-load default ground truths and QA if files exist."""
    gt_map = {"__loaded__": False, "map": {}, "norm": {}}
    qa_map = {"__loaded__": False, "map": {}, "norm": {}}
    log = get_logger()
    
    # Auto-load default ground truths and QA if files exist
    try:
        for _cand in [
            Path("RAG/data/gear_wear_qa.jsonl"),  # Primary source for ground truth
            Path("RAG/data/ground_truth_dataset.json"),
            Path("gear_wear_ground_truth_context_free.json"),
            Path("data")/"gear_wear_ground_truth_context_free.json",
            Path("gear_wear_ground_truth.json"),
            Path("data")/"gear_wear_ground_truth.json",
        ]:
            if _cand.exists():
                m, msg = _load_gt_file(str(_cand))
                if m:
                    gt_map["__loaded__"] = True
                    gt_map["map"] = m
                    gt_map["norm"] = { _norm_q(k): v for k, v in m.items() }
                    log.info("GT auto-load: %s", msg)
                    break
    except Exception:
        pass
    try:
        for _cand in [
            Path("RAG/data/gear_wear_qa_context_free.jsonl"),
            Path("RAG/data/gear_wear_qa.jsonl"),
            Path("gear_wear_qa_context_free.jsonl"),
            Path("data")/"gear_wear_qa_context_free.jsonl",
            Path("gear_wear_qa.jsonl"),
            Path("data")/"gear_wear_qa.jsonl",
            Path("gear_wear_qa.json"),
            Path("data")/"gear_wear_qa.json",
        ]:
            if _cand.exists():
                m, msg = _load_qa_file(str(_cand))
                if m:
                    qa_map["__loaded__"] = True
                    qa_map["map"] = m
                    qa_map["norm"] = { _norm_q(k): v for k, v in m.items() }
                    log.info("QA auto-load: %s", msg)
                    break
    except Exception:
        pass
    
    return gt_map, qa_map
