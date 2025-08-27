# RAGAS imports with robust fallbacks
try:
	from ragas import evaluate
	from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
	try:
		from ragas.run_config import RunConfig  # optional in some versions
	except Exception:  # pragma: no cover
		RunConfig = None  # type: ignore
except Exception:  # pragma: no cover
	evaluate = None  # type: ignore
	faithfulness = answer_relevancy = context_precision = context_recall = None  # type: ignore
	RunConfig = None  # type: ignore

# Optional Google safety enums (available in newer google-generativeai)
try:
	from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
except Exception:  # pragma: no cover
	HarmCategory = None  # type: ignore
	HarmBlockThreshold = None  # type: ignore

try:
	from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover
	Dataset = None  # type: ignore

import os
import math
import re
from typing import List, Tuple

# Optional sklearn for overlap F1
try:
	from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore
except Exception:  # pragma: no cover
	precision_score = recall_score = f1_score = None  # type: ignore


def _simple_tokens(text: str) -> List[str]:
	t = (text or "").lower()
	# keep alphanumerics, collapse whitespace
	t = re.sub(r"[^a-z0-9]+", " ", t)
	toks = [w for w in t.split() if len(w) >= 2]
	return toks


def overlap_prf1(reference: str, contexts: List[str]) -> Tuple[float, float, float]:
	"""Compute a simple token-overlap precision/recall/F1 between reference and concatenated contexts.
	If sklearn is available, use f1_score/precision_score/recall_score over token presence vectors.
	Otherwise fall back to set-overlap math.
	"""
	ref_tokens = set(_simple_tokens(reference or ""))
	ctx_tokens = set(_simple_tokens("\n".join(contexts or [])))
	if not ref_tokens and not ctx_tokens:
		return float("nan"), float("nan"), float("nan")
	# Guard all three to satisfy type checker (each may be None if sklearn missing)
	if precision_score is not None and recall_score is not None and f1_score is not None:
		vocab = sorted(ref_tokens.union(ctx_tokens))
		y_true = [1 if v in ref_tokens else 0 for v in vocab]
		y_pred = [1 if v in ctx_tokens else 0 for v in vocab]
		# handle edge cases when all zeros
		try:
			p = precision_score(y_true, y_pred, zero_division=0)
			r = recall_score(y_true, y_pred, zero_division=0)
			f1 = f1_score(y_true, y_pred, zero_division=0)
		except Exception:
			# fallback to set math
			inter = len(ref_tokens & ctx_tokens)
			p = inter / max(1, len(ctx_tokens))
			r = inter / max(1, len(ref_tokens))
			f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
		return float(p), float(r), float(f1)
	# Set-based fallback
	inter = len(ref_tokens & ctx_tokens)
	p = inter / max(1, len(ctx_tokens))
	r = inter / max(1, len(ref_tokens))
	f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
	return float(p), float(r), float(f1)

def _setup_ragas_llm():
	"""Setup LLM for RAGAS evaluation - supports both OpenAI and Google."""
    
	# Allow explicit provider override
	provider = (os.getenv("RAGAS_LLM_PROVIDER") or "").lower().strip()
	try_openai_first = provider == "openai" or (provider == "" and os.getenv("OPENAI_API_KEY"))
	try_google_next = provider == "google" or (provider == "" and os.getenv("GOOGLE_API_KEY"))

	# Prefer OpenAI for RAGAS (compatibility with evaluation prompts)
	if try_openai_first:
		try:
			from langchain_openai import ChatOpenAI
			openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-nano")
			llm = ChatOpenAI(model=openai_model, temperature=0)
			print(f"[RAGAS LLM] Using OpenAI model: {openai_model}")
			return llm
		except Exception as e:
			print(f"Failed to setup OpenAI LLM for RAGAS: {e}")

	# Fallback to Google Gemini if available
	if try_google_next:
		try:
			from langchain_google_genai import ChatGoogleGenerativeAI
			preferred_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-pro")
			safety_settings = None  # keep default safety behavior; avoid enum incompatibilities
			llm = ChatGoogleGenerativeAI(
				model=preferred_model,
				temperature=0,
				convert_system_message_to_human=True,
				safety_settings=safety_settings,
			)
			print(f"[RAGAS LLM] Using Google Gemini model: {preferred_model}")
			return llm
		except Exception as e:
			print(f"Failed to setup Google LLM for RAGAS: {e}")
	
	return None

def _setup_ragas_embeddings():
	"""Setup embeddings for RAGAS evaluation - supports both OpenAI and Google."""
	
	# Try Google first if API key is available  
	if os.getenv("GOOGLE_API_KEY"):
		try:
			from langchain_google_genai import GoogleGenerativeAIEmbeddings
			
			embeddings = GoogleGenerativeAIEmbeddings(
				model="models/text-embedding-004"
			)
			print("[RAGAS Embeddings] Using Google text-embedding-004")
			return embeddings
		except Exception as e:
			print(f"Failed to setup Google embeddings for RAGAS: {e}")
	
	# Fallback to OpenAI if available
	if os.getenv("OPENAI_API_KEY"):
		try:
			from langchain_openai import OpenAIEmbeddings
			
			embeddings = OpenAIEmbeddings(
				model="text-embedding-3-small"
			)
			print("[RAGAS Embeddings] Using OpenAI text-embedding-3-small")
			return embeddings
		except Exception as e:
			print(f"Failed to setup OpenAI embeddings for RAGAS: {e}")
	
	return None


def run_eval(dataset):
	if evaluate is None:
		raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")
	
	# Use explicitly configured LLM/embeddings to ensure supported models are used
	print("Using RAGAS with configured LLM and embeddings")
	
	# Ensure proper Dataset object to avoid API differences
	ds = dataset
	try:
		if Dataset is not None and not hasattr(dataset, "select"):
			ds = Dataset.from_dict(dataset)  # type: ignore[arg-type]
	except Exception:
		ds = dataset
	
	# Choose metrics based on dataset columns/presence to avoid NaNs
	has_ref = False
	has_gt = False
	try:
		refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
		has_ref = any(bool(r) for r in (refs or []))
	except Exception:
		has_ref = False
	try:
		gts = dataset.get("ground_truths") if isinstance(dataset, dict) else dataset["ground_truths"]
		# treat as present if any non-empty list
		has_gt = any(isinstance(x, list) and len(x) > 0 for x in (gts or []))
	except Exception:
		has_gt = False
	metrics: list = [m for m in (faithfulness, answer_relevancy) if m is not None]
	if has_ref and context_precision is not None:
		metrics.append(context_precision)
	if has_gt and context_recall is not None:
		metrics.append(context_recall)
	
	if not metrics:
		raise RuntimeError("No RAGAS metrics available")
	
	# Run evaluation with configured LLM/embeddings
	llm = _setup_ragas_llm()
	emb = _setup_ragas_embeddings()
	run_config = None
	try:
		if RunConfig is not None:
			run_config = RunConfig()  # type: ignore[call-arg]
	except Exception:
		run_config = None
	try:
		if run_config is not None:
			result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
		else:
			result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
	except TypeError:
		# Fallback for older RAGAS versions
		try:
			if run_config is not None:
				result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
			else:
				result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
		except Exception as e:
			raise RuntimeError(f"RAGAS evaluation failed: {e}") from e
	# Build summary robustly from per-question outputs
	def _mean_safe(vals):
		vals2 = []
		for v in vals:
			try:
				fv = float(v)
				if not (isinstance(fv, float) and (fv != fv)):
					vals2.append(fv)
			except Exception:
				pass
		return float(sum(vals2)/len(vals2)) if vals2 else float("nan")

	faith = relev = cprec = crec = float("nan")
	# Preferred: use to_pandas per-question and average
	try:
		if hasattr(result, "to_pandas"):
			df = result.to_pandas()  # type: ignore
			faith = _mean_safe(df.get("faithfulness", []))
			relev = _mean_safe(df.get("answer_relevancy", []))
			cprec = _mean_safe(df.get("context_precision", []))
			crec = _mean_safe(df.get("context_recall", []))
			out = {
				"faithfulness": faith,
				"answer_relevancy": relev,
				"context_precision": cprec,
				"context_recall": crec,
			}
			# Add heuristic overlap metrics if reference/contexts present and ragas metrics are NaN
			try:
				refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
				ctxs = dataset.get("contexts") if isinstance(dataset, dict) else dataset["contexts"]
			except Exception:
				refs, ctxs = [], []
			try:
				if refs and ctxs:
					p_list, r_list, f1_list = [], [], []
					for ref, ctx in zip(refs, ctxs):
						p, r, f1v = overlap_prf1(ref or "", list(ctx or []))
						p_list.append(p); r_list.append(r); f1_list.append(f1v)
					out["overlap_precision"] = _mean_safe(p_list)
					out["overlap_recall"] = _mean_safe(r_list)
					out["overlap_f1"] = _mean_safe(f1_list)
			except Exception:
				pass
			return out
	except Exception:
		pass
	# Next: try dict-like summary
	try:
		raw = dict(result)  # type: ignore
		as_dict = {str(k): raw[k] for k in raw.keys()}
		faith = float(as_dict["faithfulness"]) if "faithfulness" in as_dict else float("nan")
		relev = float(as_dict["answer_relevancy"]) if "answer_relevancy" in as_dict else float("nan")
		cprec = float(as_dict["context_precision"]) if "context_precision" in as_dict else float("nan")
		crec = float(as_dict["context_recall"]) if "context_recall" in as_dict else float("nan")
		out = {
			"faithfulness": faith,
			"answer_relevancy": relev,
			"context_precision": cprec,
			"context_recall": crec,
		}
		# Add heuristic overlap metrics
		try:
			refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
			ctxs = dataset.get("contexts") if isinstance(dataset, dict) else dataset["contexts"]
			if refs and ctxs:
				p_list, r_list, f1_list = [], [], []
				for ref, ctx in zip(refs, ctxs):
					p, r, f1v = overlap_prf1(ref or "", list(ctx or []))
					p_list.append(p); r_list.append(r); f1_list.append(f1v)
				out["overlap_precision"] = _mean_safe(p_list)
				out["overlap_recall"] = _mean_safe(r_list)
				out["overlap_f1"] = _mean_safe(f1_list)
		except Exception:
			pass
		return out
	except Exception:
		pass
	# Finally: handle .results list shape
	try:
		items = list(getattr(result, "results") or [])
		faith = _mean_safe([r.get("faithfulness") for r in items])
		relev = _mean_safe([r.get("answer_relevancy") for r in items])
		cprec = _mean_safe([r.get("context_precision") for r in items])
		crec = _mean_safe([r.get("context_recall") for r in items])
		return {
			"faithfulness": faith,
			"answer_relevancy": relev,
			"context_precision": cprec,
			"context_recall": crec,
		}
	except Exception:
		return {
			"faithfulness": float("nan"),
			"answer_relevancy": float("nan"),
			"context_precision": float("nan"),
			"context_recall": float("nan"),
			"overlap_precision": float("nan"),
			"overlap_recall": float("nan"),
			"overlap_f1": float("nan"),
		}


def run_eval_detailed(dataset):
	"""Run RAGAS and return (summary_metrics, per_question) where per_question is a
	list of dicts including metrics per item when available. Falls back to empty list otherwise.
	"""
	if evaluate is None:
		raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")
	# Ensure Dataset object
	ds = dataset
	try:
		if Dataset is not None and not hasattr(dataset, "select"):
			ds = Dataset.from_dict(dataset)  # type: ignore[arg-type]
	except Exception:
		ds = dataset
	# Choose metrics dynamically based on columns/presence
	has_ref = False
	has_gt = False
	try:
		refs = dataset.get("reference") if isinstance(dataset, dict) else dataset["reference"]
		has_ref = any(bool(r) for r in (refs or []))
	except Exception:
		has_ref = False
	try:
		gts = dataset.get("ground_truths") if isinstance(dataset, dict) else dataset["ground_truths"]
		has_gt = any(isinstance(x, list) and len(x) > 0 for x in (gts or []))
	except Exception:
		has_gt = False
	metrics: list = [m for m in (faithfulness, answer_relevancy) if m is not None]
	if has_ref and context_precision is not None:
		metrics.append(context_precision)
	if has_gt and context_recall is not None:
		metrics.append(context_recall)
	if not metrics:
		raise RuntimeError("No RAGAS metrics available")
	# Evaluate
	llm = _setup_ragas_llm()
	emb = _setup_ragas_embeddings()
	run_config = None
	try:
		if RunConfig is not None:
			run_config = RunConfig()  # type: ignore[call-arg]
	except Exception:
		run_config = None
	try:
		if run_config is not None:
			result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
		else:
			result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
	except TypeError:
		# older versions may not accept named args
		try:
			if run_config is not None:
				result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)  # type: ignore
			else:
				result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)  # type: ignore
		except Exception:
			result = evaluate(ds, metrics=metrics)  # type: ignore
	# Per-question extraction aligned with original inputs
	per_q = []
	def _maybe_float(x):
		try:
			return float(x)
		except Exception:
			return None
	try:
		# Build a view of original inputs for alignment
		def _get_col(key):
			if isinstance(ds, dict):
				return ds.get(key, [])
			try:
				return ds[key]  # datasets style
			except Exception:
				return []
		q_list = list(_get_col("question"))
		a_list = list(_get_col("answer"))
		r_list = list(_get_col("reference"))
		n = max(len(q_list), len(a_list), len(r_list))

		def _pick(series_like, names):
			# pandas Series or dict-like getter
			for name in names:
				try:
					v = series_like.get(name)
					if v is not None:
						return v
				except Exception:
					pass
			# try fuzzy by lower contains
			try:
				keys = list(getattr(series_like, 'index', [])) or list(getattr(series_like, 'keys')())
			except Exception:
				keys = []
			lower = {str(k).lower(): k for k in keys}
			for target in names:
				t = str(target).lower()
				for lk, orig in lower.items():
					if t in lk:
						try:
							return series_like.get(orig)
						except Exception:
							continue
			return None

		if hasattr(result, "to_pandas"):
			df = result.to_pandas()  # type: ignore
			m = min(n, getattr(df, "shape", [0])[0])
			for i in range(m):
				row = df.iloc[i]
				rec = {
					"question": q_list[i] if i < len(q_list) else None,
					"answer": a_list[i] if i < len(a_list) else None,
					"reference": r_list[i] if i < len(r_list) else None,
					"faithfulness": _maybe_float(_pick(row, ["faithfulness", "faithfulness_score", "faith"])),
					"answer_relevancy": _maybe_float(_pick(row, ["answer_relevancy", "relevancy", "answer_rel"])) ,
					"context_precision": _maybe_float(_pick(row, ["context_precision", "ctx_precision", "precision"])),
					"context_recall": _maybe_float(_pick(row, ["context_recall", "ctx_recall", "recall"])),
				}
				# Add overlap metrics per-question
				try:
					ref = rec.get("reference") or ""
					# Use original contexts for this index
					ctxs = _get_col("contexts")[i] if i < len(_get_col("contexts")) else []
					p, r, f1v = overlap_prf1(ref, list(ctxs or []))
					rec["overlap_precision"], rec["overlap_recall"], rec["overlap_f1"] = p, r, f1v
				except Exception:
					pass
				per_q.append(rec)
		elif hasattr(result, "results"):
			items = list(getattr(result, "results") or [])
			m = min(n, len(items))
			for i in range(m):
				item = items[i]
				rec = {
					"question": q_list[i] if i < len(q_list) else None,
					"answer": a_list[i] if i < len(a_list) else None,
					"reference": r_list[i] if i < len(r_list) else None,
					"faithfulness": _maybe_float(_pick(item, ["faithfulness", "faithfulness_score", "faith"])),
					"answer_relevancy": _maybe_float(_pick(item, ["answer_relevancy", "relevancy", "answer_rel"])) ,
					"context_precision": _maybe_float(_pick(item, ["context_precision", "ctx_precision", "precision"])),
					"context_recall": _maybe_float(_pick(item, ["context_recall", "ctx_recall", "recall"])),
				}
				try:
					ref = rec.get("reference") or ""
					ctxs = _get_col("contexts")[i] if i < len(_get_col("contexts")) else []
					p, r, f1v = overlap_prf1(ref, list(ctxs or []))
					rec["overlap_precision"], rec["overlap_recall"], rec["overlap_f1"] = p, r, f1v
				except Exception:
					pass
				per_q.append(rec)
		# If results shorter than dataset, pad remaining with None metrics
		for i in range(len(per_q), n):
			per_q.append({
				"question": q_list[i] if i < len(q_list) else None,
				"answer": a_list[i] if i < len(a_list) else None,
				"reference": r_list[i] if i < len(r_list) else None,
				"faithfulness": None,
				"answer_relevancy": None,
				"context_precision": None,
				"context_recall": None,
			})
	except Exception:
		per_q = []

	# Compute summary from per-question metrics as a robust fallback
	def _mean_safe(values):
		vals = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
		return float(sum(vals) / len(vals)) if vals else None

	summary = {
		"faithfulness": _mean_safe([r.get("faithfulness") for r in per_q]),
		"answer_relevancy": _mean_safe([r.get("answer_relevancy") for r in per_q]),
		"context_precision": _mean_safe([r.get("context_precision") for r in per_q]),
		"context_recall": _mean_safe([r.get("context_recall") for r in per_q]),
	}
	# Compute overlap metrics summary
	try:
		summary["overlap_precision"] = _mean_safe([r.get("overlap_precision") for r in per_q])
		summary["overlap_recall"] = _mean_safe([r.get("overlap_recall") for r in per_q])
		summary["overlap_f1"] = _mean_safe([r.get("overlap_f1") for r in per_q])
	except Exception:
		pass

	return summary, per_q


TARGETS = {
	"context_precision": 0.75,
	"context_recall": 0.70,
	"faithfulness": 0.85,
	"table_qa_accuracy": 0.90,
}


def pretty_metrics(m: dict) -> str:
	def _fmt(x):
		try:
			import math as _m
			if x is None:
				return "n/a"
			if isinstance(x, float) and (_m.isnan(x) or _m.isinf(x)):
				return "n/a"
			if isinstance(x, (int, float)):
				return f"{x:.3f}"
			return str(x)
		except Exception:
			return str(x)
	lines = [
		f"Faithfulness: {_fmt(m.get('faithfulness'))}",
		f"Answer relevancy: {_fmt(m.get('answer_relevancy'))}",
		f"Context precision: {_fmt(m.get('context_precision'))}",
		f"Context recall: {_fmt(m.get('context_recall'))}",
	]
	# Optional heuristic overlap metrics (no LLM/embeddings required)
	if any(k in m for k in ("overlap_precision", "overlap_recall", "overlap_f1")):
		lines.append(f"Overlap precision (heuristic): {_fmt(m.get('overlap_precision'))}")
		lines.append(f"Overlap recall (heuristic): {_fmt(m.get('overlap_recall'))}")
		lines.append(f"Overlap F1 (heuristic): {_fmt(m.get('overlap_f1'))}")
	return "\n".join(lines)

