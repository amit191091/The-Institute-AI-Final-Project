try:
	from ragas import evaluate
	from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except Exception:  # pragma: no cover
	evaluate = None
	faithfulness = answer_relevancy = context_precision = context_recall = None

try:
	from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover
	Dataset = None  # type: ignore


def run_eval(dataset):
	if evaluate is None:
		raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")
	# Ensure proper Dataset object to avoid API differences
	ds = dataset
	try:
		if Dataset is not None and not hasattr(dataset, "select"):
			ds = Dataset.from_dict(dataset)  # type: ignore[arg-type]
	except Exception:
		ds = dataset
	metrics = [m for m in (faithfulness, answer_relevancy, context_precision, context_recall) if m is not None]
	# Some versions of ragas expect positional argument; others accept keywords
	try:
		result = evaluate(dataset=ds, metrics=metrics)  # type: ignore
	except TypeError:
		result = evaluate(ds, metrics=metrics)  # type: ignore
	# RAGAS returns an EvaluationResult with dict-like access in most versions
	try:
		as_dict = {str(k): float(v) for k, v in dict(result).items()}  # type: ignore
		faith = as_dict.get("faithfulness", float("nan"))
		relev = as_dict.get("answer_relevancy", float("nan"))
		cprec = as_dict.get("context_precision", float("nan"))
		crec = as_dict.get("context_recall", float("nan"))
	except Exception:
		try:
			faith = float(result["faithfulness"])  # type: ignore[index]
		except Exception:
			faith = float("nan")
		try:
			relev = float(result["answer_relevancy"])  # type: ignore[index]
		except Exception:
			relev = float("nan")
		try:
			cprec = float(result["context_precision"])  # type: ignore[index]
		except Exception:
			cprec = float("nan")
		try:
			crec = float(result["context_recall"])  # type: ignore[index]
		except Exception:
			crec = float("nan")
	return {
		"faithfulness": faith,
		"answer_relevancy": relev,
		"context_precision": cprec,
		"context_recall": crec,
	}


TARGETS = {
	"context_precision": 0.75,
	"context_recall": 0.70,
	"faithfulness": 0.85,
	"table_qa_accuracy": 0.90,
}


def pretty_metrics(m: dict) -> str:
	lines = [
		f"Faithfulness: {m.get('faithfulness', 'n/a')}",
		f"Answer relevancy: {m.get('answer_relevancy', 'n/a')}",
		f"Context precision: {m.get('context_precision', 'n/a')}",
		f"Context recall: {m.get('context_recall', 'n/a')}",
	]
	return "\n".join(lines)

