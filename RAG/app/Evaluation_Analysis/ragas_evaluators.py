#!/usr/bin/env python3
"""
RAGAS Evaluators
================

RAGAS evaluation functions.
"""

import os
from typing import Dict, Any, List

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

try:
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore

from .ragas_setup import _setup_ragas_llm, _setup_ragas_embeddings
from .table_qa_evaluators import calculate_table_qa_accuracy


def run_eval(dataset):
	"""Run RAGAS evaluation on dataset."""
	if evaluate is None:
		raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")
	
	# Use explicitly configured LLM/embeddings to ensure supported models are used
	print("Using RAGAS with configured LLM and embeddings")
	
	# Terminal logging for evaluation
	print(f"   Dataset size: {len(dataset.get('question', []))} questions")
	
	# Check environment settings
	force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
	print(f"   FORCE_OPENAI_ONLY: {force_openai}")
	print(f"   GOOGLE_API_KEY available: {bool(os.getenv('GOOGLE_API_KEY'))}")
	print(f"   OPENAI_API_KEY available: {bool(os.getenv('OPENAI_API_KEY'))}")
	
	# Setup LLM and embeddings
	llm = _setup_ragas_llm()
	embeddings = _setup_ragas_embeddings()
	
	if not llm:
		raise RuntimeError("No LLM available for RAGAS evaluation")
	
	if not embeddings:
		raise RuntimeError("No embeddings available for RAGAS evaluation")
	
	# Convert to Dataset format
	if Dataset is not None:
		ds = Dataset.from_dict(dataset)
	else:
		ds = dataset
	
	# Run evaluation
	results = evaluate(
		ds,
		metrics=[
			answer_relevancy,
			context_precision,
			context_recall,
			faithfulness,
		],
		llm=llm,
		embeddings=embeddings,
	)
	
	# Calculate additional custom metrics
	questions = dataset.get("question", [])
	answers = dataset.get("answer", [])
	ground_truths = dataset.get("ground_truths", [])
	
	# Calculate Table-QA accuracy
	table_qa_accuracy = calculate_table_qa_accuracy(questions, answers, ground_truths)
	results["table_qa_accuracy"] = table_qa_accuracy
	
	return results


def run_eval_detailed(dataset, hybrid_retriever=None, llm=None):
	"""Run detailed RAGAS evaluation with per-question results."""
	if evaluate is None:
		raise RuntimeError("ragas not installed. pip install ragas datasets evaluate")
	
	# Setup LLM and embeddings
	if not llm:
		llm = _setup_ragas_llm()
	if not llm:
		raise RuntimeError("No LLM available for RAGAS evaluation")
	
	embeddings = _setup_ragas_embeddings()
	if not embeddings:
		raise RuntimeError("No embeddings available for RAGAS evaluation")
	
	# Convert to Dataset format
	if Dataset is not None:
		ds = Dataset.from_dict(dataset)
	else:
		ds = dataset
	
	# Run evaluation
	results = evaluate(
		ds,
		metrics=[
			answer_relevancy,
			context_precision,
			context_recall,
			faithfulness,
		],
		llm=llm,
		embeddings=embeddings,
	)
	
	# Get per-question results
	per_question = []
	for i, row in enumerate(ds):
		question = row.get("question", "")
		answer = row.get("answer", "")
		contexts = row.get("contexts", [])
		reference = row.get("reference", "")
		
		# Calculate additional metrics
		from .answer_evaluators import calculate_answer_correctness
		correctness = calculate_answer_correctness(answer, reference)
		
		per_question.append({
			"question": question,
			"answer": answer,
			"contexts": contexts,
			"reference": reference,
			"answer_correctness": correctness,
			"answer_relevancy": results["answer_relevancy"][i] if "answer_relevancy" in results else None,
			"context_precision": results["context_precision"][i] if "context_precision" in results else None,
			"context_recall": results["context_recall"][i] if "context_recall" in results else None,
			"faithfulness": results["faithfulness"][i] if "faithfulness" in results else None,
		})
	
	# Calculate Table-QA accuracy for the entire dataset
	questions = dataset.get("question", [])
	answers = dataset.get("answer", [])
	ground_truths = dataset.get("ground_truths", [])
	table_qa_accuracy = calculate_table_qa_accuracy(questions, answers, ground_truths)
	results["table_qa_accuracy"] = table_qa_accuracy
	
	return results, per_question


def pretty_metrics(metrics: Dict[str, Any]) -> str:
	"""Format metrics for pretty printing."""
	if not metrics:
		return "No metrics available"
	
	lines = []
	for metric_name, value in metrics.items():
		if isinstance(value, (int, float)):
			lines.append(f"{metric_name}: {value:.3f}")
		else:
			lines.append(f"{metric_name}: {value}")
	
	return "\n".join(lines)
