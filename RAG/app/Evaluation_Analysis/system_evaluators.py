#!/usr/bin/env python3
"""
System Evaluators
================

RAG system evaluation functions.
"""

import json
from pathlib import Path
from typing import Dict, Any

from RAG.app.logger import get_logger
from .answer_evaluators import calculate_answer_correctness


def evaluate_rag_system(ground_truth_path: Path, pipeline) -> Dict[str, Any]:
	"""Evaluate RAG system using ground truth data."""
	log = get_logger()
	
	# Load ground truth data
	if not ground_truth_path.exists():
		raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
	
	with open(ground_truth_path, 'r', encoding='utf-8') as f:
		ground_truth_data = json.load(f)
	
	results = []
	total_score = 0
	
	for item in ground_truth_data:
		question = item.get("question", "")
		expected_answer = item.get("answer", "")
		
		if not question or not expected_answer:
			continue
		
		try:
			# Get answer from pipeline
			response = pipeline.query(question)
			actual_answer = response.get("answer", "")
			
			# Evaluate answer
			correctness = calculate_answer_correctness(actual_answer, expected_answer)
			
			result = {
				"question": question,
				"expected": expected_answer,
				"actual": actual_answer,
				"correctness": correctness
			}
			
			results.append(result)
			total_score += correctness
			
		except Exception as e:
			log.error(f"Error evaluating question '{question}': {e}")
			continue
	
	# Calculate overall metrics
	avg_score = total_score / len(results) if results else 0
	
	return {
		"total_questions": len(results),
		"average_correctness": avg_score,
		"results": results
	}
