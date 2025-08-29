#!/usr/bin/env python3
"""
Answer Evaluators
================

Answer correctness evaluation functions.
"""

import re
from typing import Dict, Any, Tuple
from difflib import SequenceMatcher

from RAG.app.config import settings
from .text_processors import (
    extract_numbers,
    extract_dates,
    extract_technical_terms,
    extract_citations,
    simple_similarity
)


def calculate_answer_correctness(answer: str, ground_truth: str) -> float:
	"""Calculate answer correctness by comparing against ground truth.
	Uses multiple similarity metrics for robust evaluation.
	"""
	if not answer or not ground_truth:
		return 0.0
	
	# Normalize text for comparison
	def normalize_text(text: str) -> Tuple[str, str]:
		# Remove extra whitespace and convert to lowercase
		text = re.sub(r'\s+', ' ', text.lower().strip())
		# Remove punctuation for some comparisons
		text_no_punct = re.sub(r'[^\w\s]', '', text)
		return text, text_no_punct
	
	answer_norm, answer_no_punct = normalize_text(answer)
	gt_norm, gt_no_punct = normalize_text(ground_truth)
	
	# Multiple similarity metrics
	scores = []
	
	# 1. Sequence matcher (overall similarity)
	seq_similarity = SequenceMatcher(None, answer_norm, gt_norm).ratio()
	scores.append(seq_similarity)
	
	# 2. Token overlap (word-level similarity)
	answer_tokens = set(answer_norm.split())
	gt_tokens = set(gt_norm.split())
	if gt_tokens:
		token_overlap = len(answer_tokens & gt_tokens) / len(gt_tokens)
		scores.append(token_overlap)
	
	# 3. Exact match bonus
	exact_match = 1.0 if answer_norm == gt_norm else 0.0
	scores.append(exact_match)
	
	# 4. Number extraction and comparison
	answer_numbers = extract_numbers(answer)
	gt_numbers = extract_numbers(ground_truth)
	
	if gt_numbers:
		# Compare numbers (allow for small differences)
		number_matches = 0
		for gt_num in gt_numbers:
			for ans_num in answer_numbers:
				if abs(gt_num - ans_num) < 0.01:  # 1% tolerance
					number_matches += 1
					break
		number_similarity = number_matches / len(gt_numbers)
		scores.append(number_similarity)
	
	# 5. Date extraction and comparison
	answer_dates = extract_dates(answer)
	gt_dates = extract_dates(ground_truth)
	
	if gt_dates:
		date_matches = len(set(answer_dates) & set(gt_dates))
		date_similarity = date_matches / len(gt_dates)
		scores.append(date_similarity)
	
	# 6. Technical terms comparison
	answer_terms = extract_technical_terms(answer)
	gt_terms = extract_technical_terms(ground_truth)
	
	if gt_terms:
		term_matches = len(answer_terms & gt_terms)
		term_similarity = term_matches / len(gt_terms)
		scores.append(term_similarity)
	
	# 7. Citation accuracy
	answer_citations = extract_citations(answer)
	gt_citations = extract_citations(ground_truth)
	
	if gt_citations:
		citation_matches = len(answer_citations & gt_citations)
		citation_similarity = citation_matches / len(gt_citations)
		scores.append(citation_similarity)
	
	# Calculate weighted average using configurable weights
	if scores:
		# Use configurable weights from centralized settings
		weights = settings.evaluation.ANSWER_CORRECTNESS_WEIGHT_LIST[:len(scores)]  # Truncate if fewer scores
		# Normalize weights
		total_weight = sum(weights)
		weights = [w/total_weight for w in weights]
		
		weighted_score = sum(s * w for s, w in zip(scores, weights))
		return min(weighted_score, 1.0)  # Cap at 1.0
	
	return 0.0


def evaluate_answer_simple(question: str, ground_truth: str, rag_answer: str) -> Dict[str, Any]:
	"""Simple evaluation of a single answer against ground truth."""
	
	# Basic similarity
	text_similarity = simple_similarity(ground_truth, rag_answer)
	
	# Check if key numbers match (for numerical questions)
	gt_numbers = set(extract_numbers(ground_truth))
	ra_numbers = set(extract_numbers(rag_answer))
	number_accuracy = len(gt_numbers & ra_numbers) / len(gt_numbers) if gt_numbers else 1.0
	
	# Check if key terms are present
	gt_words = set(ground_truth.lower().split())
	ra_words = set(rag_answer.lower().split())
	key_term_coverage = len(gt_words & ra_words) / len(gt_words) if gt_words else 1.0
	
	# Overall score (weighted combination)
	overall_score = (text_similarity * 0.4 + number_accuracy * 0.3 + key_term_coverage * 0.3)
	
	return {
		"text_similarity": text_similarity,
		"number_accuracy": number_accuracy,
		"key_term_coverage": key_term_coverage,
		"overall_score": overall_score,
		"gt_numbers": list(gt_numbers),
		"ra_numbers": list(ra_numbers),
		"missing_numbers": list(gt_numbers - ra_numbers),
		"extra_numbers": list(ra_numbers - gt_numbers)
	}
