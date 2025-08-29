#!/usr/bin/env python3
"""
Table-QA Evaluators
==================

Table-QA specific evaluation functions.
"""

from typing import List

from .answer_evaluators import calculate_answer_correctness


def calculate_table_qa_accuracy(questions: List[str], answers: List[str], ground_truths: List[List[str]]) -> float:
	"""
	Calculate Table-QA specific accuracy for questions that involve table data.
	
	Args:
		questions: List of questions
		answers: List of RAG answers
		ground_truths: List of ground truth answers
		
	Returns:
		float: Table-QA accuracy score (0.0 to 1.0)
	"""
	table_keywords = [
		"table", "wear depth", "case w", "transmission", "sensors", 
		"accelerometer", "tachometer", "sensitivity", "module", "ratio"
	]
	
	table_qa_pairs = []
	
	for i, question in enumerate(questions):
		# Check if question involves table data
		is_table_question = any(keyword in question.lower() for keyword in table_keywords)
		
		if is_table_question:
			rag_answer = answers[i]
			ground_truth = ground_truths[i][0] if ground_truths[i] else ""
			
			# Calculate accuracy for this table question
			accuracy = calculate_answer_correctness(rag_answer, ground_truth)
			table_qa_pairs.append(accuracy)
	
	if not table_qa_pairs:
		return 0.0
	
	return sum(table_qa_pairs) / len(table_qa_pairs)
