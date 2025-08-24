from typing import List, Protocol

from langchain.schema import Document

from app.prompts import (
	NEEDLE_PROMPT,
	NEEDLE_SYSTEM,
	SUMMARY_PROMPT,
	SUMMARY_SYSTEM,
	TABLE_PROMPT,
	TABLE_SYSTEM,
)

class LLMCallable(Protocol):
	def __call__(self, prompt: str) -> str:  # noqa: D401
		...


def route_question(q: str) -> str:
	ql = q.lower()
	if any(w in ql for w in ("summarize", "overview", "overall", "conclusion", "brief")):
		return "summary"
	# Route to hierarchical table handler for data questions
	if any(w in ql for w in [
		"table", "chart", "value", "figure", "fig ", "image", "graph", "plot",
		"wear depth", "module", "ratio", "teeth", "lubricant", "rms", "vibration",
		"recording", "record", "day", "date", "measurement", "data"
	]):
		return "table"
	return "needle"


def identify_csv_targets(question: str) -> List[str]:
	"""
	Identify which CSV files should be searched based on question content.
	Returns a list of CSV filenames to prioritize in search.
	"""
	ql = question.lower()
	targets = []
	
	# Wear depth and tooth analysis questions
	if any(w in ql for w in ["wear depth", "wear case", "tooth", "teeth", "wear measurement"]):
		targets.extend([
			"all_teeth_results.csv",
			"single_tooth_results.csv", 
			"ground_truth_measurements.csv"
		])
	
	# Vibration analysis questions
	if any(w in ql for w in ["rms", "vibration", "fme", "frequency", "amplitude"]):
		targets.extend([
			"RMS15.csv",
			"RMS45.csv",
			"FME Values.csv"
		])
	
	# Recording and data collection questions
	if any(w in ql for w in ["recording", "record", "day", "date", "time", "measurement", "data collection"]):
		targets.append("Records.csv")
	
	# Gear specifications - transmission ratio is likely in the main report, not CSV files
	if any(w in ql for w in ["module", "gear type", "lubricant"]):
		# These might be in the main report, but also check CSV files
		targets.extend([
			"all_teeth_results.csv",
			"single_tooth_results.csv"
		])
	
	# Transmission ratio - this is typically in the main report, not CSV files
	if any(w in ql for w in ["transmission ratio", "ratio", "gear ratio"]):
		# Don't add CSV targets for transmission ratio questions
		# They should be found in the main report
		pass
	
	return list(set(targets))


def _matches_csv_target(file_name: str, csv_targets: List[str]) -> bool:
	"""
	Check if a file name matches any of the CSV targets.
	Handles both full paths and just filenames.
	"""
	if not file_name:
		return False
	
	# Extract just the filename from the path
	import os
	filename = os.path.basename(file_name)
	
	return filename in csv_targets


def render_context(docs: List[Document], max_chars: int = 8000) -> str:
	out, n = [], 0
	for d in docs:
		# Remove document path information from context
		piece = d.page_content.strip()
		n += len(piece)
		if n > max_chars:
			break
		out.append(piece)
	return "\n\n".join(out)


def answer_summary(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = SUMMARY_SYSTEM + "\n" + SUMMARY_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


def answer_needle(llm: LLMCallable, docs: List[Document], question: str) -> str:
	ctx = render_context(docs)
	prompt = NEEDLE_SYSTEM + "\n" + NEEDLE_PROMPT.format(context=ctx, question=question)
	return llm(prompt).strip()


def answer_table(llm: LLMCallable, docs: List[Document], question: str, hybrid_retriever=None) -> str:
	"""
	Hierarchical search strategy:
	1. First try to find answer in main report (PDF)
	2. If not found, search specific CSV files based on question type
	"""
	# Identify which CSV files to prioritize based on question
	csv_targets = identify_csv_targets(question)
	
	# Strategy 1: Try PDF first (main report)
	pdf_docs = [d for d in docs if "Gear wear Failure.pdf" in d.metadata.get("file_name", "")]
	if pdf_docs:
		ctx = render_context(pdf_docs)
		prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
		answer = llm(prompt).strip()
		
		# Check if we got a meaningful answer (not "not found" or similar)
		if not any(phrase in answer.lower() for phrase in ["not found", "not provided", "cannot find", "not available", "no information", "not mentioned"]):
			return answer
	
	# Strategy 2: If we have a hybrid retriever, search specifically in CSV files
	if hybrid_retriever and csv_targets:
		try:
			# Create a modified query to target CSV files
			csv_query = f"{question} [search in CSV data files: {', '.join(csv_targets)}]"
			csv_candidates = hybrid_retriever.get_relevant_documents(csv_query)
			
			# Filter to only CSV files we're interested in
			csv_docs = [d for d in csv_candidates if _matches_csv_target(d.metadata.get("file_name"), csv_targets)]
			
			if csv_docs:
				ctx = render_context(csv_docs)
				prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
				answer = llm(prompt).strip()
				
				# Check if we got a meaningful answer
				if not any(phrase in answer.lower() for phrase in ["not found", "not provided", "cannot find", "not available", "no information", "not mentioned"]):
					return answer
		except Exception as e:
			# Fallback to searching in existing docs
			pass
	
	# Strategy 3: Search in CSV documents from the original retrieval
	csv_docs = [d for d in docs if _matches_csv_target(d.metadata.get("file_name"), csv_targets)]
	if csv_docs:
		ctx = render_context(csv_docs)
		prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
		answer = llm(prompt).strip()
		
		# Check if we got a meaningful answer
		if not any(phrase in answer.lower() for phrase in ["not found", "not provided", "cannot find", "not available", "no information", "not mentioned"]):
			return answer
	
	# Strategy 4: Search all remaining documents
	all_remaining = [d for d in docs if d not in pdf_docs and d not in csv_docs]
	if all_remaining:
		ctx = render_context(all_remaining)
		prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
		return llm(prompt).strip()
	
	# Fallback
	return "I could not find the requested information in the available data sources."
