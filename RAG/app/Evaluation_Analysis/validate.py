from typing import Tuple, List, Optional
from langchain.schema import Document
from RAG.app.logger import get_logger
from RAG.app.config import settings


def validate_min_pages(num_pages: int, min_pages: Optional[int] = None) -> Tuple[bool, str]:
	"""
	Validate that the number of pages meets minimum requirements.
	
	Args:
		num_pages: Number of pages in the document
		min_pages: Minimum required pages (defaults to config setting)
		
	Returns:
		Tuple[bool, str]: (is_valid, message)
	"""
	if min_pages is None:
		min_pages = settings.chunking.MIN_PAGES
	
	if num_pages < min_pages:
		return False, f"Document has {num_pages} pages; requires >= {min_pages}."
	return True, "OK"


def validate_chunk_tokens(tok_counts: list[int], avg_range: Optional[Tuple[int, int]] = None, max_tok: Optional[int] = None) -> Tuple[bool, str]:
	"""
	Validate that chunk token counts meet requirements.
	
	Args:
		tok_counts: List of token counts for chunks
		avg_range: Acceptable range for average tokens (defaults to config setting)
		max_tok: Maximum tokens per chunk (defaults to config setting)
		
	Returns:
		Tuple[bool, str]: (is_valid, message)
	"""
	if avg_range is None:
		avg_range = settings.chunking.CHUNK_TOK_AVG_RANGE
	if max_tok is None:
		max_tok = settings.chunking.CHUNK_TOK_MAX
	
	avg = sum(tok_counts) / max(1, len(tok_counts))
	if not (avg_range[0] <= avg <= avg_range[1]):
		return False, f"Avg tokens {avg:.1f} not in {avg_range}."
	if any(t > max_tok for t in tok_counts):
		return False, f"One or more chunks exceed max {max_tok} tokens."
	return True, "OK"


def validate_document_pages(documents: List[Document], min_pages: Optional[int] = None) -> bool:
	"""
	Validate that documents meet minimum page requirements.
	
	Args:
		documents: List of documents to validate
		min_pages: Minimum required pages (defaults to config setting)
		
	Returns:
		bool: True if validation passes, False otherwise
	"""
	log = get_logger()
	
	# Count unique pages across all documents
	unique_pages = set()
	for doc in documents:
		if hasattr(doc, 'metadata') and doc.metadata:
			page = doc.metadata.get('page', 1)
			if isinstance(page, int):
				unique_pages.add(page)
	
	num_pages = len(unique_pages) if unique_pages else 1
	
	try:
		is_valid, message = validate_min_pages(num_pages, min_pages)
		if is_valid:
			log.info(f"Document validation passed: {num_pages} pages")
			return True
		else:
			log.error(f"Document validation failed: {message}")
			return False
	except Exception as e:
		log.error(f"Document validation failed: {e}")
		return False


def validate_documents(documents: List[Document], min_pages: Optional[int] = None) -> bool:
	"""
	Validate that documents meet minimum requirements.
	This is the main validation function that can be used by services.
	
	Args:
		documents: List of documents to validate
		min_pages: Minimum required pages (defaults to config setting)
		
	Returns:
		bool: True if validation passes, False otherwise
	"""
	if not documents:
		return False
	
	# Validate page requirements
	if not validate_document_pages(documents, min_pages):
		return False
	
	# Additional validations can be added here
	# For example: validate document content, metadata, etc.
	
	return True

