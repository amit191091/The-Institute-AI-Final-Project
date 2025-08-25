from typing import Tuple


def validate_min_pages(num_pages: int, min_pages: int = 10) -> Tuple[bool, str]:
	if num_pages < min_pages:
		return False, f"Document has {num_pages} pages; requires >= {min_pages}."
	return True, "OK"


def validate_chunk_tokens(tok_counts: list[int], avg_range=(250, 500), max_tok=800) -> Tuple[bool, str]:
	avg = sum(tok_counts) / max(1, len(tok_counts))
	if not (avg_range[0] <= avg <= avg_range[1]):
		return False, f"Avg tokens {avg:.1f} not in {avg_range}."
	if any(t > max_tok for t in tok_counts):
		return False, f"One or more chunks exceed max {max_tok} tokens."
	return True, "OK"

