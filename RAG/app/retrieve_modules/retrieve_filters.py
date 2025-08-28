import os
import re
from typing import Dict, List
from langchain.schema import Document
from RAG.app.logger import get_logger


def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
	if not filters:
		return docs
	def ok(meta: dict):
		for k, v in (filters or {}).items():
			if k == "section":
				# Enhanced section matching with fuzzy logic
				meta_section = meta.get("section") or meta.get("section_type")
				if meta_section != v:
					# Try fuzzy matching for similar sections
					if v == "Table" and "table" in str(meta_section).lower():
						continue
					elif v == "Figure" and any(x in str(meta_section).lower() for x in ["figure", "image", "fig"]):
						continue
					return False
			elif k == "figure_number":
				# Enhanced figure number matching
				mn = meta.get("figure_number")
				if str(mn) == str(v):
					continue
				# Check various metadata fields for figure references
				label = str(meta.get("figure_label") or meta.get("caption") or meta.get("title") or "")
				if not re.match(rf"^\s*figure\s*{int(str(v))}\b", label, re.I):
					# Also check content for figure references
					content = str(meta.get("page_content") or "")
					if not re.search(rf"figure\s*{int(str(v))}", content, re.I):
						return False
			elif k == "table_number":
				# Enhanced table number matching
				mn = meta.get("table_number")
				if str(mn) == str(v):
					continue
				label = str(meta.get("table_label") or meta.get("caption") or meta.get("title") or "")
				if not re.match(rf"^\s*table\s*{int(str(v))}\b", label, re.I):
					# Also check content for table references
					content = str(meta.get("page_content") or "")
					if not re.search(rf"table\s*{int(str(v))}", content, re.I):
						return False
			elif k == "case_id":
				# Enhanced case ID matching with fuzzy logic
				meta_case = str(meta.get(k) or "")
				if meta_case.lower() == str(v).lower():
					continue
				# Check content for case ID references
				content = str(meta.get("page_content") or "")
				if str(v).lower() in content.lower():
					continue
				return False
			elif k == "client_id":
				# Enhanced client ID matching
				meta_client = str(meta.get(k) or "")
				if meta_client.lower() == str(v).lower():
					continue
				# Check content for client ID references
				content = str(meta.get("page_content") or "")
				if str(v).lower() in content.lower():
					continue
				return False
			elif k == "incident_date":
				# Enhanced date matching
				meta_date = str(meta.get(k) or "")
				if meta_date == str(v):
					continue
				# Check content for date references
				content = str(meta.get("page_content") or "")
				if str(v) in content:
					continue
				return False
			else:
				# Default exact matching for other fields
				if meta.get(k) != v:
					return False
		return True
	
	# Apply filters
	out = [d for d in docs if ok(d.metadata)]
	
	# Enhanced logging with more details
	try:
		if os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1", "true", "yes"):
			log = get_logger()
			log.debug("Filtered %d docs -> %d docs", len(docs), len(out))
	except Exception:
		pass
	
	return out
