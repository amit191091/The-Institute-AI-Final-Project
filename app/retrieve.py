import os
import re
from typing import Dict, List, Optional
from app.logger import get_logger

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
# Optional Cross-Encoder reranker
try:
	from app.reranker_ce import rerank as ce_rerank  # type: ignore
except Exception:  # pragma: no cover
	ce_rerank = None  # type: ignore
from app.logger import trace_func
try:
	from app.query_intent import get_intent  # optional LLM router
except Exception:
	get_intent = None  # type: ignore
try:
	from app.conversation_context import conversation_context  # conversation tracking
except Exception:
	conversation_context = None  # type: ignore


@trace_func
def query_analyzer(q: str) -> Dict:
	"""Extract keywords, case/client IDs, dates to build metadata filters.
	Also returns a 'canonical' simplified query from a rules-based pre-agent.
	Enhanced with document domain detection and conversation context to prevent cross-document contamination.
	"""
	filt: Dict[str, str] = {}
	
	# CONVERSATION CONTEXT: Check for ambiguous queries and get document preferences
	conversation_info = {}
	if conversation_context is not None:
		ambiguity_info = conversation_context.detect_ambiguous_query(q)
		conversation_info = {
			'is_ambiguous': ambiguity_info['is_ambiguous'],
			'preferred_document': ambiguity_info['preferred_document'],
			'available_documents': ambiguity_info['available_documents'],
			'needs_disambiguation': ambiguity_info['is_ambiguous'] and len(ambiguity_info['available_documents']) > 1
		}
		
		# If query is ambiguous but we have conversation context, bias toward preferred document
		bias_document = conversation_context.should_bias_retrieval(q)
		if bias_document:
			# Extract just the core document name (e.g., "Gear wear Failure.pdf" -> "gear")
			if "gear" in bias_document.lower():
				filt["conversation_bias"] = "gear_wear"
			elif "bearing" in bias_document.lower():
				filt["conversation_bias"] = "bearing"
	
	# Use LLM intent if available, otherwise use direct analysis
	if get_intent is not None and (os.getenv("RAG_USE_LLM_ROUTER", "0").lower() in ("1","true","yes")):
		simp = get_intent(q)
	else:
		# Direct analysis without complex simplify_question preprocessing
		ql = q.lower()
		simp = {
			"wants_figure": any(w in ql for w in ("figure", "image", "fig ", "photo", "plot", "graph")),
			"wants_table": "table" in ql,
			"canonical": q.strip(),
		}
		# Extract specific figure/table numbers
		fig_match = re.search(r"figure\s+(\d+)", ql)
		if fig_match:
			simp["figure_number"] = int(fig_match.group(1))
		table_match = re.search(r"table\s+(\d+)", ql) 
		if table_match:
			simp["table_number"] = int(table_match.group(1))
		# Extract case ID patterns (e.g., W26, case 45)
		case_match = re.search(r"\b(?:case\s+)?([wW]\d{1,3}|\d{1,3})\b", q)
		if case_match:
			simp["case_id"] = case_match.group(1)
	
	# DOMAIN DETECTION: Identify document subject to prevent cross-document contamination
	domain_keywords = {
		"gear": ["gear", "gears", "gearbox", "tooth", "teeth", "mesh", "meshing", "pinion", "spur", "transmission", "ratio"],
		"bearing": ["bearing", "bearings", "sliding", "journal", "thrust", "ball", "roller", "race", "cage"],
		"shaft": ["shaft", "shafts", "coupling", "alignment", "balance", "runout"],
		"vibration": ["vibration", "rms", "fme", "gmf", "frequency", "spectral", "spectrum", "amplitude"],
		"lubrication": ["oil", "lubricant", "lubrication", "viscosity", "contamination", "degradation"]
	}
	
	detected_domains = []
	for domain, keywords in domain_keywords.items():
		if any(keyword in ql for keyword in keywords):
			detected_domains.append(domain)
	
	# For gear-related questions, strongly prefer gear wear documents
	if "gear" in detected_domains or any(term in ql for term in ["gear wear", "tooth wear", "flank wear"]):
		filt["primary_subject"] = "gear_wear"  # This will be used to scope retrieval
	elif "bearing" in detected_domains and "gear" not in detected_domains:
		filt["primary_subject"] = "bearing"
	
	# Add conversation and domain info to trace for debugging
	simp["detected_domains"] = detected_domains
	simp["conversation_info"] = conversation_info
	
	# Safer patterns: require word boundaries; avoid matching 'id' inside 'did'
	# Accept 'case: XYZ' or 'case id: XYZ' or 'client: ABC' but not bare 'id'
	# Only capture case id when explicitly labeled with ':' or '-' (avoid 'wear case corresponds')
	mcase = re.search(r"\bcase(?:\s*id)?\s*[:\-]\s*([A-Za-z0-9_-]{2,})", q, re.I)
	if mcase:
		filt["case_id"] = mcase.group(1)
	mclient = re.search(r"\bclient(?:\s*id)?\b[:\-\s]*([A-Za-z0-9_-]{3,})", q, re.I)
	if mclient:
		filt["client_id"] = mclient.group(1)
	mdate = re.search(r"(20\d{2}-\d{2}-\d{2})", q)
	if mdate:
		filt["incident_date"] = mdate.group(1)
	ql = q.lower()
	# Section hints from simplifier or raw tokens
	# Prefer Figure when both table and figure cues appear
	if bool(simp.get("wants_figure")) or any(w in ql for w in ("figure", "image", "fig ", "photo", "plot", "graph")):
		filt["section"] = "Figure"
	elif bool(simp.get("wants_table")) or "table" in ql:
		filt["section"] = "Table"
	# Specific number hints
	if simp.get("table_number"):
		filt["table_number"] = str(simp.get("table_number"))
	if simp.get("figure_number"):
		filt["figure_number"] = str(simp.get("figure_number"))
	# Case id from simplifier (e.g., W26)
	if simp.get("case_id") and "case_id" not in filt:
		filt["case_id"] = str(simp.get("case_id"))
	return {
		"filters": filt,
		"keywords": re.findall(r"[A-Za-z0-9°%]+", q)[:10],
		"canonical": str(simp.get("canonical") or "").strip() or None,
		"intent": simp,  # expose full simplifier intent for downstream routing/augmentation
	}


@trace_func
def apply_filters(docs: List[Document], filters: Dict) -> List[Document]:
	if not filters:
		return docs
	def ok(meta: dict):
		for k, v in (filters or {}).items():
			# New: allow scoping by dataset_id or source_document_id
			if k in ("dataset_id", "source_document_id"):
				if meta.get(k) != v:
					return False
			# CONVERSATION BIAS: Soft preference for documents from conversation context
			if k == "conversation_bias":
				# This is a soft filter - we'll use it for scoring/reranking rather than hard filtering
				continue
			# 'case_id' in our corpus (e.g., W1, W13) typically lives in table cell text, not metadata.
			# Ignoring this hard filter avoids over-pruning to zero and lets reranker/lexical handle it.
			if k == "case_id":
				continue
			if k == "section":
				sec = (meta.get("section") or meta.get("section_type"))
				# Treat TableCell mini-docs as part of Table for filtering purposes
				if v == "Table":
					if sec not in ("Table", "TableCell"):
						return False
				else:
					if sec != v:
						return False
			elif k == "figure_number":
				# Support int/str and fallback to label prefix
				mn = meta.get("figure_number")
				if str(mn) == str(v):
					continue
				label = str(meta.get("figure_label") or meta.get("caption") or "")
				import re as _re
				if not _re.match(rf"^\s*figure\s*{int(str(v))}\b", label, _re.I):
					return False
			elif k == "table_number":
				mn = meta.get("table_number")
				if str(mn) == str(v):
					continue
				label = str(meta.get("table_label") or "")
				import re as _re
				if not _re.match(rf"^\s*table\s*{int(str(v))}\b", label, _re.I):
					return False
			else:
				if meta.get(k) != v:
					return False
		return True
	out = [d for d in docs if ok(d.metadata)]
	try:
		if os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1", "true", "yes"):
			log = get_logger()
			log.debug("FILTER: %d -> %d using %s", len(docs), len(out), filters)
	except Exception:
		pass
	# Soft fallback: if strict filters over-prune, relax 'section' first, then explicit numbers
	if not out and filters:
		try:
			relaxed = dict(filters)
			changed = False
			if "section" in relaxed:
				relaxed.pop("section", None)
				changed = True
			# Re-evaluate with relaxed filters (no section)
			if changed:
				def ok_rel(meta: dict):
					for k, v in (relaxed or {}).items():
						if k == "case_id":
							continue
						if k == "figure_number":
							mn = meta.get("figure_number")
							if str(mn) == str(v):
								continue
							label = str(meta.get("figure_label") or meta.get("caption") or "")
							import re as _re
							if not _re.match(rf"^\s*figure\s*{int(str(v))}\b", label, _re.I):
								return False
						elif k == "table_number":
							mn = meta.get("table_number")
							if str(mn) == str(v):
								continue
							label = str(meta.get("table_label") or "")
							import re as _re
							if not _re.match(rf"^\s*table\s*{int(str(v))}\b", label, _re.I):
								return False
						else:
							if meta.get(k) != v:
								return False
					return True
				out_rel = [d for d in docs if ok_rel(d.metadata)]
				if os.getenv("RAG_TRACE", "0").lower() in ("1","true","yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1","true","yes"):
					try:
						log = get_logger()
						log.debug("FILTER-FALLBACK(section): %d -> %d using %s", len(docs), len(out_rel), relaxed)
					except Exception:
						pass
				out = out_rel
			# If still empty, drop explicit figure/table numbers as well
			if not out and relaxed:
				relaxed2 = {k: v for k, v in relaxed.items() if k not in ("figure_number", "table_number")}
				if relaxed2 != relaxed:
					def ok_rel2(meta: dict):
						for k, v in (relaxed2 or {}).items():
							if k == "case_id":
								continue
							if meta.get(k) != v:
								return False
						return True
					out_rel2 = [d for d in docs if ok_rel2(d.metadata)]
					if os.getenv("RAG_TRACE", "0").lower() in ("1","true","yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1","true","yes"):
						try:
							log = get_logger()
							log.debug("FILTER-FALLBACK(numbers): %d -> %d using %s", len(docs), len(out_rel2), relaxed2)
						except Exception:
							pass
					out = out_rel2
		except Exception:
			pass
	# Final fallback: never return empty if we had candidates
	return out or docs


@trace_func
def build_hybrid_retriever(dense_store, sparse_retriever, dense_k: int = 10):
	"""Create an ensemble retriever with tunable weights via env vars.
	Defaults favor sparse slightly for keyword-heavy tech PDFs.
	"""
	print("this is me on the hybrid retriver")
	dense = dense_store.as_retriever(search_kwargs={"k": dense_k})
	try:
		sw = float(os.getenv("RAG_SPARSE_WEIGHT", "0.6"))
		dw = float(os.getenv("RAG_DENSE_WEIGHT", "0.4"))
		total = (sw + dw) or 1.0
		sw, dw = sw / total, dw / total
		print(f"this is me on the hybrid retriver sw {sw} dw {dw}")
	except Exception:
		sw, dw = 0.6, 0.4
	return EnsembleRetriever(retrievers=[sparse_retriever, dense], weights=[sw, dw])


@trace_func
def lexical_overlap(a: str, b: str) -> float:
	A, B = set(a.lower().split()), set(b.lower().split())
	if not A or not B:
		return 0.0
	return len(A & B) / len(A | B)


@trace_func
def calculate_content_quality_score(content: str, metadata: dict) -> float:
	"""Calculate content quality score to prioritize prose over metadata.
	
	This addresses the core retrieval issue where low-quality chunks (figure captions,
	table headers, etc.) are ranked equally with high-quality analytical prose.
	
	Returns:
		float: Quality multiplier (0.5-1.5) where higher = better content
	"""
	if not content or len(content.strip()) < 10:
		return 0.5
	
	content_lower = content.lower().strip()
	section_type = metadata.get("section") or metadata.get("section_type", "")
	
	# Strong penalties for known low-quality content types
	if section_type in ("Figure", "TableCell"):
		return 0.6
	
	# Detect figure captions and headers (short, metadata-like text)
	if len(content) < 100:
		# Very short content - likely caption, header, or metadata
		if any(indicator in content_lower for indicator in 
			   ["figure", "fig.", "table", "p.", "page", "source:", "caption"]):
			return 0.65
		# Short but could be meaningful (table data, technical specs)
		return 0.75
	
	# Detect prose vs. structured data characteristics
	sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 5]
	avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
	
	# Indicators of high-quality analytical prose
	prose_indicators = 0.0
	
	# Complex sentence structure (good for analysis)
	if avg_sentence_length > 15:
		prose_indicators += 0.3
	elif avg_sentence_length > 10:
		prose_indicators += 0.2
	
	# Analytical language patterns
	analytical_terms = [
		"analysis", "shows", "indicates", "demonstrates", "reveals", "suggests",
		"observed", "measured", "calculated", "determined", "found", "results",
		"conclusion", "evidence", "data", "study", "examination", "investigation"
	]
	analytical_matches = sum(1 for term in analytical_terms if term in content_lower)
	prose_indicators += min(0.3, analytical_matches * 0.05)
	
	# Technical depth indicators (good for gear analysis)
	technical_terms = [
		"spectral", "domain", "frequency", "amplitude", "peaks", "baseline",
		"mesh", "vibration", "wear", "rms", "analysis", "signal", "broadband"
	]
	technical_matches = sum(1 for term in technical_terms if term in content_lower)
	prose_indicators += min(0.2, technical_matches * 0.03)
	
	# Penalties for metadata-like content
	metadata_penalties = 0.0
	
	# List-like or bullet-point content
	if content.count('\n•') > 2 or content.count('\n-') > 2:
		metadata_penalties += 0.2
	
	# Very short lines (typical of captions/headers)
	lines = [line.strip() for line in content.split('\n') if line.strip()]
	if lines:
		avg_line_length = sum(len(line) for line in lines) / len(lines)
		if avg_line_length < 30:
			metadata_penalties += 0.3
	
	# Numeric-heavy content without context (raw data tables)
	import re
	numbers = re.findall(r'\d+\.?\d*', content)
	if len(numbers) > len(content.split()) * 0.3:  # More than 30% numbers
		if not any(term in content_lower for term in analytical_terms):
			metadata_penalties += 0.2
	
	# Calculate final quality score
	base_quality = 1.0
	quality_score = base_quality + prose_indicators - metadata_penalties
	
	# Ensure score stays in reasonable bounds
	return max(0.5, min(1.5, quality_score))


@trace_func
def calculate_semantic_relevance_boost(query: str, content: str, metadata: dict) -> float:
	"""Calculate semantic relevance boost for content that directly addresses the query.
	
	This addresses the issue where relevant content gets buried under noise by
	specifically rewarding content that contains the answer pattern.
	
	Returns:
		float: Relevance boost (0.0-0.5) to add to base score
	"""
	query_lower = query.lower()
	content_lower = content.lower()
	
	relevance_boost = 0.0
	
	# Direct topic matching - reward content that discusses the same concepts
	if "spectral" in query_lower and "spectral" in content_lower:
		if "domain" in content_lower and "analysis" in content_lower:
			relevance_boost += 0.2  # Exact topic match
	
	if "baseline" in query_lower and "baseline" in content_lower:
		relevance_boost += 0.15
	
	# Context density - reward content where query terms appear close together
	query_terms = [term for term in query_lower.split() 
				   if len(term) > 3 and term not in ["the", "and", "for", "with", "what", "how"]]
	
	if len(query_terms) >= 2:
		# Find windows where multiple query terms appear close together
		words = content_lower.split()
		for i in range(len(words) - 10):
			window = " ".join(words[i:i+10])
			matches_in_window = sum(1 for term in query_terms if term in window)
			if matches_in_window >= 2:
				relevance_boost += min(0.15, matches_in_window * 0.03)
				break
	
	# Answer pattern detection - look for content that provides explanations
	if any(question_word in query_lower for question_word in ["what", "how", "why", "describe"]):
		# Reward content with explanatory patterns
		explanatory_patterns = [
			r'shows?\s+that', r'indicates?\s+that', r'demonstrates?\s+that',
			r'reveals?\s+that', r'suggests?\s+that', r'found\s+that',
			r'observed\s+that', r'measured\s+', r'calculated\s+',
			r'analysis\s+shows?', r'results?\s+show', r'data\s+shows?'
		]
		
		import re
		for pattern in explanatory_patterns:
			if re.search(pattern, content_lower):
				relevance_boost += 0.1
				break
	
	# Length appropriateness - for "what" questions, prefer substantial explanations
	if "what" in query_lower or "describe" in query_lower:
		if len(content) > 200 and len(content) < 1000:  # Sweet spot for explanations
			relevance_boost += 0.05
	
	return min(0.5, relevance_boost)


@trace_func
def rerank_candidates(query: str, candidates: List[Document], top_n: int = 8, filters: Optional[Dict] = None) -> List[Document]:
	# If CE reranker is enabled and available, prefer it
	try:
		if os.getenv("RAG_USE_CE_RERANKER", "0").lower() in ("1", "true", "yes") and ce_rerank is not None:
			return ce_rerank(query, candidates, top_n=top_n)
	except Exception:
		pass
		
	# Extract conversation bias from filters
	conversation_bias = (filters or {}).get("conversation_bias")
	
	def _normalize(text: str) -> str:
		# lightweight normalization to improve sparse/lexical matching
		t = (text or "").lower()
		t = t.replace("μ", "u").replace("µ", "u")  # microunit to 'u'
		t = t.replace("μm", "um")
		t = t.replace("%", " percent ")
		t = t.replace("rps", " rps ")  # ensure token boundaries
		t = t.replace("–", "-").replace("—", "-")  # dashes
		# patterns like 10-15% -> add 10to15 percent token
		try:
			t = re.sub(r"(\d+)\s*-\s*(\d+)\s*percent", lambda m: f"{m.group(1)}to{m.group(2)} percent", t)
		except Exception:
			pass
		return t

	ql = _normalize(query)
	kws = set(re.findall(r"[A-Za-z0-9°%]+", ql))
	# Query-type flags for domain-specific co-occurrence boosts
	is_percent_query = ("percent" in ql)
	has_rms = ("rms" in ql)
	has_rps = (" rps " in ql) or ("rps" in ql)
	ai_image_task = any(w in ql for w in ("ai", "image", "vision", "task", "detection", "segmentation"))
	# Domain hint detection (gear vs materials) to bias sources; avoids introducing new flags
	dom_hint: str | None = None
	try:
		if re.search(r"\b(rps|rms|wear|w\d{1,2}|gear|mesh)\b", ql, re.I):
			dom_hint = "gear"
		if re.search(r"\b(cuticle|exocuticle|endocuticle|lamellae|pincer|tubules|conditioning|anneal)\b", ql, re.I):
			dom_hint = "materials"
	except Exception:
		dom_hint = None
	# Section preference based on query intent
	sec_pref = None
	if "table" in ql:
		sec_pref = "Table"
	elif any(w in ql for w in ("figure", "image", "fig ", "plot", "graph", "photo")):
		sec_pref = "Figure"
	# Sensor/metric/threshold inventory tends to live in tables; bias accordingly
	if any(w in ql for w in ("sensor", "sensors", "accelerometer", "tachometer", "instrumentation", "threshold", "alert threshold", "limits")):
		sec_pref = "Table"

	# Detect month/day phrases to boost timeline/date matches
	_months = [
		"january","february","march","april","may","june","july","august","september","october","november","december"
	]
	month_in_q = None
	day_in_q: str | None = None
	for m in _months:
		if m in ql:
			month_in_q = m
			break
	# capture patterns like "June 13th" or "June 13"
	md = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?", ql)
	if md:
		month_in_q = md.group(1)
		day_in_q = md.group(2)
	# also support ISO-like dates
	iso_in_q = re.search(r"20\d{2}-\d{2}-\d{2}", ql)

	def _len_penalty(n: int, is_figure: bool) -> float:
		# Loosen penalty for short figure captions so multiple figures can surface
		if is_figure:
			if n < 80:
				return 0.97
			if n > 2000:
				return 0.92
			return 1.0
		if n < 120:
			return 0.93
		if n > 3000:
			return 0.9
		return 1.0

	def _extractor_bonus(md: dict) -> float:
		if md.get("section") != "Table":
			return 0.0
		ext = str(md.get("extractor", ""))
		if ext.startswith("pdfplumber"):
			return 0.08
		if ext.startswith("tabula"):
			return 0.05
		if ext.startswith("camelot"):
			return 0.03
		if ext.startswith("synth"):
			return 0.0
		return 0.0

	# Extract numbers from query for light numeric co-occurrence boost
	q_numbers = set(re.findall(r"\b\d{1,4}\b", ql))

	scored = []
	for d in candidates:
		md = d.metadata or {}
		# normalized lexical overlap for content and metadata
		content_norm = _normalize(d.page_content)
		base = lexical_overlap(ql, content_norm)
		meta = _normalize(" ".join(map(str, md.values())))
		meta_boost = 0.2 * lexical_overlap(" ".join(kws), meta)
		# Extra boost for instrumentation/threshold queries matching table or metadata
		if any(w in ql for w in ("sensor", "sensors", "accelerometer", "tachometer", "instrumentation")):
			if (md.get("section") or md.get("section_type")) == "Table":
				meta_boost += 0.2
			if any(k in str(meta).lower() for k in ("sensor", "accelerometer", "tachometer")):
				meta_boost += 0.1
		sec_boost = 0.15 if (sec_pref and (md.get("section") or md.get("section_type")) == sec_pref) else 0.0
		src_boost = _extractor_bonus(md)

		# Date/timeline boost: if query mentions a month/day or ISO date and doc contains it
		date_boost = 0.0
		text_l = d.page_content.lower()
		try:
			if iso_in_q and iso_in_q.group(0) in text_l:
				date_boost += 0.25
			if month_in_q and month_in_q in text_l:
				date_boost += 0.18
				if day_in_q and re.search(rf"\b{month_in_q}\s+{day_in_q}(?:st|nd|rd|th)?\b", text_l):
					date_boost += 0.17
		except Exception:
			pass


		# Metadata token boost for dates (from attach_metadata)
		tokens_boost = 0.0
		try:
			months_md = [str(x).lower() for x in (md.get("month_tokens") or [])]
			days_md = [str(x) for x in (md.get("day_tokens") or [])]
			if month_in_q and months_md and month_in_q in months_md:
				tokens_boost += 0.12
			if day_in_q and days_md and day_in_q in days_md:
				tokens_boost += 0.12
		except Exception:
			pass

		# Bonus for explicit numbering for figures to reduce ambiguity
		number_bonus = 0.0
		try:
			if (md.get("section") == "Figure" or md.get("section_type") == "Figure") and md.get("figure_number"):
				number_bonus += 0.08
			if (md.get("section") == "Table" or md.get("section_type") == "Table") and md.get("table_number"):
				number_bonus += 0.05
			# Extra boost if the query asks for a specific Figure/Table number
			mf = re.search(r"\bfigure\s*(\d{1,3})\b", ql)
			if mf and str(md.get("figure_number")) == mf.group(1):
				number_bonus += 0.25
			mt = re.search(r"\btable\s*(\d{1,3})\b", ql)
			if mt and str(md.get("table_number")) == mt.group(1):
				number_bonus += 0.2
		except Exception:
			pass

		# Numeric co-occurrence boost (helps RPS, case numbers, table values)
		num_boost = 0.0
		try:
			if q_numbers:
				d_numbers = set(re.findall(r"\b\d{1,4}\b", content_norm))
				common = q_numbers & d_numbers
				if common:
					num_boost += min(0.12, 0.02 * len(common))
		except Exception:
			pass

		# Signal/measurement co-occurrence (RMS + percent + RPS in same chunk)
		signal_boost = 0.0
		try:
			if has_rms and is_percent_query and ("rms" in content_norm) and ("percent" in content_norm):
				signal_boost += 0.12
				# Extra nudge if specific speeds appear together
				if ("45" in ql and "45" in content_norm) or ("15" in ql and "15" in content_norm):
					signal_boost += 0.03
			if has_rps and (" rps " in content_norm or "rps" in content_norm):
				signal_boost += 0.05
				if ("45" in ql and "45" in content_norm) or ("15" in ql and "15" in content_norm):
					signal_boost += 0.03
		except Exception:
			pass

		# Figure OCR-aware adjustment: prefer figures that carry text when query asks for AI image tasks
		fig_text_adj = 0.0
		try:
			is_figure = (md.get("section") == "Figure" or md.get("section_type") == "Figure")
			if is_figure:
				if md.get("ocr_text"):
					fig_text_adj += 0.04
				elif ai_image_task:
					fig_text_adj -= 0.06
		except Exception:
			pass
		# Domain-aware filename boost
		domain_boost = 0.0
		try:
			if dom_hint:
				name = str((md.get("file_name") or md.get("file_path") or "")).lower()
				if dom_hint == "gear":
					if any(k in name for k in ("gear", "wear", "gear_wear")):
						domain_boost += 0.12
				elif dom_hint == "materials":
					if any(k in name for k in ("material", "cuticle", "exocuticle", "endocuticle")):
						domain_boost += 0.12
		except Exception:
			domain_boost = 0.0

		# NEW: Conversation bias boost
		conversation_boost = 0.0
		try:
			if conversation_bias:
				file_name = str(md.get("file_name", "")).lower()
				if conversation_bias == "gear_wear" and "gear" in file_name:
					conversation_boost += 0.15  # Strong bias toward gear documents
				elif conversation_bias == "bearing" and "bearing" in file_name:
					conversation_boost += 0.15  # Strong bias toward bearing documents
		except Exception:
			pass

		# NEW: Content quality scoring to prioritize prose over metadata
		quality_multiplier = calculate_content_quality_score(d.page_content, md)
		semantic_relevance = calculate_semantic_relevance_boost(query, d.page_content, md)

		# Calculate base score with all existing boosts including conversation bias
		base_score = (base + meta_boost + sec_boost + src_boost + date_boost + tokens_boost + 
					  number_bonus + num_boost + signal_boost + fig_text_adj + domain_boost + 
					  conversation_boost + semantic_relevance)
		
		# Apply quality multiplier to prioritize high-quality content
		score = base_score * quality_multiplier * _len_penalty(len(d.page_content), (md.get("section") == "Figure" or md.get("section_type") == "Figure"))
		
		# Attach transient score for UI/debug (do not persist to vectorstore)
		try:
			md_dbg = dict(md)
			md_dbg["_score"] = round(float(score), 4)
			md_dbg["_quality"] = round(float(quality_multiplier), 3)  # Debug info
			md_dbg["_semantic"] = round(float(semantic_relevance), 3)  # Debug info
			d.metadata = md_dbg
		except Exception:
			pass
		scored.append((score, len(d.page_content), d))

	# Sort and dedupe by (file,page,section,anchor/path) and collapse near-duplicate figure captions
	scored.sort(key=lambda x: (-x[0], x[1]))
	seen = set()
	seen_fig_captions = set()
	unique: List[Document] = []
	for s, ln, d in scored:
		md = d.metadata or {}
		key = (
			md.get("file_name"),
			md.get("page"),
			md.get("section") or md.get("section_type"),
			md.get("anchor") or md.get("table_md_path") or md.get("table_csv_path") or md.get("image_path")
		)
		if key in seen:
			continue
		# Collapse duplicates by similar figure captions to surface distinct figures
		try:
			if (md.get("section") == "Figure" or md.get("section_type") == "Figure"):
				cap = (md.get("figure_label") or "").strip().lower()
				cap_sig = cap[:80]
				if cap_sig and cap_sig in seen_fig_captions:
					continue
				if cap_sig:
					seen_fig_captions.add(cap_sig)
		except Exception:
			pass
		seen.add(key)
		unique.append(d)
		if len(unique) >= top_n:
			break

	try:
		if os.getenv("RAG_TRACE", "0").lower() in ("1", "true", "yes") or os.getenv("RAG_TRACE_RETRIEVAL", "0").lower() in ("1", "true", "yes"):
			log = get_logger()
			for i, d in enumerate(unique[:top_n], start=1):
				md = d.metadata or {}
				log.debug("RERANK[%d]: %s p%s %s score=%.4f", i, md.get("file_name"), md.get("page"), md.get("section"), (md.get("_score") or 0.0))
	except Exception:
		pass

	# Dominant-file gating: if one source clearly dominates the top results, keep a single-source context
	try:
		if unique:
			counts: dict[str, int] = {}
			for d in unique:
				fn = str((d.metadata or {}).get("file_name") or "")
				counts[fn] = counts.get(fn, 0) + 1
			# pick the file with max count (ignore empty name)
			dom_file = None
			dom_count = 0
			for fn, c in counts.items():
				if fn and c > dom_count:
					dom_file, dom_count = fn, c
			if dom_file:
				total = len(unique)
				# gate if majority is from one file
				if dom_count >= max(3, int(0.6 * total)):
					unique = [d for d in unique if str((d.metadata or {}).get("file_name") or "") == dom_file]
	except Exception:
		pass
	# Optional precision pruning by score threshold
	try:
		import os as _os
		thr = float(_os.getenv("RAG_MIN_CTX_SCORE", "0").strip() or 0.0)
	except Exception:
		thr = 0.0
	if thr > 0:
		pruned = []
		for d in unique:
			try:
				sc = float((d.metadata or {}).get("_score") or 0.0)
			except Exception:
				sc = 0.0
			if sc >= thr:
				pruned.append(d)
		if pruned:
			return pruned[:top_n]
	return unique[:top_n]

