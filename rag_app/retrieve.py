"""
Query analysis and hybrid retrieval with metadata filtering and lightweight reranking
"""
import re
import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict

from langchain.docstore.document import Document

from .indexing import HybridIndex
from .metadata import extract_entities as extract_text_entities
from .utils import extract_numeric_values
from .config import settings

logger = logging.getLogger(__name__)


class SimpleReranker:
    """A lightweight reranker using token overlap and simple heuristics.

    Avoids heavy ML dependencies (transformers/sentence-transformers) for fast startup.
    """

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"\b\w{3,}\b", (text or "").lower())

    def predict(self, pairs: List[List[str]]) -> List[float]:
        scores: List[float] = []
        for query, text in pairs:
            q = self._tok(query)
            t = self._tok(text)
            if not t:
                scores.append(0.0)
                continue
            qset = set(q)
            overlap = sum(1 for tok in t if tok in qset)
            density = overlap / max(1, len(t))
            phrase = 1.0 if (query or "").strip().lower() in (text or "").lower() else 0.0
            short_pen = -0.2 if len(text) < 80 else 0.0
            score = overlap + 2.0 * density + 1.5 * phrase + short_pen
            scores.append(score)
        return scores


class QueryAnalyzer:
    """Analyzes queries to extract intent, entities, filters, and preferences."""

    def __init__(self) -> None:
        self.intent_patterns = {
            "summary": [r"\bsummar", r"\boverview", r"\babstract", r"\bmain\s+points"],
            "timeline": [r"\btimeline", r"\bchronolog", r"\bhistory", r"\bsequence", r"\bwhen"],
            "table": [r"\btable", r"\bdata", r"\bmeasurement", r"\bvalue", r"\bnumber"],
            "technical": [r"\banalysis", r"\btechnical", r"\bmethod", r"\bprocedure", r"\bhow"],
            "specific": [r"\bspecific", r"\bexact", r"\bprecise", r"\bdetail"],
        }
        self.entity_patterns = {
            "component": [r"\b(gear|bearing|shaft|housing|seal)s?\b"],
            "failure_mode": [r"\b(wear|fatigue|crack|fracture|seizure|overheat)\b"],
            "measurement": [r"\b\d+(?:\.\d+)?\s?(?:μm|um|mm|cm|MPa|RPM|Hz|°C)\b"],
            "case_id": [r"\b(?:case|client)[-_ ]?\w+\b"],
            "date": [r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"],
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        q_lower = (query or "").lower().strip()
        entities = self._extract_query_entities(query)
        filters = self._generate_metadata_filters(query)
        keywords = self._extract_keywords(query)
        numeric_constraints = self._extract_numeric_constraints(query)
        section_pref = self._detect_section_preference(q_lower)
        complexity = self._assess_query_complexity(q_lower)

        return {
            "original_query": query,
            "intent": self._classify_intent(q_lower),
            "entities": entities,
            "filters": filters,
            "keywords": keywords,
            "numeric_constraints": numeric_constraints,
            "section_preference": section_pref,
            "complexity": complexity,
        }

    def _classify_intent(self, q_lower: str) -> str:
        scores = {k: 0 for k in self.intent_patterns}
        for intent, patterns in self.intent_patterns.items():
            for p in patterns:
                if re.search(p, q_lower, re.IGNORECASE):
                    scores[intent] += 1
        return max(scores, key=lambda k: scores[k]) if any(scores.values()) else "general"

    def _extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        found = defaultdict(list)
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for m in re.finditer(pattern, query or "", re.IGNORECASE):
                    found[entity_type].append(m.group(0))
        return {k: sorted(set(v)) for k, v in found.items()}

    def _generate_metadata_filters(self, query: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}

        # IDs
        m = re.search(r"\bcase[-_ ]?(\w+)\b", query or "", re.IGNORECASE)
        if m:
            filters["case_id"] = m.group(1)
        m = re.search(r"\bclient[-_ ]?(\w+)\b", query or "", re.IGNORECASE)
        if m:
            filters["client_id"] = m.group(1)

        # Year/date
        m = re.search(r"\b(20\d{2})\b", query or "")
        if m:
            filters["year"] = m.group(1)

        # Failure types and severity
        ql = (query or "").lower()
        for ft in ["wear", "fatigue", "crack", "fracture", "seizure", "overheat"]:
            if ft in ql:
                filters["incident_type"] = ft
                break
        if any(x in ql for x in ["critical", "severe", "emergency"]):
            filters["severity"] = "critical"
        elif any(x in ql for x in ["major", "significant", "serious"]):
            filters["severity"] = "high"

        return filters

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "this", "that", "these", "those", "can", "could",
            "should", "would", "will", "do", "does", "did", "have", "has", "had",
        }
        words = re.findall(r"\b\w+\b", query or "")
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        return keywords[:10]

    def _extract_numeric_constraints(self, query: str) -> List[Dict[str, Any]]:
        nums = extract_numeric_values(query or "")
        constraints: List[Dict[str, Any]] = []
        for value, unit, start, end in nums:
            constraints.append(
                {
                    "value": value,
                    "unit": unit,
                    "context": (query or "")[max(0, start - 20) : min(len(query or ""), end + 20)],
                    "operator": "exact",
                }
            )
        return constraints

    def _detect_section_preference(self, q_lower: str) -> Optional[str]:
        indicators = {
            "Table": ["table", "data", "measurement", "value", "number"],
            "Figure": ["figure", "image", "chart", "graph", "diagram"],
            "Summary": ["summary", "overview", "abstract"],
            "Timeline": ["timeline", "chronology", "history", "sequence"],
            "Analysis": ["analysis", "method", "procedure", "technical"],
            "Conclusion": ["conclusion", "result", "finding", "outcome"],
        }
        for section, words in indicators.items():
            if any(w in q_lower for w in words):
                return section
        return None

    def _assess_query_complexity(self, q_lower: str) -> str:
        wc = len((q_lower or "").split())
        if wc <= 3:
            return "simple"
        if wc <= 8:
            return "medium"
        return "complex"


class HybridRetriever:
    """Hybrid retrieval combining dense, sparse, and metadata filtering with lightweight reranking."""

    def __init__(self, index: HybridIndex) -> None:
        self.index = index
        self.query_analyzer = QueryAnalyzer()
        self.reranker = SimpleReranker()

    def retrieve(self, query: str, k_final: Optional[int] = None) -> List[Document]:
        k_final = k_final or settings.CONTEXT_TOP_N

        analysis = self.query_analyzer.analyze_query(query)
        logger.info("Query analysis intent: %s", analysis.get("intent"))

        if analysis.get("intent") == "table":
            return self._retrieve_table_focused(query, analysis, k_final)

        return self._retrieve_hybrid(query, analysis, k_final)

    def _retrieve_hybrid(self, query: str, analysis: Dict[str, Any], k_final: int) -> List[Document]:
        filters = dict(analysis.get("filters") or {})
        section_pref = analysis.get("section_preference")
        if section_pref:
            filters["section"] = section_pref

        dense_results = self.index.search_dense(query, k=settings.DENSE_K, filter_dict=filters or None)
        sparse_results = self.index.search_sparse(query, k=settings.SPARSE_K, filter_dict=filters or None)

        combined_results = self._combine_and_deduplicate(dense_results, sparse_results)

        if self.reranker and len(combined_results) > k_final:
            combined_results = self._rerank_documents(query, combined_results)

        return combined_results[:k_final]

    def _retrieve_table_focused(self, query: str, analysis: Dict[str, Any], k_final: int) -> List[Document]:
        table_summaries = self.index.get_table_summaries(analysis.get("filters"))
        if table_summaries:
            table_docs: List[Document] = []
            for table_info in table_summaries[: max(1, k_final // 2)]:
                table_docs.append(
                    Document(
                        page_content=table_info.get("raw_content", ""),
                        metadata={
                            "file_name": table_info.get("file_name"),
                            "page": table_info.get("page_number", 1),
                            "section": "Table",
                            "table_id": table_info.get("table_id"),
                        },
                    )
                )

            remaining_k = k_final - len(table_docs)
            if remaining_k > 0:
                other = self._retrieve_hybrid(query, analysis, remaining_k * 2)
                other = [d for d in other if d.metadata.get("section") != "Table"]
                table_docs.extend(other[:remaining_k])

            return table_docs

        return self._retrieve_hybrid(query, analysis, k_final)

    def _combine_and_deduplicate(self, dense_results: List[Document], sparse_results: List[Document]) -> List[Document]:
        seen = set()
        combined: List[Document] = []
        for doc in dense_results:
            key = hash((doc.metadata.get("file_name"), doc.metadata.get("chunk_id"), doc.page_content[:200]))
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        for doc in sparse_results:
            key = hash((doc.metadata.get("file_name"), doc.metadata.get("chunk_id"), doc.page_content[:200]))
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        return combined

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            pairs = [[query, (doc.page_content or "")[:512]] for doc in documents]
            scores = self.reranker.predict(pairs)
            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [d for d, _ in scored]
        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return documents


def create_retriever(index: HybridIndex) -> "HybridRetriever":
    return HybridRetriever(index)
