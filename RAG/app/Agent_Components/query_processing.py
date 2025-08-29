"""
Query processing module for the RAG system.
Handles query analysis, filtering, and preprocessing.
"""

import re
from typing import List, Dict, Any
from langchain.schema import Document

from RAG.app.interfaces import QueryProcessingInterface
from RAG.app.config import settings
from RAG.app.logger import get_logger
from RAG.app.Performance_Optimization.caching import cache_query_analysis
from RAG.app.Evaluation_Analysis.progress_tracking import monitor_performance


class QueryProcessingService(QueryProcessingInterface):
    """Service for processing and analyzing queries."""
    
    def __init__(self, config=None):
        self.config = config or settings
        self.logger = get_logger()
    
    @cache_query_analysis(ttl=3600)
    @monitor_performance
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query and extract relevant information."""
        try:
            query_lower = query.lower()
            
            # Enhanced question type detection
            question_type = self._detect_question_type(query_lower)
            
            # Extract keywords
            keywords = self._extract_keywords(query_lower)
            
            # Detect question types
            question_flags = self._detect_question_flags(query_lower)
            
            return {
                "question_type": question_type,
                "keywords": keywords,
                "original_query": query,
                **question_flags
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}")
            return {
                "question_type": "general",
                "keywords": [],
                "original_query": query,
                "is_table_question": False,
                "is_figure_question": False,
                "is_threshold_question": False,
                "is_escalation_question": False,
                "is_module_question": False
            }
    
    def apply_filters(self, documents: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Apply filters to documents based on query analysis."""
        from RAG.app.retrieve_modules.retrieve_filters import apply_filters as apply_filters_impl
        return apply_filters_impl(documents, filters)
    
    def _detect_question_type(self, query_lower: str) -> str:
        """Detect the type of question being asked."""
        if any(word in query_lower for word in ["gear", "gearbox", "transmission"]):
            return "equipment_identification"
        elif any(word in query_lower for word in ["date", "when", "through", "until"]):
            return "temporal"
        elif any(word in query_lower for word in ["how many", "number", "count", "total"]):
            return "numeric"
        elif any(word in query_lower for word in ["figure", "fig", "plot", "graph"]):
            return "figure_reference"
        elif any(word in query_lower for word in ["table", "data", "value", "measurement"]):
            return "table_reference"
        elif any(word in query_lower for word in ["threshold", "alert", "limit", "criterion"]):
            return "threshold_question"
        elif any(word in query_lower for word in ["escalation", "immediate", "urgent", "planning"]):
            return "escalation_question"
        elif any(word in query_lower for word in ["wear depth"] + self.config.query_analysis.WEAR_CASES):
            return "wear_depth_question"
        elif any(word in query_lower for word in ["accelerometer", "sensor", "dytran"]):
            return "sensor_question"
        elif any(word in query_lower for word in ["tachometer", "honeywell", "teeth"]):
            return "tachometer_question"
        elif any(word in query_lower for word in ["rms", "fft", "spectrogram", "sideband", "meshing"]):
            return "spectral_analysis"
        else:
            return "general"
    
    def _extract_keywords(self, query_lower: str) -> List[str]:
        """Extract keywords from the query."""
        keywords = []
        
        # Technical terms
        for term in self.config.query_analysis.TECHNICAL_TERMS:
            if term in query_lower:
                keywords.append(term)
        
        # Wear case identifiers
        for case in self.config.query_analysis.WEAR_CASES:
            if case in query_lower:
                keywords.append(case)
        
        # Figure references
        for fig in self.config.query_analysis.FIGURE_REFS:
            if fig in query_lower:
                keywords.append(fig)
        
        # Numbers and measurements
        numbers = re.findall(r'\d+(?:\.\d+)?', query_lower)
        keywords.extend(numbers)
        
        # Units
        for unit in self.config.query_analysis.UNITS:
            if unit in query_lower:
                keywords.append(unit)
        
        # Equipment and case identifiers
        for eq in self.config.query_analysis.EQUIPMENT:
            if eq in query_lower:
                keywords.append(eq)
        
        return keywords
    
    def _detect_question_flags(self, query_lower: str) -> Dict[str, bool]:
        """Detect various question type flags."""
        return {
            "is_table_question": any(word in query_lower for word in self.config.query_analysis.TABLE_QUESTION_KEYWORDS),
            "is_figure_question": any(word in query_lower for word in self.config.query_analysis.FIGURE_QUESTION_KEYWORDS),
            "is_threshold_question": any(word in query_lower for word in self.config.query_analysis.THRESHOLD_QUESTION_KEYWORDS),
            "is_escalation_question": any(word in query_lower for word in self.config.query_analysis.ESCALATION_QUESTION_KEYWORDS),
            "is_module_question": "module value" in query_lower or ("module" in query_lower and "value" in query_lower)
        }
    

    
    def preprocess_query(self, query: str) -> str:
        """Preprocess a query for better matching."""
        try:
            # Remove excessive whitespace
            query = ' '.join(query.split())
            
            # Normalize case for certain terms
            query = query.replace("RPM", "rpm").replace("RPS", "rps")
            
            return query.strip()
            
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {e}")
            return query


class QueryProcessingFactory:
    """Factory for creating query processing services."""
    
    @staticmethod
    def create_service(config=None) -> QueryProcessingService:
        """Create a query processing service with the given configuration."""
        return QueryProcessingService(config)
