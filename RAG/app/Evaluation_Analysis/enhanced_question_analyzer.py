"""
EnhancedQuestionAnalyzer: Advanced question analysis and classification.
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from RAG.app.Agent_Components.query_processing import QueryProcessingService
from RAG.app.logger import get_logger

logger = get_logger()


class EnhancedQuestionAnalyzer:
    """Enhanced question analyzer with advanced classification capabilities."""
    
    def __init__(self):
        self.logger = get_logger()
        
        # Question type patterns
        self.question_patterns = {
            "factual": [
                r"what is",
                r"what are",
                r"when",
                r"where",
                r"who",
                r"how many",
                r"how much"
            ],
            "analytical": [
                r"why",
                r"how does",
                r"explain",
                r"describe",
                r"compare",
                r"analyze"
            ],
            "comparative": [
                r"compare",
                r"difference",
                r"similar",
                r"versus",
                r"vs",
                r"better",
                r"worse"
            ],
            "numerical": [
                r"\d+",
                r"percentage",
                r"rate",
                r"frequency",
                r"average",
                r"mean",
                r"median"
            ],
            "temporal": [
                r"when",
                r"time",
                r"date",
                r"period",
                r"duration",
                r"before",
                r"after"
            ]
        }
        
        # Technical terms for domain-specific analysis
        self.technical_terms = {
            "gear_wear": [
                "wear", "gear", "tooth", "failure", "fatigue", "crack", "pitting",
                "scuffing", "abrasion", "corrosion", "lubrication", "vibration"
            ],
            "measurement": [
                "rms", "fft", "spectrum", "frequency", "amplitude", "signal",
                "noise", "filter", "analysis", "measurement", "sensor"
            ],
            "materials": [
                "steel", "alloy", "hardness", "tensile", "strength", "material",
                "composition", "heat treatment", "surface finish"
            ]
        }
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Comprehensive question analysis."""
        try:
            question_lower = question.lower().strip()
            
            # Basic query analysis
            query_service = QueryProcessingService()
            basic_analysis = query_service.analyze_query(question)
            
            # Enhanced analysis
            enhanced_analysis = {
                "question_type": self._classify_question_type(question_lower),
                "complexity": self._assess_complexity(question_lower),
                "domain": self._identify_domain(question_lower),
                "entities": self._extract_entities(question_lower),
                "intent": self._determine_intent(question_lower),
                "technical_terms": self._find_technical_terms(question_lower)
            }
            
            # Combine analyses
            full_analysis = {
                **basic_analysis,
                **enhanced_analysis
            }
            
            return full_analysis
            
        except Exception as e:
            self.logger.error(f"Question analysis failed: {e}")
            return {"error": str(e)}
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return qtype
        return "general"
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity."""
        word_count = len(question.split())
        
        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "moderate"
        else:
            return "complex"
    
    def _identify_domain(self, question: str) -> str:
        """Identify the domain of the question."""
        for domain, terms in self.technical_terms.items():
            for term in terms:
                if term in question:
                    return domain
        return "general"
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract named entities from the question."""
        entities = []
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        entities.extend(numbers)
        
        # Extract technical terms
        for domain, terms in self.technical_terms.items():
            for term in terms:
                if term in question:
                    entities.append(term)
        
        return list(set(entities))
    
    def _determine_intent(self, question: str) -> str:
        """Determine the intent of the question."""
        if any(word in question for word in ["what", "who", "when", "where"]):
            return "information_retrieval"
        elif any(word in question for word in ["why", "how", "explain"]):
            return "explanation"
        elif any(word in question for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in question for word in ["analyze", "evaluate", "assess"]):
            return "analysis"
        else:
            return "general"
    
    def _find_technical_terms(self, question: str) -> Dict[str, List[str]]:
        """Find technical terms by domain."""
        found_terms = {}
        
        for domain, terms in self.technical_terms.items():
            domain_terms = [term for term in terms if term in question]
            if domain_terms:
                found_terms[domain] = domain_terms
        
        return found_terms
    
    def suggest_related_questions(self, question: str) -> List[str]:
        """Suggest related questions based on analysis."""
        try:
            analysis = self.analyze_question(question)
            suggestions = []
            
            # Generate suggestions based on question type
            if analysis.get("question_type") == "factual":
                suggestions.extend([
                    "What are the causes of this?",
                    "How does this affect performance?",
                    "What are the solutions?"
                ])
            
            if analysis.get("domain") == "gear_wear":
                suggestions.extend([
                    "What are the common wear patterns?",
                    "How can wear be prevented?",
                    "What are the maintenance recommendations?"
                ])
            
            if analysis.get("domain") == "measurement":
                suggestions.extend([
                    "What equipment is used for measurement?",
                    "How accurate are these measurements?",
                    "What are the measurement standards?"
                ])
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Question suggestion failed: {e}")
            return []
    
    def validate_question(self, question: str) -> Dict[str, Any]:
        """Validate if the question is well-formed."""
        try:
            validation = {
                "is_valid": True,
                "issues": [],
                "suggestions": []
            }
            
            if not question or len(question.strip()) < 3:
                validation["is_valid"] = False
                validation["issues"].append("Question too short")
            
            if len(question.split()) > 50:
                validation["issues"].append("Question very long - consider breaking it down")
            
            if not any(word in question.lower() for word in ["what", "how", "why", "when", "where", "who", "which"]):
                validation["suggestions"].append("Consider making this a question")
            
            return validation
            
        except Exception as e:
            return {"is_valid": False, "issues": [f"Validation error: {e}"], "suggestions": []}
