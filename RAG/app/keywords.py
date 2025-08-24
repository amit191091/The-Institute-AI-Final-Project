"""
Keywords module for gear wear analysis RAG system.
Centralizes all keyword definitions and extraction logic.
"""

import re
from typing import List, Set, Dict, Tuple

# Common stop words to filter out
COMMON_WORDS = {
    "what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
    "of", "with", "by", "how", "many", "when", "where", "why", "which", "who", "were", 
    "was", "are", "do", "does", "did", "have", "has", "had", "will", "would", "could", 
    "should", "can", "may", "might", "this", "that", "these", "those", "it", "its", 
    "their", "there", "here", "from", "into", "over", "under", "than", "as", "be", 
    "been", "being"
}

# Technical multi-word terms specific to gear analysis
TECHNICAL_MULTI_WORD_TERMS = [
    "transmission ratio", "gear module", "wear depth", "failure date", 
    "wear cases", "vibration analysis", "measurement data", "gear teeth",
    "lubricant type", "oil grade", "rms value", "frequency analysis",
    "root mean square", "amplitude level", "technical analysis",
    "gear wear", "failure analysis", "vibration measurement",
    "wear measurement", "gear failure", "transmission system",
    "gear parameters", "wear analysis", "vibration data"
]

# Technical values and measurements
TECHNICAL_VALUES = [
    "18/35", "18:35", "3mm", "3 mm", "15w/40", "15w-40", "35", "15",
    "18", "3", "40", "15w", "w40", "rpm", "hz", "db", "mm", "cm",
    "degrees", "percent", "%", "ratio", "module"
]

# Core technical terms for gear analysis
CORE_TECHNICAL_TERMS = [
    "gear", "transmission", "ratio", "module", "wear", "depth", "failure", 
    "vibration", "measurement", "data", "table", "figure", "analysis",
    "teeth", "tooth", "mesh", "lubricant", "oil", "lubrication", "rms",
    "frequency", "amplitude", "root", "mean", "square", "technical",
    "parameter", "system", "case", "study", "report", "document"
]

# Regex patterns for key information extraction
KEY_PATTERNS = [
    r'\b18/35\b', r'\b18:35\b', r'\b3\s*mm\b', r'\bjune\s*15\b', r'\b15\s*june\b',
    r'\btransmission\s*ratio\b', r'\bgear\s*module\b', r'\bfailure\s*date\b',
    r'\bwear\s*depth\b', r'\bvibration\b', r'\bmeasurement\b',
    r'\b15w/40\b', r'\b15w-40\b', r'\brms\b', r'\broot\s*mean\s*square\b',
    r'\bfrequency\s*analysis\b', r'\bgear\s*teeth\b', r'\bwear\s*cases\b'
]

# Question intent mapping
QUESTION_INTENT_MAPPING = {
    "transmission ratio": ["transmission ratio", "18/35", "18:35", "ratio", "gear"],
    "gear module": ["gear module", "3 mm", "module", "gear"],
    "failure date": ["failure", "june 15", "15 june", "date", "when"],
    "wear depth": ["wear depth", "measurement", "data", "wear", "depth"],
    "wear cases": ["cases", "wear", "data", "measurement", "35"],
    "vibration": ["vibration", "rms", "measurement", "data", "frequency"],
    "lubricant": ["lubricant", "oil", "15w/40", "15w-40", "lubrication"],
    "teeth": ["teeth", "gear", "tooth", "mesh", "transmission"]
}

def get_common_words() -> Set[str]:
    """Get the set of common words to filter out."""
    return COMMON_WORDS.copy()

def get_technical_multi_word_terms() -> List[str]:
    """Get list of technical multi-word terms."""
    return TECHNICAL_MULTI_WORD_TERMS.copy()

def get_technical_values() -> List[str]:
    """Get list of technical values and measurements."""
    return TECHNICAL_VALUES.copy()

def get_core_technical_terms() -> List[str]:
    """Get list of core technical terms."""
    return CORE_TECHNICAL_TERMS.copy()

def get_key_patterns() -> List[str]:
    """Get list of regex patterns for key information extraction."""
    return KEY_PATTERNS.copy()

def extract_question_keywords(question: str) -> Set[str]:
    """
    Extract important keywords from a question.
    
    Args:
        question: The input question string
        
    Returns:
        Set of extracted keywords
    """
    if not question:
        return set()
    
    question_lower = question.lower()
    
    # Remove common words and focus on technical terms
    words = set(re.findall(r'\b\w+\b', question_lower))
    keywords = words - COMMON_WORDS
    
    # Add multi-word terms
    for term in TECHNICAL_MULTI_WORD_TERMS:
        if term in question_lower:
            keywords.add(term)
    
    # Add technical values
    for value in TECHNICAL_VALUES:
        if value in question_lower:
            keywords.add(value)
    
    return keywords

def extract_expected_information(question: str) -> List[str]:
    """
    Extract expected information based on question type.
    
    Args:
        question: The input question string
        
    Returns:
        List of expected information terms
    """
    if not question:
        return []
    
    question_lower = question.lower()
    expected = []
    
    # Check for specific technical queries
    for intent, terms in QUESTION_INTENT_MAPPING.items():
        if intent in question_lower:
            expected.extend(terms)
    
    # Add general technical terms that should be present
    for term in CORE_TECHNICAL_TERMS:
        if term in question_lower and term not in expected:
            expected.append(term)
    
    # Add question-specific keywords
    question_words = re.findall(r'\b\w+\b', question_lower)
    technical_words = ["ratio", "module", "depth", "cases", "failure", 
                      "vibration", "measurement", "data", "table", "figure"]
    
    for word in question_words:
        if word in technical_words and word not in expected:
            expected.append(word)
    
    return expected

def extract_key_information(text: str) -> Set[str]:
    """
    Extract key information from text using regex patterns.
    
    Args:
        text: The input text to analyze
        
    Returns:
        Set of extracted key information
    """
    if not text:
        return set()
    
    info = set()
    for pattern in KEY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info.update(matches)
    
    return info

def get_technical_bonus_terms() -> List[str]:
    """Get terms that provide bonus scoring in evaluation."""
    return CORE_TECHNICAL_TERMS.copy()

def calculate_keyword_overlap(text1: str, text2: str) -> float:
    """
    Calculate keyword overlap between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    keywords1 = extract_question_keywords(text1)
    keywords2 = extract_question_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union) if union else 0.0

def get_domain_specific_keywords() -> Dict[str, List[str]]:
	"""
	Get domain-specific keyword categories.
	
	Returns:
		Dictionary mapping categories to keyword lists
	"""
	return {
		"measurements": ["depth", "width", "length", "height", "diameter", "radius"],
		"units": ["mm", "cm", "m", "degrees", "percent", "%", "rpm", "hz", "db"],
		"gear_components": ["teeth", "tooth", "mesh", "module", "ratio", "transmission"],
		"wear_analysis": ["wear", "depth", "measurement", "analysis", "failure"],
		"vibration_analysis": ["vibration", "rms", "frequency", "amplitude", "root", "mean", "square"],
		"lubrication": ["lubricant", "oil", "lubrication", "grade", "15w/40", "15w-40"],
		"data_types": ["table", "figure", "data", "measurement", "analysis", "report"]
	}

def extract_question_intent(question: str) -> List[str]:
	"""
	Extract the intent of the question.
	
	Args:
		question: The input question string
		
	Returns:
		List of detected intents
	"""
	if not question:
		return []
	
	question_lower = question.lower()
	intent = []
	
	if "transmission ratio" in question_lower:
		intent.append("transmission ratio")
	if "gear module" in question_lower:
		intent.append("gear module")
	if "failure" in question_lower and "when" in question_lower:
		intent.append("failure date")
	if "wear depth" in question_lower:
		intent.append("wear depth")
	if "wear cases" in question_lower:
		intent.append("wear cases")
	if "vibration" in question_lower:
		intent.append("vibration analysis")
	if "lubricant" in question_lower or "oil" in question_lower:
		intent.append("lubrication")
	if "teeth" in question_lower:
		intent.append("gear teeth")
	
	return intent
