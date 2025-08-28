import re
from typing import Dict, Any
from RAG.app.config import settings


def query_analyzer(query: str) -> Dict[str, Any]:
    """Analyze query to extract filters, keywords, and canonical form."""
    query_lower = query.lower()
    
    # Enhanced question type detection
    question_type = "general"
    if any(word in query_lower for word in ["gear", "gearbox", "transmission"]):
        question_type = "equipment_identification"
    elif any(word in query_lower for word in ["date", "when", "through", "until"]):
        question_type = "temporal"
    elif any(word in query_lower for word in ["how many", "number", "count", "total"]):
        question_type = "numeric"
    elif any(word in query_lower for word in ["figure", "fig", "plot", "graph"]):
        question_type = "figure_reference"
    elif any(word in query_lower for word in ["table", "data", "value", "measurement"]):
        question_type = "table_reference"
    elif any(word in query_lower for word in ["threshold", "alert", "limit", "criterion"]):
        question_type = "threshold_question"
    elif any(word in query_lower for word in ["escalation", "immediate", "urgent", "planning"]):
        question_type = "escalation_question"
    elif any(word in query_lower for word in ["wear depth", "wear cases"] + settings.query_analysis.WEAR_CASES):
        question_type = "wear_depth_question"
    elif any(word in query_lower for word in ["accelerometer", "sensor", "dytran"]):
        question_type = "sensor_question"
    elif any(word in query_lower for word in ["tachometer", "honeywell", "teeth"]):
        question_type = "tachometer_question"
    elif any(word in query_lower for word in ["rms", "fft", "spectrogram", "sideband", "meshing"]):
        question_type = "spectral_analysis"
    
    # Enhanced keyword extraction
    keywords = []
    
    # Technical terms from config
    for term in settings.query_analysis.TECHNICAL_TERMS:
        if term in query_lower:
            keywords.append(term)
    
    # Wear case identifiers from config
    for case in settings.query_analysis.WEAR_CASES:
        if case in query_lower:
            keywords.append(case)
    
    # Figure references from config
    for fig in settings.query_analysis.FIGURE_REFS:
        if fig in query_lower:
            keywords.append(fig)
    
    # Numbers and measurements
    numbers = re.findall(r'\d+(?:\.\d+)?', query)
    keywords.extend(numbers)
    
    # Units from config
    for unit in settings.query_analysis.UNITS:
        if unit in query_lower:
            keywords.append(unit)
    
    # Equipment and case identifiers from config
    for eq in settings.query_analysis.EQUIPMENT:
        if eq in query_lower:
            keywords.append(eq)
    
    # Enhanced table and figure detection from config
    is_table_question = any(word in query_lower for word in settings.query_analysis.TABLE_QUESTION_KEYWORDS)
    is_figure_question = any(word in query_lower for word in settings.query_analysis.FIGURE_QUESTION_KEYWORDS)
    is_threshold_question = any(word in query_lower for word in settings.query_analysis.THRESHOLD_QUESTION_KEYWORDS)
    is_escalation_question = any(word in query_lower for word in settings.query_analysis.ESCALATION_QUESTION_KEYWORDS)
    is_module_question = "module value" in query_lower or ("module" in query_lower and "value" in query_lower)
    
    return {
        "question_type": question_type,
        "keywords": keywords,
        "is_table_question": is_table_question,
        "is_figure_question": is_figure_question,
        "is_threshold_question": is_threshold_question,
        "is_escalation_question": is_escalation_question,
        "is_module_question": is_module_question,
        "original_query": query
    }
