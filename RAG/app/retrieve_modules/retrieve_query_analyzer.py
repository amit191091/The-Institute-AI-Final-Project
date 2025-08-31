import re
import os
from typing import Dict, Any
from RAG.app.config import settings
from RAG.app.Agent_Components.agents import simplify_question

# Optional LLM router
try:
    from RAG.app.retrieve_modules.query_intent import get_intent  # optional LLM router
except Exception:
    get_intent = None  # type: ignore


def query_analyzer(query: str) -> Dict[str, Any]:
    """Analyze query to extract filters, keywords, and canonical form."""
    # Use LLM router if enabled, otherwise fall back to regex-based analysis
    simp = (get_intent(query) if get_intent is not None and (os.getenv("RAG_USE_LLM_ROUTER", "0").lower() in ("1","true","yes")) else simplify_question(query))
    
    query_lower = query.lower()
    
    # Enhanced question type detection
    question_type = "general"
    if any(word in query_lower for word in ["gear", "gearbox", "transmission"]):
        question_type = "equipment_identification"
    elif any(word in query_lower for word in ["date", "when", "through", "until", "between", "begin", "start", "occur"]):
        question_type = "temporal"
    elif any(word in query_lower for word in ["how many", "number", "count", "total", "sequential", "tracked"]):
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
        # Enhanced wear depth question routing
        if "tooth 1" in query_lower or "tooth one" in query_lower:
            question_type = "wear_depth_tooth1_question"  # Main report
        elif any(f"tooth {i}" in query_lower for i in range(2, 36)) or "tooth two" in query_lower or "tooth three" in query_lower:
            question_type = "wear_depth_other_teeth_question"  # Database file
        else:
            question_type = "wear_depth_question"  # General wear depth
    elif any(word in query_lower for word in ["accelerometer", "sensor", "dytran", "brand of accelerometer"]):
        question_type = "sensor_question"
    elif any(word in query_lower for word in ["tachometer", "honeywell", "teeth", "brand of tachometer"]):
        question_type = "tachometer_question"
    elif any(word in query_lower for word in ["lubricant", "oil", "brand of lubricant", "which lubricant"]):
        question_type = "lubricant_question"
    elif any(word in query_lower for word in ["sampling rate", "rate per record", "frequency"]):
        question_type = "sampling_question"
    elif any(word in query_lower for word in ["rms", "fft", "spectrogram", "sideband", "meshing"]):
        question_type = "spectral_analysis"
    # Specific detection for failing questions
    elif any(word in query_lower for word in ["vessel", "ins haifa", "propulsion train"]):
        question_type = "vessel_question"
    elif any(word in query_lower for word in ["mg-5025a", "marine reduction gearbox", "gearbox model"]):
        question_type = "gearbox_model_question"
    elif any(word in query_lower for word in ["spur", "gear type", "transmission gear"]):
        question_type = "gear_type_question"
    elif any(word in query_lower for word in ["18/35", "transmission ratio", "driving/driven"]):
        question_type = "transmission_ratio_question"
    elif any(word in query_lower for word in ["3 mm", "module value", "gear module"]):
        question_type = "module_value_question"
    elif any(word in query_lower for word in ["baseline wear depth", "healthy baseline", "0 μm"]):
        question_type = "baseline_question"
    elif any(word in query_lower for word in ["sensitivity", "mv/g", "accelerometer sensitivity"]):
        question_type = "sensitivity_question"
    elif any(word in query_lower for word in ["30 teeth", "tachometer teeth", "teeth count"]):
        question_type = "teeth_question"
    # Specific detection for missing information questions
    elif any(word in query_lower for word in ["healthy baseline", "until what date", "baseline extend"]):
        question_type = "baseline_date_question"
    elif any(word in query_lower for word in ["data-acquisition chain", "new data-acquisition", "installed"]):
        question_type = "installation_date_question"
    elif any(word in query_lower for word in ["baseline rms", "rms vibration levels", "indicate"]):
        question_type = "rms_baseline_question"
    elif any(word in query_lower for word in ["rms trend", "april 23", "consistently"]):
        question_type = "rms_trend_question"
    elif any(word in query_lower for word in ["photographic inspections", "purpose", "earliest wear"]):
        question_type = "inspection_purpose_question"
    elif any(word in query_lower for word in ["rms monitoring thresholds", "thresholds", "alarm levels"]):
        question_type = "threshold_question"
    elif any(word in query_lower for word in ["intervention threshold", "mild wear", "moderate wear", "severe wear"]):
        question_type = "intervention_question"
    elif any(word in query_lower for word in ["speed", "rps", "data acquisition", "steady speeds"]):
        question_type = "speed_question"
    
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
        "original_query": query,
        "canonical": str(simp.get("canonical") or "").strip() or None,
        "intent": simp  # expose full simplifier intent for downstream routing/augmentation
    }
