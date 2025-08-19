"""
Metadata extraction and schema definition for failure analysis documents
"""
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from collections import Counter
import logging

from .utils import extract_numeric_values, clean_text

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Metadata extraction class for failure analysis documents
    """
    
    def __init__(self):
        self.stop_words = STOP_WORDS
    
    def extract_metadata(self, text: str, source_file: str = "", chunk_id: str = "") -> Dict[str, Any]:
        """
        Extract comprehensive metadata from text chunk
        
        Args:
            text: Text content to analyze
            source_file: Source file path
            chunk_id: Unique chunk identifier
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            # Basic metadata
            metadata = {
                "file_name": source_file,
                "chunk_id": chunk_id,
                "section_type": "Text",  # Default
                "chunk_summary": self._generate_summary(text),
                "keywords": extract_keywords(text),
                "entities": extract_entities(text),
                "char_count": len(text),
                "word_count": len(text.split()),
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            # Extract domain-specific information
            incident_info = extract_incident_info(text)
            metadata.update(incident_info)
            
            # Extract numeric values
            numeric_values = extract_numeric_values(text)
            if numeric_values:
                metadata["numeric_values"] = numeric_values
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {
                "file_name": source_file,
                "chunk_id": chunk_id,
                "section_type": "Text",
                "error": str(e)
            }
    
    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a brief summary of the text"""
        cleaned = clean_text(text)
        if len(cleaned) <= max_length:
            return cleaned
        
        # Take first sentence or first max_length characters
        sentences = cleaned.split('.')
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0].strip() + '.'
        
        return cleaned[:max_length].strip() + '...'

# Section type enumeration as per specifications
SECTION_ENUM = {"Summary", "Timeline", "Table", "Figure", "Analysis", "Conclusion", "Text", "Appendix"}

# Common stop words for keyword extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your'
}

# Gear and bearing related keywords for prioritization
DOMAIN_KEYWORDS = {
    'gear', 'bearing', 'tooth', 'teeth', 'wear', 'failure', 'fatigue', 'crack', 'fracture',
    'vibration', 'noise', 'lubrication', 'temperature', 'load', 'stress', 'strain', 'rpm',
    'torque', 'shaft', 'housing', 'seal', 'grease', 'oil', 'maintenance', 'inspection',
    'analysis', 'diagnosis', 'condition', 'monitoring', 'deterioration', 'damage', 'defect'
}

def classify_section_type(kind: str, text: str) -> str:
    """
    Classify section type based on element kind and text content
    
    Args:
        kind: Element category from document parser
        text: Text content of the element
        
    Returns:
        Section type from SECTION_ENUM
    """
    k = (kind or "").lower()
    t = (text or "").lower()
    
    # Direct mappings
    if k == "table":
        return "Table"
    if k in ("figure", "image"):
        return "Figure"
    
    # Text-based classification
    if any(word in t for word in ["timeline", "chronology", "sequence", "history"]):
        return "Timeline"
    
    if any(word in t for word in ["conclusion", "summary", "abstract", "overview"]):
        if "conclusion" in t:
            return "Conclusion"
        return "Summary"
    
    if any(word in t for word in ["analysis", "method", "procedure", "results", "discussion", "findings"]):
        return "Analysis"
    
    if any(word in t for word in ["appendix", "attachment", "supplement"]):
        return "Appendix"
    
    return "Text"

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract keywords using frequency analysis with domain awareness
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Clean and tokenize
    text_clean = clean_text(text.lower())
    words = re.findall(r'\\b[a-z0-9]+\\b', text_clean)
    
    # Filter out stop words and short words
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    if not words:
        return []
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Boost domain-specific keywords
    for word in word_freq:
        if word in DOMAIN_KEYWORDS:
            word_freq[word] *= 2
    
    # Get top keywords
    top_words = [word for word, _ in word_freq.most_common(top_n)]
    
    return top_words

def extract_entities(text: str) -> List[str]:
    """
    Extract entities relevant to bearing/gear failure analysis
    
    Args:
        text: Input text
        
    Returns:
        List of extracted entities
    """
    if not text:
        return []
    
    entities = []
    
    # Patterns for different entity types
    patterns = [
        # Case/Client IDs
        (r'\\b(?:CASE|case)[-_ ]?\\d+\\b', "case_id"),
        (r'\\b(?:CLIENT|client)[-_ ]?\\w+\\b', "client_id"),
        
        # Component IDs
        (r'\\b(?:BRG|brg|bearing)[-_ ]?\\w+\\b', "bearing_id"),
        (r'\\b(?:GEAR|gear)[-_ ]?\\w+\\b', "gear_id"),
        (r'\\b(?:SHAFT|shaft)[-_ ]?\\w+\\b', "shaft_id"),
        
        # Model numbers
        (r'\\b[A-Z]{2,4}[-]?\\d{3,6}[A-Z]?\\b', "model_number"),
        
        # Dates
        (r'\\b\\d{4}-\\d{2}-\\d{2}\\b', "date"),
        (r'\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b', "date"),
        
        # Measurements with units
        (r'\\b\\d+(?:\\.\\d+)?\\s?(?:μm|um|mm|cm|m|MPa|GPa|RPM|Hz|°C|N|kN|kW)\\b', "measurement"),
        
        # Failure modes
        (r'\\b(?:fatigue|wear|crack|fracture|seizure|overheat|misalignment)\\b', "failure_mode"),
    ]
    
    for pattern, entity_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity_value = match.group(0)
            entities.append(f"{entity_type}:{entity_value}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    return unique_entities[:20]  # Limit to top 20 entities

def extract_incident_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extract incident-related information (type, date, severity)
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with incident information
    """
    incident_info: Dict[str, Optional[str]] = {
        "incident_type": None,
        "incident_date": None,
        "amount_range": None,
        "severity": None
    }
    
    text_lower = text.lower()
    
    # Incident type detection
    failure_patterns = {
        "wear": r'\\b(?:wear|wearing|worn)\\b',
        "fatigue": r'\\b(?:fatigue|fatigue)\\b',
        "fracture": r'\\b(?:fracture|fractured|break|broken)\\b',
        "seizure": r'\\b(?:seiz|seized|seizure)\\b',
        "overheat": r'\\b(?:overheat|overheating|thermal)\\b',
        "corrosion": r'\\b(?:corros|rust|oxidat)\\b',
        "misalignment": r'\\b(?:misalign|misalignment)\\b'
    }
    
    for failure_type, pattern in failure_patterns.items():
        if re.search(pattern, text_lower):
            incident_info["incident_type"] = failure_type
            break
    
    # Date extraction
    date_patterns = [
        r'\\b(20\\d{2})[-/](\\d{1,2})[-/](\\d{1,2})\\b',  # YYYY-MM-DD or YYYY/MM/DD
        r'\\b(\\d{1,2})[-/](\\d{1,2})[-/](20\\d{2})\\b',  # MM-DD-YYYY or MM/DD/YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.group(1)) == 4:  # YYYY first
                date_str = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
            else:  # MM first
                date_str = f"{match.group(3)}-{match.group(1).zfill(2)}-{match.group(2).zfill(2)}"
            incident_info["incident_date"] = date_str
            break
    
    # Amount/measurement range extraction
    numeric_values = extract_numeric_values(text)
    if numeric_values:
        # Group by unit type
        measurements_by_unit = {}
        for value, unit, _, _ in numeric_values:
            if unit not in measurements_by_unit:
                measurements_by_unit[unit] = []
            measurements_by_unit[unit].append(value)
        
        # Create range for most common unit
        if measurements_by_unit:
            most_common_unit = max(measurements_by_unit.keys(), key=lambda k: len(measurements_by_unit[k]))
            values = measurements_by_unit[most_common_unit]
            if len(values) > 1:
                min_val, max_val = min(values), max(values)
                incident_info["amount_range"] = f"{min_val}-{max_val} {most_common_unit}"
            else:
                incident_info["amount_range"] = f"{values[0]} {most_common_unit}"
    
    # Severity assessment based on keywords
    severity_indicators = {
        "critical": ["critical", "severe", "catastrophic", "immediate", "emergency"],
        "high": ["significant", "major", "serious", "urgent", "high"],
        "medium": ["moderate", "noticeable", "medium", "intermediate"],
        "low": ["minor", "slight", "low", "minimal"]
    }
    
    for severity, keywords in severity_indicators.items():
        if any(keyword in text_lower for keyword in keywords):
            incident_info["severity"] = severity
            break
    
    return incident_info

def attach_metadata(chunk: Dict[str, Any], client_id: Optional[str] = None, case_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Attach comprehensive metadata to a document chunk
    
    Args:
        chunk: Document chunk with basic information
        client_id: Client identifier
        case_id: Case identifier
        
    Returns:
        Document with full metadata schema
    """
    content = chunk.get("content", "")
    
    # Extract all metadata components
    entities = extract_entities(content)
    incident_info = extract_incident_info(content)
    keywords = extract_keywords(content)
    
    # Build comprehensive metadata (≥5 fields as required)
    metadata = {
        # Required fields (always present)
        "file_name": chunk.get("file_name", "unknown"),
        "page": chunk.get("page", 1),
        "section": chunk.get("section_type", "Text"),
        "chunk_summary": content.split('.')[0][:200] if content else "",
        "keywords": keywords,
        
        # Anchoring information
        "anchor": chunk.get("anchor"),
        "table_row_range": chunk.get("table_row_range"),
        "table_col_names": chunk.get("table_col_names"),
        
        # Identification metadata
        "client_id": client_id,
        "case_id": case_id,
        
        # Content analysis metadata
        "critical_entities": entities,
        "incident_type": incident_info["incident_type"],
        "incident_date": incident_info["incident_date"],
        "amount_range": incident_info["amount_range"],
        "severity": incident_info["severity"],
        
        # Processing metadata
        "processed_at": datetime.now().isoformat(),
        "content_length": len(content),
        "token_count": len(content.split()),  # Rough approximation
        
        # Domain-specific metadata
        "has_measurements": bool(extract_numeric_values(content)),
        "is_table": chunk.get("section_type") == "Table",
        "is_figure": chunk.get("section_type") == "Figure",
        "domain_relevance": len([k for k in keywords if k in DOMAIN_KEYWORDS]) / max(len(keywords), 1)
    }
    
    return {
        "page_content": content,
        "metadata": metadata
    }

def validate_metadata_completeness(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that metadata meets the ≥5 fields requirement
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    required_fields = ["file_name", "page", "section", "chunk_summary", "keywords"]
    
    missing_fields = []
    for field in required_fields:
        if field not in metadata or metadata[field] is None:
            missing_fields.append(field)
    
    # Count non-null fields
    non_null_fields = sum(1 for v in metadata.values() if v is not None)
    
    is_valid = len(missing_fields) == 0 and non_null_fields >= 5
    
    return is_valid, missing_fields
