"""
Utility functions for text processing, summarization, and token management
"""
import re
import tiktoken
from typing import List, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def get_encoding(model: str = "gpt-4") -> tiktoken.Encoding:
    """Get tiktoken encoding for token counting"""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for most modern models
        return tiktoken.get_encoding("cl100k_base")

def approx_token_len(text: str, model: str = "gpt-4") -> int:
    """Approximate token length of text"""
    try:
        encoding = get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit"""
    try:
        encoding = get_encoding(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception:
        # Fallback: character-based truncation
        estimated_chars = max_tokens * 4
        return text[:estimated_chars]

def simple_summarize(text: str, ratio: float = 0.05, min_length: int = 50) -> str:
    """
    Simple extractive summarization by selecting important sentences
    
    Args:
        text: Input text to summarize
        ratio: Proportion of original text to keep (0.05 = 5%)
        min_length: Minimum length of summary
    """
    if not text or not text.strip():
        return text
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return text
    
    # Calculate target number of sentences
    target_sentences = max(1, int(len(sentences) * ratio))
    
    # Score sentences by keyword frequency
    words = re.findall(r'\\b\\w+\\b', text.lower())
    word_freq = Counter(words)
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Score sentences
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words_in_sentence = re.findall(r'\\b\\w+\\b', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words_in_sentence if word not in stop_words)
        # Boost score for longer sentences (more informative)
        score *= len(words_in_sentence) / 10
        sentence_scores.append((score, i, sentence))
    
    # Select top sentences
    sentence_scores.sort(reverse=True)
    selected_indices = sorted([idx for _, idx, _ in sentence_scores[:target_sentences]])
    
    # Reconstruct summary maintaining order
    summary_sentences = [sentences[i] for i in selected_indices]
    summary = '. '.join(summary_sentences)
    
    # Ensure minimum length
    if len(summary) < min_length and len(sentences) > target_sentences:
        # Add more sentences if summary is too short
        additional_sentences = min(3, len(sentences) - target_sentences)
        additional_indices = [idx for _, idx, _ in sentence_scores[target_sentences:target_sentences + additional_sentences]]
        all_indices = sorted(selected_indices + additional_indices)
        summary_sentences = [sentences[i] for i in all_indices]
        summary = '. '.join(summary_sentences)
    
    return summary.strip()

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    return text.strip()

def extract_numeric_values(text: str) -> List[tuple]:
    """Extract numeric values with units from text"""
    patterns = [
        r'(\d+(?:\.\d+)?)\s?(μm|um|microns?|mm|cm|m)',  # Length units
        r'(\d+(?:\.\d+)?)\s?(MPa|GPa|Pa|psi)',           # Pressure units
        r'(\d+(?:\.\d+)?)\s?(RPM|rpm|Hz|kHz)',          # Frequency units
        r'(\d+(?:\.\d+)?)\s?(°C|C|°F|F|K)',             # Temperature units
        r'(\d+(?:\.\d+)?)\s?(N|kN|lbf|kg)',             # Force/Weight units
        r'(\d+(?:\.\d+)?)\s?(kW|W|HP|hp)',              # Power units
    ]
    
    results = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2)
            results.append((value, unit.lower(), match.start(), match.end()))
    
    return results

def format_context_for_llm(chunks: List[dict], query: str) -> str:
    """Format retrieved chunks for LLM consumption"""
    formatted_contexts = []
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        content = chunk.get('page_content', chunk.get('content', ''))
        
        # Format metadata information
        meta_info = []
        if metadata.get('file_name'):
            meta_info.append(f"Document: {metadata['file_name']}")
        if metadata.get('page'):
            meta_info.append(f"Page: {metadata['page']}")
        if metadata.get('section'):
            meta_info.append(f"Section: {metadata['section']}")
        
        meta_str = " | ".join(meta_info) if meta_info else "Source information unavailable"
        
        formatted_chunk = f"""Context {i}:
Source: {meta_str}
Content: {content.strip()}
---"""
        
        formatted_contexts.append(formatted_chunk)
    
    return "\\n\\n".join(formatted_contexts)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(re.findall(r'\\b\\w+\\b', text1.lower()))
    words2 = set(re.findall(r'\\b\\w+\\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def validate_query(query: str) -> bool:
    """
    Validate a user query
    
    Args:
        query: User query string
        
    Returns:
        bool: True if query is valid, False otherwise
    """
    if not query or not isinstance(query, str):
        return False
        
    # Strip whitespace
    query = query.strip()
    
    # Check minimum length
    if len(query) < 3:
        return False
        
    # Check maximum length (avoid very long queries)
    if len(query) > 1000:
        return False
        
    # Check for only special characters
    if not re.search(r'[a-zA-Z0-9]', query):
        return False
        
    # Check for reasonable character content
    alphanumeric_ratio = len(re.findall(r'[a-zA-Z0-9\s]', query)) / len(query)
    if alphanumeric_ratio < 0.5:  # At least 50% should be alphanumeric/spaces
        return False
        
    return True
