#!/usr/bin/env python3
"""
Fact Miner Module
================

Provides deterministic fact mining and answer canonicalization capabilities.
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from langchain.schema import Document


def mine_answer_from_context(question: str, docs: List[Document]) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Mine a deterministic answer from context using lexical overlap.
    
    Args:
        question: The question to answer
        docs: List of relevant documents
        
    Returns:
        Tuple of (answer, metadata) or (None, {}) if no answer found
    """
    try:
        # Extract key terms from question
        q_terms = set(re.findall(r'\b\w{3,}\b', question.lower()))
        
        # Concatenate all document content
        all_content = " ".join([d.page_content for d in docs if d.page_content])
        
        # Find sentences containing question terms
        sentences = re.split(r'[.!?]+', all_content)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_terms = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
            overlap = len(q_terms.intersection(sentence_terms))
            if overlap >= 2:  # At least 2 terms overlap
                relevant_sentences.append((sentence.strip(), overlap))
        
        if not relevant_sentences:
            return None, {}
        
        # Sort by overlap and take the best
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentence = relevant_sentences[0][0]
        
        # Clean up the sentence
        answer = re.sub(r'\s+', ' ', best_sentence).strip()
        
        metadata = {
            "method": "lexical_overlap",
            "overlap_score": relevant_sentences[0][1],
            "total_sentences": len(sentences),
            "relevant_sentences": len(relevant_sentences)
        }
        
        return answer, metadata
        
    except Exception as e:
        return None, {"error": str(e)}


def canonicalize_answer(question: str, answer: str) -> str:
    """
    Canonicalize answer phrasing to reduce drift when equivalent.
    
    Args:
        question: The original question
        answer: The answer to canonicalize
        
    Returns:
        Canonicalized answer
    """
    try:
        if not answer:
            return answer
        
        # Remove common prefixes that don't add value
        prefixes_to_remove = [
            "Based on the information provided,",
            "According to the document,",
            "The document states that",
            "Based on the data,",
            "The information shows that",
            "From the provided information,",
            "The document indicates that"
        ]
        
        canonicalized = answer
        for prefix in prefixes_to_remove:
            if canonicalized.lower().startswith(prefix.lower()):
                canonicalized = canonicalized[len(prefix):].strip()
                break
        
        # Ensure proper sentence structure
        if canonicalized and not canonicalized.endswith(('.', '!', '?')):
            canonicalized = canonicalized.rstrip('.') + '.'
        
        # Remove excessive whitespace
        canonicalized = re.sub(r'\s+', ' ', canonicalized).strip()
        
        return canonicalized
        
    except Exception:
        return answer
