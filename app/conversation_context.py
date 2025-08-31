"""
Conversation Context Manager for maintaining topic coherence across questions.
Addresses the issue where ambiguous questions get answered from wrong documents.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import re
from app.logger import get_logger

class ConversationContext:
    """Tracks conversation state to maintain topic coherence."""
    
    def __init__(self, context_window_minutes: int = 10):
        self.logger = get_logger()
        self.context_window = timedelta(minutes=context_window_minutes)
        
        # Track document interactions with timestamps
        self.document_history: List[Dict] = []  # [{document, timestamp, confidence}]
        
        # Track topic keywords from recent questions
        self.topic_keywords: Set[str] = set()
        
        # Recent questions for disambiguation
        self.recent_questions: List[Dict] = []  # [{question, timestamp, document}]
    
    def add_interaction(self, document_name: str, question: str, confidence: float = 1.0):
        """Record a document interaction from a question-answer session."""
        now = datetime.now()
        
        # Add to document history
        self.document_history.append({
            'document': document_name,
            'timestamp': now,
            'confidence': confidence,
            'question': question[:100]  # Store snippet for debugging
        })
        
        # Add to recent questions
        self.recent_questions.append({
            'question': question,
            'timestamp': now,
            'document': document_name
        })
        
        # Extract and store topic keywords
        self._extract_topic_keywords(question)
        
        # Clean old entries
        self._cleanup_old_entries()
        
        self.logger.debug(f"Added interaction: {document_name} - '{question[:50]}...' (confidence: {confidence})")
    
    def get_preferred_document(self) -> Optional[str]:
        """Get the most likely document the user is discussing."""
        if not self.document_history:
            return None
        
        # Score documents by recency and frequency
        now = datetime.now()
        document_scores = {}
        
        for interaction in self.document_history:
            doc = interaction['document']
            age_minutes = (now - interaction['timestamp']).total_seconds() / 60
            confidence = interaction['confidence']
            
            # Decay score by age
            age_factor = max(0, 1 - (age_minutes / self.context_window.total_seconds() * 60))
            score = confidence * age_factor
            
            document_scores[doc] = document_scores.get(doc, 0) + score
        
        if not document_scores:
            return None
        
        # Return document with highest score
        preferred = max(document_scores.items(), key=lambda x: x[1])
        self.logger.debug(f"Preferred document: {preferred[0]} (score: {preferred[1]:.2f})")
        return preferred[0]
    
    def detect_ambiguous_query(self, question: str) -> Dict:
        """Detect if a query is ambiguous and might need disambiguation."""
        # Patterns that indicate ambiguous queries
        ambiguous_patterns = [
            r'\b(how much|what was|what is|how did|when did)\b.*\b(change|differ|vary)\b',
            r'\b(the|this|that)\s+(test|experiment|measurement|value|result)\b',
            r'\b(between|across|during)\s+(test|experiment|trial)s?\b',
            r'\b(rms|vibration|frequency|temperature)\b.*\b(change|increase|decrease)\b',
            r'\bthe\s+(rms|vibration|frequency|temperature)\b.*\b(between|during)\b'
        ]
        
        is_ambiguous = any(re.search(pattern, question.lower()) for pattern in ambiguous_patterns)
        
        # Check if query lacks specific document/context indicators
        specific_indicators = [
            r'\b(gear|bearing|sliding|journal)\s+(wear|test|failure|document|report)\b',
            r'\b(figure|table|page)\s+\d+\b',
            r'\b[A-Z]\d+\b',  # Case IDs like W26
            r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
            r'\b(gear|gears|gearbox|tooth|teeth)\b',  # Gear-specific terms
            r'\b(bearing|bearings|sliding|journal)\b'  # Bearing-specific terms
        ]
        
        has_specifics = any(re.search(pattern, question.lower()) for pattern in specific_indicators)
        
        # Get available documents from recent context
        available_docs = set()
        if self.document_history:
            available_docs = {h['document'] for h in self.document_history[-5:]}  # Last 5 interactions
        
        # Only consider it ambiguous if it matches patterns AND lacks specifics AND we have multiple docs in context
        needs_disambiguation = (is_ambiguous and not has_specifics and len(available_docs) > 1)
        
        return {
            'is_ambiguous': is_ambiguous and not has_specifics,
            'has_context': bool(self.document_history),
            'preferred_document': self.get_preferred_document(),
            'available_documents': list(available_docs),
            'needs_disambiguation': needs_disambiguation,
            'confidence': 0.8 if (is_ambiguous and not has_specifics) else 0.2
        }
    
    def generate_disambiguation_prompt(self, question: str, available_documents: List[str]) -> str:
        """Generate a clarification question for ambiguous queries."""
        doc_list = ", ".join(f'"{doc}"' for doc in available_documents[-3:])  # Show last 3
        
        return (f"Your question '{question}' could apply to multiple documents. "
                f"Are you asking about: {doc_list}? "
                f"Please specify which document or say 'all documents' for a comprehensive answer.")
    
    def should_bias_retrieval(self, question: str) -> Optional[str]:
        """Determine if retrieval should be biased toward a specific document."""
        ambiguity_info = self.detect_ambiguous_query(question)
        
        # Always bias if we have context, even for non-ambiguous questions (unless they're very specific)
        if ambiguity_info['has_context']:
            preferred = ambiguity_info['preferred_document']
            if preferred:
                # For ambiguous questions, always bias
                if ambiguity_info['is_ambiguous']:
                    self.logger.info(f"Biasing retrieval toward: {preferred} (ambiguous query)")
                    return preferred
                # For non-ambiguous but general questions, still bias if confidence is high
                elif len(self.document_history) >= 2:  # Multiple recent interactions
                    self.logger.info(f"Biasing retrieval toward: {preferred} (conversation context)")
                    return preferred
        
        return None
    
    def _extract_topic_keywords(self, question: str):
        """Extract and store topic-relevant keywords."""
        # Technical terms relevant to failure analysis
        technical_terms = re.findall(r'\b(rms|vibration|frequency|bearing|gear|failure|wear|fatigue|rpm|temperature|pressure|stress|strain)\b', question.lower())
        self.topic_keywords.update(technical_terms)
        
        # Keep only recent keywords (last 20)
        if len(self.topic_keywords) > 20:
            # This is a simple approach; in production, you'd want to be smarter about keyword aging
            self.topic_keywords = set(list(self.topic_keywords)[-20:])
    
    def _cleanup_old_entries(self):
        """Remove entries older than the context window."""
        now = datetime.now()
        cutoff = now - self.context_window
        
        self.document_history = [h for h in self.document_history if h['timestamp'] > cutoff]
        self.recent_questions = [q for q in self.recent_questions if q['timestamp'] > cutoff]
    
    def get_context_summary(self) -> Dict:
        """Get a summary of current conversation context."""
        return {
            'preferred_document': self.get_preferred_document(),
            'recent_documents': list(set(h['document'] for h in self.document_history[-5:])),
            'topic_keywords': list(self.topic_keywords),
            'interaction_count': len(self.document_history),
            'context_age_minutes': (datetime.now() - self.document_history[0]['timestamp']).total_seconds() / 60 if self.document_history else 0
        }

# Global conversation context instance
conversation_context = ConversationContext()
