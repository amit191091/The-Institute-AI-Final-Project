from typing import Dict, List, Optional, Tuple
import json
import re
from pathlib import Path
from app.logger import get_logger

class AutoEvaluator:
    """
    Automatic evaluation system that can handle questions not in ground truth dataset.
    Uses multiple strategies to generate evaluation scores.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.log = get_logger()
        
    def evaluate_question(self, question: str, rag_answer: str, context_docs: List, 
                         existing_ground_truth: Optional[str] = None) -> Dict:
        """
        Evaluate a question using multiple strategies:
        1. Use existing ground truth if available
        2. Generate synthetic ground truth using LLM
        3. Use semantic similarity scoring
        4. Use fact extraction and validation
        """
        
        scores = {}
        
        # Strategy 1: Use existing ground truth if available
        if existing_ground_truth:
            scores.update(self._evaluate_with_ground_truth(rag_answer, existing_ground_truth))
        
        # Strategy 2: Generate synthetic ground truth
        synthetic_gt = self._generate_synthetic_ground_truth(question, context_docs)
        if synthetic_gt:
            scores.update(self._evaluate_with_ground_truth(rag_answer, synthetic_gt))
            scores['synthetic_gt_used'] = True
        else:
            scores['synthetic_gt_used'] = False
        
        # Strategy 3: Semantic similarity scoring
        scores.update(self._semantic_similarity_scoring(question, rag_answer, context_docs))
        
        # Strategy 4: Fact extraction and validation
        scores.update(self._fact_extraction_scoring(question, rag_answer, context_docs))
        
        # Strategy 5: Answer completeness scoring
        scores.update(self._completeness_scoring(question, rag_answer))
        
        return scores
    
    def _generate_synthetic_ground_truth(self, question: str, context_docs: List) -> Optional[str]:
        """Generate a synthetic ground truth answer using the LLM."""
        try:
            context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])
            
            prompt = f"""
            Based on the following context, provide the most accurate and complete answer to the question.
            Focus on factual accuracy and completeness.
            
            Context:
            {context_text}
            
            Question: {question}
            
            Provide only the answer, no explanations:
            """
            
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Failed to generate synthetic ground truth: {e}"
    
    def _evaluate_with_ground_truth(self, rag_answer: str, ground_truth: str) -> Dict:
        """Evaluate answer against ground truth using multiple metrics."""
        scores = {}
        
        # Simple keyword matching
        rag_words = set(re.findall(r'\b\w+\b', rag_answer.lower()))
        gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        
        if gt_words:
            keyword_overlap = len(rag_words.intersection(gt_words)) / len(gt_words)
            scores['keyword_precision'] = keyword_overlap
            scores['keyword_recall'] = len(rag_words.intersection(gt_words)) / len(rag_words) if rag_words else 0
        
        # Number extraction and comparison
        rag_numbers = re.findall(r'\d+(?:\.\d+)?', rag_answer)
        gt_numbers = re.findall(r'\d+(?:\.\d+)?', ground_truth)
        
        if gt_numbers:
            number_accuracy = len(set(rag_numbers).intersection(set(gt_numbers))) / len(set(gt_numbers))
            scores['number_accuracy'] = number_accuracy
        
        # Length-based scoring
        if ground_truth:
            length_ratio = len(rag_answer) / len(ground_truth) if len(ground_truth) > 0 else 0
            scores['length_ratio'] = min(length_ratio, 2.0) / 2.0  # Normalize to 0-1
        
        return scores
    
    def _semantic_similarity_scoring(self, question: str, rag_answer: str, context_docs: List) -> Dict:
        """Score based on semantic similarity between answer and context."""
        scores = {}
        
        try:
            # Use LLM to evaluate semantic relevance
            context_text = "\n\n".join([doc.page_content for doc in context_docs[:2]])
            
            prompt = f"""
            Rate the relevance and accuracy of the answer to the question on a scale of 0-1.
            Consider:
            1. Does the answer directly address the question?
            2. Is the information accurate based on the context?
            3. Is the answer complete and specific?
            
            Question: {question}
            Context: {context_text[:1000]}
            Answer: {rag_answer}
            
            Provide only a number between 0 and 1:
            """
            
            relevance_score = self.llm(prompt).strip()
            try:
                scores['semantic_relevance'] = float(relevance_score)
            except ValueError:
                scores['semantic_relevance'] = 0.5
                
        except Exception as e:
            scores['semantic_relevance'] = 0.5
        
        return scores
    
    def _fact_extraction_scoring(self, question: str, rag_answer: str, context_docs: List) -> Dict:
        """Score based on fact extraction and validation."""
        scores = {}
        
        try:
            # Extract key facts from the answer
            fact_prompt = f"""
            Extract the key factual information from this answer. List only the main facts:
            
            Answer: {rag_answer}
            
            Facts:
            """
            
            extracted_facts = self.llm(fact_prompt).strip()
            
            # Validate facts against context
            context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])
            
            validation_prompt = f"""
            For each fact, indicate if it's supported by the context (1) or not (0):
            
            Context: {context_text[:1500]}
            
            Facts to validate:
            {extracted_facts}
            
            Provide only 1 or 0 for each fact, separated by commas:
            """
            
            validation_result = self.llm(validation_prompt).strip()
            
            try:
                validations = [int(x.strip()) for x in validation_result.split(',') if x.strip().isdigit()]
                if validations:
                    scores['fact_accuracy'] = sum(validations) / len(validations)
                else:
                    scores['fact_accuracy'] = 0.5
            except ValueError:
                scores['fact_accuracy'] = 0.5
                
        except Exception as e:
            scores['fact_accuracy'] = 0.5
        
        return scores
    
    def _completeness_scoring(self, question: str, rag_answer: str) -> Dict:
        """Score based on answer completeness."""
        scores = {}
        
        # Check for common completeness indicators
        completeness_indicators = [
            'not found', 'not provided', 'cannot find', 'not available', 
            'no information', 'not mentioned', 'does not contain', 'cannot provide'
        ]
        
        has_negative_indicators = any(indicator in rag_answer.lower() for indicator in completeness_indicators)
        
        if has_negative_indicators:
            scores['completeness'] = 0.0
        else:
            # Score based on answer length and specificity
            if len(rag_answer) < 20:
                scores['completeness'] = 0.3
            elif len(rag_answer) < 50:
                scores['completeness'] = 0.6
            else:
                scores['completeness'] = 0.9
        
        # Check for citations/references
        if any(word in rag_answer.lower() for word in ['[', ']', 'page', 'table', 'figure']):
            scores['citation_present'] = 1.0
        else:
            scores['citation_present'] = 0.0
        
        return scores
    
    def calculate_overall_score(self, scores: Dict) -> float:
        """Calculate an overall evaluation score from individual metrics."""
        weights = {
            'keyword_precision': 0.2,
            'keyword_recall': 0.2,
            'number_accuracy': 0.15,
            'semantic_relevance': 0.2,
            'fact_accuracy': 0.15,
            'completeness': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores and scores[metric] is not None:
                overall_score += scores[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return overall_score / total_weight
        else:
            return 0.5  # Default score if no metrics available

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Simple lexical overlap as a fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
        except Exception as e:
            return 0.5  # Default score on error