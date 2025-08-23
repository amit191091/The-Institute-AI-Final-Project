from typing import Dict, List, Optional, Tuple
import re
from app.logger import get_logger

class EnhancedQuestionAnalyzer:
    """
    Enhanced question analyzer that provides better understanding of question types
    and suggests appropriate evaluation strategies.
    """
    
    def __init__(self):
        self.log = get_logger()
        
    def analyze_question(self, question: str) -> Dict:
        """
        Analyze a question to determine its type and evaluation requirements.
        """
        analysis = {
            'question_type': self._classify_question_type(question),
            'expected_answer_format': self._determine_answer_format(question),
            'evaluation_strategies': self._suggest_evaluation_strategies(question),
            'difficulty_estimate': self._estimate_difficulty(question),
            'key_entities': self._extract_key_entities(question),
            'measurement_units': self._extract_measurement_units(question)
        }
        
        return analysis
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()
        
        # Factual questions
        if any(word in question_lower for word in ['what is', 'what are', 'what was', 'what were']):
            if any(word in question_lower for word in ['ratio', 'number', 'value', 'amount', 'size', 'dimension']):
                return 'factual_numeric'
            elif any(word in question_lower for word in ['type', 'kind', 'model', 'brand', 'name']):
                return 'factual_categorical'
            else:
                return 'factual_general'
        
        # Analytical questions
        elif any(word in question_lower for word in ['how', 'why', 'explain', 'describe', 'analyze']):
            return 'analytical'
        
        # Comparative questions
        elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'between']):
            return 'comparative'
        
        # Temporal questions
        elif any(word in question_lower for word in ['when', 'during', 'after', 'before', 'timeline']):
            return 'temporal'
        
        # Table/Figure questions
        elif any(word in question_lower for word in ['table', 'figure', 'chart', 'graph']):
            return 'table_figure'
        
        else:
            return 'general'
    
    def _determine_answer_format(self, question: str) -> str:
        """Determine the expected format of the answer."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['ratio', 'percentage', 'number', 'value']):
            return 'numeric'
        elif any(word in question_lower for word in ['list', 'all', 'examples']):
            return 'list'
        elif any(word in question_lower for word in ['explain', 'describe', 'how', 'why']):
            return 'explanatory'
        elif any(word in question_lower for word in ['yes', 'no', 'true', 'false']):
            return 'boolean'
        else:
            return 'text'
    
    def _suggest_evaluation_strategies(self, question: str) -> List[str]:
        """Suggest appropriate evaluation strategies based on question type."""
        question_type = self._classify_question_type(question)
        
        strategies = {
            'factual_numeric': ['exact_match', 'number_validation', 'unit_validation', 'range_check'],
            'factual_categorical': ['keyword_matching', 'entity_extraction', 'semantic_similarity'],
            'factual_general': ['keyword_matching', 'semantic_similarity', 'fact_validation'],
            'analytical': ['semantic_similarity', 'fact_extraction', 'completeness_check'],
            'comparative': ['comparison_extraction', 'semantic_similarity', 'fact_validation'],
            'temporal': ['date_extraction', 'timeline_validation', 'semantic_similarity'],
            'table_figure': ['table_extraction', 'figure_analysis', 'data_validation'],
            'general': ['semantic_similarity', 'keyword_matching', 'completeness_check']
        }
        
        return strategies.get(question_type, ['semantic_similarity', 'keyword_matching'])
    
    def _estimate_difficulty(self, question: str) -> str:
        """Estimate the difficulty level of the question."""
        question_lower = question.lower()
        
        # Easy indicators
        easy_indicators = ['what is', 'what are', 'name', 'type', 'model', 'brand']
        if any(indicator in question_lower for indicator in easy_indicators):
            return 'easy'
        
        # Hard indicators
        hard_indicators = ['explain', 'analyze', 'compare', 'why', 'how', 'relationship']
        if any(indicator in question_lower for indicator in hard_indicators):
            return 'hard'
        
        return 'medium'
    
    def _extract_key_entities(self, question: str) -> List[str]:
        """Extract key entities from the question."""
        # Simple entity extraction - can be enhanced with NER
        entities = []
        
        # Extract technical terms
        technical_terms = re.findall(r'\b[A-Z][A-Z0-9-]+\b', question)
        entities.extend(technical_terms)
        
        # Extract measurement terms
        measurement_terms = re.findall(r'\b\d+(?:\.\d+)?\s?(?:mm|μm|MPa|RPM|°C|N|kN|Hz|MPH|kW|mV/g)\b', question)
        entities.extend(measurement_terms)
        
        # Extract gear-related terms
        gear_terms = re.findall(r'\b(?:gear|tooth|wear|transmission|ratio|lubricant|accelerometer|sensor)\b', question.lower())
        entities.extend(gear_terms)
        
        return list(set(entities))
    
    def _extract_measurement_units(self, question: str) -> List[str]:
        """Extract measurement units from the question."""
        units = re.findall(r'\b(?:mm|μm|MPa|RPM|°C|N|kN|Hz|MPH|kW|mV/g)\b', question)
        return list(set(units))
    
    def generate_evaluation_prompt(self, question: str, rag_answer: str, context_docs: List) -> str:
        """Generate a customized evaluation prompt based on question analysis."""
        analysis = self.analyze_question(question)
        
        context_text = "\n\n".join([doc.page_content for doc in context_docs[:2]])
        
        base_prompt = f"""
        Evaluate the following answer to the question based on the provided context.
        
        Question: {question}
        Question Type: {analysis['question_type']}
        Expected Format: {analysis['expected_answer_format']}
        Difficulty: {analysis['difficulty_estimate']}
        Key Entities: {', '.join(analysis['key_entities'])}
        
        Context: {context_text[:1000]}
        Answer: {rag_answer}
        
        Evaluation Criteria:
        """
        
        # Add specific criteria based on question type
        if analysis['question_type'] == 'factual_numeric':
            base_prompt += """
        1. Numeric Accuracy: Are the numbers correct?
        2. Unit Consistency: Are the units appropriate and consistent?
        3. Precision: Is the level of precision appropriate?
        4. Completeness: Is the numeric answer complete?
        """
        elif analysis['question_type'] == 'factual_categorical':
            base_prompt += """
        1. Entity Accuracy: Are the named entities correct?
        2. Categorization: Is the categorization accurate?
        3. Specificity: Is the answer specific enough?
        4. Completeness: Are all relevant categories mentioned?
        """
        elif analysis['question_type'] == 'analytical':
            base_prompt += """
        1. Logical Reasoning: Is the reasoning sound?
        2. Evidence Support: Is the analysis supported by evidence?
        3. Completeness: Is the analysis comprehensive?
        4. Clarity: Is the explanation clear and understandable?
        """
        else:
            base_prompt += """
        1. Relevance: Does the answer address the question?
        2. Accuracy: Is the information accurate?
        3. Completeness: Is the answer complete?
        4. Clarity: Is the answer clear and understandable?
        """
        
        base_prompt += """
        
        Provide a score between 0 and 1 for each criterion, then calculate an overall score.
        Format your response as: criterion1: score, criterion2: score, ..., overall: score
        """
        
        return base_prompt