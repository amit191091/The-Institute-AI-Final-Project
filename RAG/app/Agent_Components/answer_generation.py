"""
Answer generation module for the RAG system.
Handles generating answers from context documents and formatting responses.
"""

from typing import List, Dict, Any
from langchain.schema import Document

from RAG.app.interfaces import AnswerGenerationInterface
from RAG.app.config import settings
from RAG.app.logger import get_logger
from RAG.app.pipeline_modules.pipeline_utils import LLM
from RAG.app.Performance_Optimization.caching import cache_llm_response
from RAG.app.Evaluation_Analysis.progress_tracking import monitor_performance


class AnswerGenerationService(AnswerGenerationInterface):
    """Service for generating answers from context documents."""
    
    def __init__(self, config=None, llm=None):
        self.config = config or settings
        self.logger = get_logger()
        self.llm = llm or LLM()
    
    def generate_answer(self, query: str, context_documents: List[Document]) -> str:
        """Generate an answer based on query and context documents."""
        try:
            if not context_documents:
                return "I don't have enough information to answer this question."
            
            # Prepare context from documents
            context = self._prepare_context(context_documents)
            
            # Generate answer using LLM
            answer = self._generate_with_llm(query, context)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def format_answer(self, answer: str, sources: List[Document]) -> Dict[str, Any]:
        """Format the answer with source information."""
        try:
            # Extract source information
            source_info = self._extract_source_info(sources)
            
            # Format the response
            formatted_response = {
                "answer": answer,
                "sources": source_info,
                "num_sources": len(sources),
                "confidence": self._calculate_confidence(sources)
            }
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error formatting answer: {e}")
            return {
                "answer": answer,
                "sources": [],
                "num_sources": 0,
                "confidence": "low"
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from documents."""
        try:
            context_parts = []
            
            for i, doc in enumerate(documents, 1):
                # Extract relevant information
                content = doc.page_content or ""
                metadata = doc.metadata or {}
                
                # Create context entry
                context_entry = f"Source {i}:\n"
                
                # Add metadata if available
                if metadata.get("file_name"):
                    context_entry += f"File: {metadata['file_name']}\n"
                if metadata.get("page"):
                    context_entry += f"Page: {metadata['page']}\n"
                if metadata.get("section"):
                    context_entry += f"Section: {metadata['section']}\n"
                
                # Add content (truncated if too long)
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                context_entry += f"Content: {content}\n"
                context_parts.append(context_entry)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error preparing context: {e}")
            return "Context preparation failed."
    
    @cache_llm_response(ttl=3600)
    @monitor_performance
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate answer using the language model."""
        try:
            # Create prompt for the LLM
            prompt = self._create_prompt(query, context)
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating with LLM: {e}")
            return "I couldn't generate a response. Please try again."
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the language model."""
        return f"""You are a helpful assistant that answers questions based on the provided context. 
Please answer the following question using only the information provided in the context.

Question: {query}

Context:
{context}

Please provide a clear, concise answer based on the context. If the context doesn't contain enough information to answer the question, say so. Do not make up information that is not in the context.

Answer:"""
    
    def _extract_source_info(self, sources: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        try:
            source_info = []
            
            for doc in sources:
                metadata = doc.metadata or {}
                
                source = {
                    "file_name": metadata.get("file_name", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                    "content_preview": self._get_content_preview(doc.page_content or "")
                }
                
                source_info.append(source)
            
            return source_info
            
        except Exception as e:
            self.logger.error(f"Error extracting source info: {e}")
            return []
    
    def _get_content_preview(self, content: str, max_length: int = 200) -> str:
        """Get a preview of the content."""
        if len(content) <= max_length:
            return content
        
        # Try to break at a word boundary
        preview = content[:max_length]
        last_space = preview.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can break at a reasonable word boundary
            return preview[:last_space] + "..."
        else:
            return preview + "..."
    
    def _calculate_confidence(self, sources: List[Document]) -> str:
        """Calculate confidence level based on sources."""
        try:
            if not sources:
                return "low"
            
            # Simple confidence calculation based on number and quality of sources
            num_sources = len(sources)
            
            # Check source quality (has content, metadata, etc.)
            quality_sources = 0
            for doc in sources:
                if doc.page_content and len(doc.page_content.strip()) > 50:
                    quality_sources += 1
            
            # Calculate confidence
            if num_sources >= 3 and quality_sources >= 2:
                return "high"
            elif num_sources >= 2 and quality_sources >= 1:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return "low"
    
    def generate_structured_answer(self, query: str, context_documents: List[Document]) -> Dict[str, Any]:
        """Generate a structured answer with metadata."""
        try:
            # Generate the answer
            answer = self.generate_answer(query, context_documents)
            
            # Format with sources
            formatted = self.format_answer(answer, context_documents)
            
            # Add query information
            formatted["query"] = query
            formatted["query_type"] = self._classify_query(query)
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error generating structured answer: {e}")
            return {
                "answer": "Error generating answer",
                "query": query,
                "sources": [],
                "num_sources": 0,
                "confidence": "low",
                "query_type": "unknown"
            }
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            return "information"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["list", "all", "every"]):
            return "enumeration"
        elif any(word in query_lower for word in ["table", "figure", "data"]):
            return "data_request"
        else:
            return "general"


class AnswerGenerationFactory:
    """Factory for creating answer generation services."""
    
    @staticmethod
    def create_service(config=None, llm=None) -> AnswerGenerationService:
        """Create an answer generation service with the given configuration."""
        return AnswerGenerationService(config, llm)
