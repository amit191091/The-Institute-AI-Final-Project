"""
Multi-agent system with router and specialized agents
"""
import re
from typing import Dict, List, Any, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.docstore.document import Document

from .retrieve import HybridRetriever
from .prompts import (
    ROUTER_PROMPT, SUMMARY_AGENT_PROMPT, NEEDLE_AGENT_PROMPT, 
    TABLE_AGENT_PROMPT, GEAR_INTEGRATION_PROMPT, GENERAL_RAG_PROMPT
)
from .utils import format_context_for_llm
from .config import settings

logger = logging.getLogger(__name__)

class RouterAgent:
    """
    Router agent that analyzes queries and directs them to appropriate specialist agents
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1
        )
    
    def route_query(self, query: str, context_summary: str = "") -> Dict[str, str]:
        """
        Route query to appropriate agent
        
        Args:
            query: User query
            context_summary: Brief summary of available context
            
        Returns:
            Dictionary with chosen agent and reasoning
        """
        try:
            prompt = ROUTER_PROMPT.format(
                query=query,
                context_summary=context_summary or "Technical failure analysis documents available"
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Parse response
            agent_match = re.search(r"AGENT:\\s*([A-Z_]+)", content)
            reason_match = re.search(r"REASON:\\s*(.+)", content, re.DOTALL)
            
            chosen_agent = agent_match.group(1) if agent_match else "SUMMARY_AGENT"
            reason = reason_match.group(1).strip() if reason_match else "Default routing"
            
            return {
                "agent": chosen_agent,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Router agent failed: {e}")
            return {
                "agent": "SUMMARY_AGENT", 
                "reason": "Fallback due to routing error"
            }

class SummaryAgent:
    """
    Agent specialized in providing comprehensive summaries and overviews
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.3
        )
    
    def generate_summary(self, query: str, contexts: List[Document]) -> Dict[str, Any]:
        """
        Generate comprehensive summary response
        
        Args:
            query: User query
            contexts: Retrieved document contexts
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            formatted_contexts = format_context_for_llm(contexts, query)
            
            prompt = SUMMARY_AGENT_PROMPT.format(
                query=query,
                contexts=formatted_contexts
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "response": response.content,
                "agent_type": "summary",
                "sources_used": len(contexts),
                "context_types": [doc.metadata.get("section", "unknown") for doc in contexts]
            }
            
        except Exception as e:
            logger.error(f"Summary agent failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error while generating the summary: {str(e)}",
                "agent_type": "summary",
                "sources_used": 0,
                "error": str(e)
            }

class NeedleAgent:
    """
    Agent specialized in finding specific, precise information
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1  # Lower temperature for precision
        )
    
    def find_specific_info(self, query: str, contexts: List[Document]) -> Dict[str, Any]:
        """
        Find specific information from contexts
        
        Args:
            query: User query
            contexts: Retrieved document contexts
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            formatted_contexts = format_context_for_llm(contexts, query)
            
            prompt = NEEDLE_AGENT_PROMPT.format(
                query=query,
                contexts=formatted_contexts
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "response": response.content,
                "agent_type": "needle",
                "sources_used": len(contexts),
                "precision_mode": True,
                "context_types": [doc.metadata.get("section", "unknown") for doc in contexts]
            }
            
        except Exception as e:
            logger.error(f"Needle agent failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error while searching for specific information: {str(e)}",
                "agent_type": "needle",
                "sources_used": 0,
                "error": str(e)
            }

class TableAgent:
    """
    Agent specialized in analyzing tables and quantitative data
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1
        )
    
    def analyze_tables(self, query: str, contexts: List[Document]) -> Dict[str, Any]:
        """
        Analyze tables and extract quantitative information
        
        Args:
            query: User query
            contexts: Retrieved document contexts (prioritizing tables)
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Filter contexts to prioritize tables
            table_contexts = [doc for doc in contexts if doc.metadata.get("section") == "Table"]
            other_contexts = [doc for doc in contexts if doc.metadata.get("section") != "Table"]
            
            # Use table contexts first, then others
            prioritized_contexts = table_contexts + other_contexts
            
            formatted_contexts = format_context_for_llm(prioritized_contexts, query)
            
            prompt = TABLE_AGENT_PROMPT.format(
                query=query,
                contexts=formatted_contexts
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "response": response.content,
                "agent_type": "table",
                "sources_used": len(contexts),
                "table_contexts": len(table_contexts),
                "context_types": [doc.metadata.get("section", "unknown") for doc in contexts]
            }
            
        except Exception as e:
            logger.error(f"Table agent failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error while analyzing table data: {str(e)}",
                "agent_type": "table",
                "sources_used": 0,
                "error": str(e)
            }

class GearIntegrationAgent:
    """
    Special agent for integrating document analysis with live gear analysis system
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.2
        )
    
    def integrate_analysis(self, query: str, document_results: str, 
                          picture_analysis: Optional[Dict] = None,
                          vibration_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Integrate document analysis with live system analysis
        
        Args:
            query: User query
            document_results: Results from document RAG analysis
            picture_analysis: Results from gear image analysis
            vibration_analysis: Results from vibration analysis
            
        Returns:
            Dictionary with integrated response
        """
        try:
            prompt = GEAR_INTEGRATION_PROMPT.format(
                query=query,
                document_results=document_results,
                picture_analysis=str(picture_analysis) if picture_analysis else "Not available",
                vibration_analysis=str(vibration_analysis) if vibration_analysis else "Not available"
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "response": response.content,
                "agent_type": "integration",
                "has_picture_data": picture_analysis is not None,
                "has_vibration_data": vibration_analysis is not None,
                "integration_mode": True
            }
            
        except Exception as e:
            logger.error(f"Integration agent failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error during integration analysis: {str(e)}",
                "agent_type": "integration",
                "error": str(e)
            }

class MultiAgentSystem:
    """
    Coordinates the multi-agent system for RAG queries
    """
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.router = RouterAgent()
        self.summary_agent = SummaryAgent()
        self.needle_agent = NeedleAgent()
        self.table_agent = TableAgent()
        self.integration_agent = GearIntegrationAgent()
        
        # Fallback general agent
        self.general_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.2
        )
    
    def process_query(self, query: str, context_k: int = None) -> Dict[str, Any]:
        """
        Process query through the multi-agent system
        
        Args:
            query: User query
            context_k: Number of contexts to retrieve
            
        Returns:
            Dictionary with response and processing metadata
        """
        context_k = context_k or settings.CONTEXT_TOP_N
        
        try:
            # Step 1: Retrieve relevant contexts
            logger.info(f"Retrieving contexts for query: {query[:100]}...")
            contexts = self.retriever.retrieve(query, k_final=context_k)
            
            if not contexts:
                return {
                    "response": "I couldn't find any relevant information in the available documents. Please try rephrasing your query or check if the documents are properly indexed.",
                    "agent_type": "error",
                    "sources_used": 0
                }
            
            # Step 2: Route to appropriate agent
            context_summary = f"{len(contexts)} relevant sections found from {len(set(doc.metadata.get('file_name', 'unknown') for doc in contexts))} documents"
            routing_result = self.router.route_query(query, context_summary)
            chosen_agent = routing_result["agent"]
            
            logger.info(f"Routed to {chosen_agent}: {routing_result['reason']}")
            
            # Step 3: Process with chosen agent
            if chosen_agent == "SUMMARY_AGENT":
                result = self.summary_agent.generate_summary(query, contexts)
            elif chosen_agent == "NEEDLE_AGENT":
                result = self.needle_agent.find_specific_info(query, contexts)
            elif chosen_agent == "TABLE_AGENT":
                result = self.table_agent.analyze_tables(query, contexts)
            else:
                # Fallback to general processing
                result = self._process_with_general_agent(query, contexts)
            
            # Add routing metadata
            result["routing_info"] = routing_result
            result["contexts_retrieved"] = len(contexts)
            result["unique_documents"] = len(set(doc.metadata.get('file_name', 'unknown') for doc in contexts))
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent processing failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "agent_type": "error",
                "error": str(e),
                "sources_used": 0
            }
    
    def process_with_integration(self, query: str, picture_analysis: Optional[Dict] = None,
                               vibration_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process query with integration to live gear analysis system
        
        Args:
            query: User query
            picture_analysis: Picture analysis results
            vibration_analysis: Vibration analysis results
            
        Returns:
            Dictionary with integrated response
        """
        try:
            # First get document analysis results
            doc_result = self.process_query(query)
            
            # Then integrate with live system data
            integration_result = self.integration_agent.integrate_analysis(
                query=query,
                document_results=doc_result["response"],
                picture_analysis=picture_analysis,
                vibration_analysis=vibration_analysis
            )
            
            # Combine metadata
            integration_result["document_analysis"] = doc_result
            
            return integration_result
            
        except Exception as e:
            logger.error(f"Integration processing failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error during integration: {str(e)}",
                "agent_type": "error",
                "error": str(e)
            }
    
    def _process_with_general_agent(self, query: str, contexts: List[Document]) -> Dict[str, Any]:
        """Fallback processing with general RAG agent"""
        try:
            formatted_contexts = format_context_for_llm(contexts, query)
            
            prompt = GENERAL_RAG_PROMPT.format(
                query=query,
                contexts=formatted_contexts
            )
            
            response = self.general_llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "response": response.content,
                "agent_type": "general",
                "sources_used": len(contexts),
                "context_types": [doc.metadata.get("section", "unknown") for doc in contexts]
            }
            
        except Exception as e:
            logger.error(f"General agent failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "agent_type": "error",
                "error": str(e),
                "sources_used": 0
            }

def create_multi_agent_system(retriever: HybridRetriever) -> MultiAgentSystem:
    """Factory function to create the multi-agent system"""
    return MultiAgentSystem(retriever)
