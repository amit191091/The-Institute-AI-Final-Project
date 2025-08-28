#!/usr/bin/env python3
"""
RAGAS Setup
===========

RAGAS setup and configuration functions.
"""

import os


def _setup_ragas_llm():
	"""Setup LLM for RAGAS evaluation - uses centralized LLM configuration."""
	
	# Import the centralized LLM setup from rag_service
	try:
		from RAG.app.rag_service import RAGService
		rag_service = RAGService()
		llm_function = rag_service._get_llm()
		
		# Convert function-based LLM to LangChain format for RAGAS
		# RAGAS expects a LangChain LLM object, so we need to create a wrapper
		try:
			from langchain_openai import ChatOpenAI
			from RAG.app.config import settings
			
			# Use the same model as configured in the main system
			openai_model = settings.llm.OPENAI_MODEL
			llm = ChatOpenAI(model=openai_model, temperature=settings.llm.TEMPERATURE)
			print(f"[RAGAS LLM] Using OpenAI model: {openai_model}")
			return llm
		except Exception as e:
			print(f"Failed to setup LangChain LLM for RAGAS: {e}")
			return None
			
	except Exception as e:
		print(f"Failed to access centralized LLM setup: {e}")
		return None


def _setup_ragas_embeddings():
	"""Setup embeddings for RAGAS evaluation - uses centralized configuration."""
	try:
		from RAG.app.config import settings
		# Try OpenAI embeddings first (consistent with main system)
		if os.getenv("OPENAI_API_KEY"):
			try:
				from langchain_openai import OpenAIEmbeddings
				embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
				print("[RAGAS Embeddings] Using OpenAI text-embedding-3-small")
				return embeddings
			except Exception as e:
				print(f"Failed to setup OpenAI embeddings for RAGAS: {e}")
		
		# Fallback to Google embeddings if OpenAI not available
		if os.getenv("GOOGLE_API_KEY"):
			try:
				from langchain_google_genai import GoogleGenerativeAIEmbeddings
				embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
				print("[RAGAS Embeddings] Using Google text-embedding-004")
				return embeddings
			except Exception as e:
				print(f"Failed to setup Google embeddings for RAGAS: {e}")
		
		return None
		
	except Exception as e:
		print(f"Failed to setup embeddings for RAGAS: {e}")
		return None
