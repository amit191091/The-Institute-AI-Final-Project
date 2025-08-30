#!/usr/bin/env python3
"""
RAGAS Setup
===========

RAGAS setup and configuration functions.
"""

import os
from RAG.app.logger import trace_func

# Optional Google safety enums (available in newer google-generativeai)
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
except Exception:  # pragma: no cover
    HarmCategory = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore


@trace_func
def _setup_ragas_llm():
    """Setup LLM for RAGAS evaluation using config-based provider selection."""
    from RAG.app.config import settings
    
    # Use config setting for primary LLM provider
    primary_provider = settings.llm.PRIMARY_LLM_PROVIDER.lower()
    
    # Check for environment override
    force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
    if force_openai:
        primary_provider = "openai"
    
    # Initialize based on primary provider setting
    if primary_provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                openai_model = os.getenv("OPENAI_CHAT_MODEL", settings.llm.OPENAI_MODEL)
                llm = ChatOpenAI(model=openai_model, temperature=0)
                print(f"[RAGAS LLM] Using OpenAI model: {openai_model}")
                return llm
            except Exception as e:
                print(f"Failed to setup OpenAI LLM for RAGAS: {e}")
                
    elif primary_provider == "google":
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                preferred_model = os.getenv("GOOGLE_CHAT_MODEL", settings.llm.GOOGLE_MODEL)
                llm = ChatGoogleGenerativeAI(
                    model=preferred_model,
                    temperature=0,
                    safety_settings=None,
                )
                print(f"[RAGAS LLM] Using Google Gemini model: {preferred_model}")
                return llm
            except Exception as e:
                print(f"Failed to setup Google LLM for RAGAS: {e}")
                
    elif primary_provider == "auto":
        # Auto mode: try Google first, then OpenAI
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                preferred_model = os.getenv("GOOGLE_CHAT_MODEL", settings.llm.GOOGLE_MODEL)
                llm = ChatGoogleGenerativeAI(
                    model=preferred_model,
                    temperature=0,
                    safety_settings=None,
                )
                print(f"[RAGAS LLM] Using Google Gemini model: {preferred_model}")
                return llm
            except Exception as e:
                print(f"Failed to setup Google LLM for RAGAS: {e}")
    
    # Fallback to OpenAI if primary provider failed
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            openai_model = os.getenv("OPENAI_CHAT_MODEL", settings.llm.OPENAI_MODEL)
            llm = ChatOpenAI(model=openai_model, temperature=0)
            print(f"[RAGAS LLM] Using OpenAI model (fallback): {openai_model}")
            return llm
        except Exception as e:
            print(f"Failed to setup OpenAI LLM for RAGAS: {e}")

    return None


@trace_func
def _setup_ragas_embeddings():
    """Setup embeddings for RAGAS evaluation using config-based provider selection."""
    from RAG.app.config import settings
    
    # Use config setting for primary LLM provider
    primary_provider = settings.llm.PRIMARY_LLM_PROVIDER.lower()
    
    # Check for environment override
    force_openai = os.getenv("FORCE_OPENAI_ONLY", "").strip().lower() in ("1", "true", "yes")
    if force_openai:
        primary_provider = "openai"
    
    # Initialize based on primary provider setting
    if primary_provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    model=settings.embedding.EMBEDDING_MODEL_OPENAI
                )
                print(f"[RAGAS Embeddings] Using OpenAI {settings.embedding.EMBEDDING_MODEL_OPENAI}")
                return embeddings
            except Exception as e:
                print(f"Failed to setup OpenAI embeddings for RAGAS: {e}")
                
    elif primary_provider == "google":
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.embedding.EMBEDDING_MODEL_GOOGLE
                )
                print(f"[RAGAS Embeddings] Using Google {settings.embedding.EMBEDDING_MODEL_GOOGLE}")
                return embeddings
            except Exception as e:
                print(f"Failed to setup Google embeddings for RAGAS: {e}")
                
    elif primary_provider == "auto":
        # Auto mode: try Google first, then OpenAI
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.embedding.EMBEDDING_MODEL_GOOGLE
                )
                print(f"[RAGAS Embeddings] Using Google {settings.embedding.EMBEDDING_MODEL_GOOGLE}")
                return embeddings
            except Exception as e:
                print(f"Failed to setup Google embeddings for RAGAS: {e}")
    
    # Fallback to OpenAI if primary provider failed
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model=settings.embedding.EMBEDDING_MODEL_OPENAI
            )
            print(f"[RAGAS Embeddings] Using OpenAI (fallback): {settings.embedding.EMBEDDING_MODEL_OPENAI}")
            return embeddings
        except Exception as e:
            print(f"Failed to setup OpenAI embeddings for RAGAS: {e}")
    
    return None
