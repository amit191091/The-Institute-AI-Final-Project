#!/usr/bin/env python3
"""
LLM Service
==========

Handles language model creation and management.
"""

import logging

from RAG.app.logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """Service for language model management."""

    def __init__(self):
        """Initialize the LLM service."""
        self.log = get_logger()

    def _get_llm(self):
        """Get LLM function for text generation."""
        try:
            # Try to use OpenAI if available
            from openai import OpenAI
            client = OpenAI()
            
            def llm_function(prompt):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0
                )
                return response.choices[0].message.content
            
            return llm_function
        except ImportError:
            logger.warning("OpenAI not available, using mock LLM")
            # Mock LLM for testing
            def mock_llm(prompt):
                return f"Mock response to: {prompt[:100]}..."
            return mock_llm
        except Exception as e:
            logger.warning(f"Error setting up LLM: {e}, using mock LLM")
            def mock_llm(prompt):
                return f"Mock response to: {prompt[:100]}..."
            return mock_llm
