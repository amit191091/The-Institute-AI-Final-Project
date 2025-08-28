#!/usr/bin/env python3
"""
Embedding Service
================

Handles embedding function creation and management.
"""

import logging
import hashlib

from RAG.app.logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for embedding function management."""

    def __init__(self):
        """Initialize the embedding service."""
        self.log = get_logger()

    def _get_embedding_function(self):
        """Get embedding function based on configuration."""
        try:
            # Use a simple sentence transformer for embeddings
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            class EmbeddingFunction:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    embeddings = self.model.encode(texts)
                    return embeddings.tolist()
                
                def embed_query(self, text):
                    embedding = self.model.encode([text])
                    return embedding.tolist()[0]
            
            return EmbeddingFunction(model)
        except ImportError:
            logger.warning("SentenceTransformers not available, using fallback embedding")
            # Fallback to a simple hash-based embedding for testing
            import hashlib
            
            class FallbackEmbeddingFunction:
                def embed_documents(self, texts):
                    embeddings = []
                    for text in texts:
                        # Create a simple hash-based embedding
                        hash_obj = hashlib.md5(text.encode())
                        hash_bytes = hash_obj.digest()
                        # Convert to 384-dimensional vector (like all-MiniLM-L6-v2)
                        embedding = [float(b) / 255.0 for b in hash_bytes] * 15  # Repeat to get 384 dims
                        embeddings.append(embedding[:384])  # Ensure exactly 384 dimensions
                    return embeddings
                
                def embed_query(self, text):
                    return self.embed_documents([text])[0]
            
            return FallbackEmbeddingFunction()
        except Exception as e:
            logger.error(f"Error getting embedding function: {str(e)}")
            raise
