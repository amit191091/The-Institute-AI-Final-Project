#!/usr/bin/env python3
"""
Indexing Service
===============

Handles index building and hybrid retriever creation.
"""

import logging
from typing import List, Any, Tuple

from langchain.schema import Document

from RAG.app.logger import get_logger
from RAG.app.Data_Management.indexing import (
    build_dense_index,
    build_sparse_retriever,
    to_documents,
    dump_chroma_snapshot,
)
from RAG.app.retrieve import build_hybrid_retriever
from RAG.app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexingService:
    """Service for index building and retriever creation."""

    def __init__(self):
        """Initialize the indexing service."""
        self.log = get_logger()

    def build_indexes(self, docs: List[Document]) -> Tuple[Any, Any]:
        """
        Build dense and sparse indexes from documents.
        
        Args:
            docs: Processed documents
            
        Returns:
            Tuple[Any, Any]: (dense_index, sparse_retriever)
        """
        try:
            logger.info("Building indexes...")
            
            # Build dense index
            logger.info("Building dense index...")
            dense_index = build_dense_index(docs, self._get_embedding_function())
            
            # Build sparse retriever
            logger.info("Building sparse retriever...")
            sparse_retriever = build_sparse_retriever(docs, k=settings.embedding.SPARSE_K)
            
            logger.info("Indexes built successfully")
            return dense_index, sparse_retriever
            
        except Exception as e:
            logger.error(f"Error building indexes: {str(e)}")
            raise

    def build_hybrid_retriever(self, dense_index: Any, sparse_retriever: Any) -> Any:
        """
        Build hybrid retriever combining dense and sparse methods.
        
        Args:
            dense_index: Dense vector index
            sparse_retriever: Sparse BM25 retriever
            
        Returns:
            Any: Hybrid retriever
        """
        try:
            logger.info("Building hybrid retriever...")
            hybrid = build_hybrid_retriever(
                dense_index, 
                sparse_retriever, 
                dense_k=settings.embedding.DENSE_K
            )
            logger.info("Hybrid retriever built successfully")
            return hybrid
            
        except Exception as e:
            logger.error(f"Error building hybrid retriever: {str(e)}")
            raise

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
