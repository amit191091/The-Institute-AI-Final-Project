"""
Indexing system with dense embeddings, sparse retrieval, and metadata filtering
"""
import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import numpy as np
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)

class HybridIndex:
    """
    Hybrid indexing system combining dense embeddings (FAISS) and sparse retrieval (BM25)
    with metadata filtering capabilities
    """
    
    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or settings.INDEX_DIR
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )
        
        self.dense_index: Optional[FAISS] = None
        self.sparse_index: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.metadata_df: Optional[pd.DataFrame] = None
        
        # Index metadata
        self.index_info = {
            "document_count": 0,
            "chunk_count": 0,
            "last_updated": None,
            "indexed_files": []
        }
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to both dense and sparse indices
        
        Args:
            documents: List of LangChain Document objects with metadata
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Adding {len(documents)} documents to index")
        
        # Store documents
        self.documents.extend(documents)
        
        # Build dense index
        self._build_dense_index()
        
        # Build sparse index
        self._build_sparse_index()
        
        # Build metadata DataFrame for filtering
        self._build_metadata_index()
        
        # Update index info
        self._update_index_info(documents)
        
        logger.info("Documents successfully added to hybrid index")
    
    def _build_dense_index(self) -> None:
        """Build or update FAISS dense vector index.

        If embedding creation fails (e.g., missing/invalid API key), log a warning and
        continue without a dense index so that sparse retrieval remains available.
        """
        try:
            if self.dense_index is None:
                # Create new index
                self.dense_index = FAISS.from_documents(self.documents, self.embeddings)
            else:
                # Add to existing index
                new_docs = self.documents[len(self.dense_index.index_to_docstore_id):]
                if new_docs:
                    new_index = FAISS.from_documents(new_docs, self.embeddings)
                    self.dense_index.merge_from(new_index)
            logger.info(f"Dense index built with {len(self.documents)} documents")
        except Exception as e:
            logger.warning(
                "Dense index unavailable (embeddings failed). Falling back to sparse-only. Error: %s", e
            )
            self.dense_index = None
    
    def _build_sparse_index(self) -> None:
        """Build BM25 sparse index"""
        try:
            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in self.documents:
                # Simple tokenization (can be enhanced with proper NLP)
                tokens = doc.page_content.lower().split()
                tokenized_docs.append(tokens)
            
            self.sparse_index = BM25Okapi(tokenized_docs)
            logger.info(f"Sparse index built with {len(tokenized_docs)} documents")
        except Exception as e:
            logger.error(f"Failed to build sparse index: {e}")
            raise
    
    def _build_metadata_index(self) -> None:
        """Build metadata DataFrame for efficient filtering"""
        try:
            metadata_records = []
            for i, doc in enumerate(self.documents):
                record = {"doc_id": i}
                record.update(doc.metadata)
                metadata_records.append(record)
            
            self.metadata_df = pd.DataFrame(metadata_records)
            logger.info(f"Metadata index built with {len(metadata_records)} records")
        except Exception as e:
            logger.error(f"Failed to build metadata index: {e}")
            raise
    
    def _update_index_info(self, new_documents: List[Document]) -> None:
        """Update index metadata information"""
        from datetime import datetime
        
        self.index_info["chunk_count"] = len(self.documents)
        self.index_info["last_updated"] = datetime.now().isoformat()
        
        # Track indexed files
        new_files = set()
        for doc in new_documents:
            file_name = doc.metadata.get("file_name")
            if file_name:
                new_files.add(file_name)
        
        self.index_info["indexed_files"].extend(list(new_files))
        self.index_info["indexed_files"] = list(set(self.index_info["indexed_files"]))
        self.index_info["document_count"] = len(self.index_info["indexed_files"])
    
    def search_dense(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Search using dense embeddings (semantic similarity)
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of relevant documents
        """
        if not self.dense_index:
            logger.warning("Dense index not available")
            return []
        
        try:
            # Apply metadata filters if provided
            if filter_dict:
                filtered_docs = self._apply_metadata_filters(filter_dict)
                if not filtered_docs:
                    return []
                
                # Create temporary index with filtered documents
                temp_index = FAISS.from_documents(filtered_docs, self.embeddings)
                results = temp_index.similarity_search(query, k=k)
            else:
                results = self.dense_index.similarity_search(query, k=k)
            
            logger.debug(f"Dense search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def search_sparse(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Search using sparse retrieval (keyword matching)
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of relevant documents
        """
        if not self.sparse_index:
            logger.warning("Sparse index not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.sparse_index.get_scores(query_tokens)
            
            # Get top-k document indices
            top_indices = np.argsort(scores)[::-1][:k * 2]  # Get more candidates for filtering
            
            # Get documents
            candidates = [self.documents[i] for i in top_indices if i < len(self.documents)]
            
            # Apply metadata filters if provided
            if filter_dict:
                candidates = [doc for doc in candidates if self._matches_filters(doc, filter_dict)]
            
            # Return top-k after filtering
            results = candidates[:k]
            
            logger.debug(f"Sparse search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _apply_metadata_filters(self, filter_dict: Dict) -> List[Document]:
        """Apply metadata filters to get candidate documents"""
        if self.metadata_df is None:
            return self.documents
        
        filtered_df = self.metadata_df.copy()
        
        for key, value in filter_dict.items():
            if key in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        # Get filtered document indices
        filtered_indices = filtered_df["doc_id"].tolist()
        
        # Return filtered documents
        return [self.documents[i] for i in filtered_indices if i < len(self.documents)]
    
    def _matches_filters(self, document: Document, filter_dict: Dict) -> bool:
        """Check if a document matches the given filters"""
        for key, value in filter_dict.items():
            doc_value = document.metadata.get(key)
            
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            else:
                if doc_value != value:
                    return False
        
        return True
    
    def get_table_summaries(self, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Get table summaries with CSV/Markdown conversion
        
        Args:
            filter_dict: Optional metadata filters
            
        Returns:
            List of table summaries with anchors
        """
        table_docs = []
        
        for doc in self.documents:
            if doc.metadata.get("section") == "Table" or doc.metadata.get("is_table"):
                if not filter_dict or self._matches_filters(doc, filter_dict):
                    table_info = {
                        "table_id": doc.metadata.get("anchor", "unknown"),
                        "page_number": doc.metadata.get("page", 1),
                        "summary": doc.page_content.split("\\n\\nRAW DATA:")[0].replace("[TABLE]\\nSUMMARY:\\n", ""),
                        "raw_content": doc.page_content,
                        "row_range": doc.metadata.get("table_row_range"),
                        "columns": doc.metadata.get("table_col_names", []),
                        "file_name": doc.metadata.get("file_name")
                    }
                    table_docs.append(table_info)
        
        return table_docs
    
    def save_index(self, save_path: Optional[Path] = None) -> None:
        """Save the complete index to disk"""
        save_path = save_path or self.index_dir
        
        try:
            # Save dense index
            if self.dense_index:
                self.dense_index.save_local(str(save_path / "dense_index"))
            
            # Save sparse index
            if self.sparse_index:
                with open(save_path / "sparse_index.pkl", "wb") as f:
                    pickle.dump(self.sparse_index, f)
            
            # Save documents
            with open(save_path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # Save metadata DataFrame
            if self.metadata_df is not None:
                self.metadata_df.to_parquet(save_path / "metadata.parquet")
            
            # Save index info
            with open(save_path / "index_info.json", "w") as f:
                json.dump(self.index_info, f, indent=2)
            
            logger.info(f"Index saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, load_path: Optional[Path] = None) -> bool:
        """Load the complete index from disk"""
        load_path = load_path or self.index_dir
        
        try:
            # Load dense index
            dense_path = load_path / "dense_index"
            if dense_path.exists():
                self.dense_index = FAISS.load_local(str(dense_path), self.embeddings)
            
            # Load sparse index
            sparse_path = load_path / "sparse_index.pkl"
            if sparse_path.exists():
                with open(sparse_path, "rb") as f:
                    self.sparse_index = pickle.load(f)
            
            # Load documents
            docs_path = load_path / "documents.pkl"
            if docs_path.exists():
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
            
            # Load metadata DataFrame
            metadata_path = load_path / "metadata.parquet"
            if metadata_path.exists():
                self.metadata_df = pd.read_parquet(metadata_path)
            
            # Load index info
            info_path = load_path / "index_info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    self.index_info = json.load(f)
            
            logger.info(f"Index loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "dense_index_size": len(self.dense_index.index_to_docstore_id) if self.dense_index else 0,
            "sparse_index_size": len(self.documents) if self.sparse_index else 0,
            "metadata_columns": list(self.metadata_df.columns) if self.metadata_df is not None else [],
            "index_info": self.index_info
        }

def create_hybrid_index(index_dir: Optional[Path] = None) -> HybridIndex:
    """Factory function to create a hybrid index"""
    return HybridIndex(index_dir)
