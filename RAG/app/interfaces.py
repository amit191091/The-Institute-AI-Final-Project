"""
Interfaces for RAG system components.
These define clear contracts between components and enable dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol
from langchain.schema import Document


class DocumentIngestionInterface(ABC):
    """Interface for document ingestion components."""
    
    @abstractmethod
    def ingest_documents(self, file_paths: List[str]) -> List[Document]:
        """Ingest documents from file paths and return Document objects."""
        pass
    
    @abstractmethod
    def validate_documents(self, documents: List[Document]) -> bool:
        """Validate that documents meet minimum requirements."""
        pass


class QueryProcessingInterface(ABC):
    """Interface for query processing components."""
    
    @abstractmethod
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query and extract relevant information."""
        pass
    
    def apply_filters(self, documents: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Apply filters to documents based on query analysis."""
        from RAG.app.retrieve_modules.retrieve_filters import apply_filters as apply_filters_impl
        return apply_filters_impl(documents, filters)


class RetrievalInterface(ABC):
    """Interface for document retrieval components."""
    
    @abstractmethod
    def retrieve_documents(self, query: str, top_k: int = 8) -> List[Document]:
        """Retrieve relevant documents for a query."""
        pass
    
    @abstractmethod
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 8) -> List[Document]:
        """Rerank documents based on relevance to query."""
        pass


class AnswerGenerationInterface(ABC):
    """Interface for answer generation components."""
    
    @abstractmethod
    def generate_answer(self, query: str, context_documents: List[Document]) -> str:
        """Generate an answer based on query and context documents."""
        pass
    
    @abstractmethod
    def format_answer(self, answer: str, sources: List[Document]) -> Dict[str, Any]:
        """Format the answer with source information."""
        pass


class EmbeddingInterface(Protocol):
    """Protocol for embedding models."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        ...
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        ...


class IndexInterface(ABC):
    """Interface for document indexing components."""
    
    @abstractmethod
    def build_index(self, documents: List[Document]) -> Any:
        """Build an index from documents."""
        pass
    
    @abstractmethod
    def search_index(self, query: str, top_k: int = 10) -> List[Document]:
        """Search the index for relevant documents."""
        pass


class ChunkingInterface(ABC):
    """Interface for document chunking components."""
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        pass
    
    @abstractmethod
    def attach_metadata(self, chunks: List[Document]) -> List[Document]:
        """Attach metadata to document chunks."""
        pass


class RAGPipelineInterface(ABC):
    """Interface for the complete RAG pipeline."""
    
    @abstractmethod
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline."""
        pass
    
    @abstractmethod
    def add_documents(self, file_paths: List[str]) -> bool:
        """Add new documents to the RAG system."""
        pass


class ConfigurationInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get_setting(self, key: str) -> Any:
        """Get a configuration setting by key."""
        pass
    
    @abstractmethod
    def update_setting(self, key: str, value: Any) -> bool:
        """Update a configuration setting."""
        pass


class LoggingInterface(ABC):
    """Interface for logging components."""
    
    @abstractmethod
    def log_info(self, message: str) -> None:
        """Log an info message."""
        pass
    
    @abstractmethod
    def log_error(self, message: str, error: Exception = None) -> None:
        """Log an error message."""
        pass
    
    @abstractmethod
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        pass
