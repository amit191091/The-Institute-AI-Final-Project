"""
Configuration settings for the Metadata-Driven Hybrid RAG System
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class Settings:
    # Directory Paths
    DATA_DIR: Path = Path("data")
    INDEX_DIR: Path = Path("index")
    REPORTS_DIR: Path = Path("reports")
    
    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    RERANK_MODEL: str = "BAAI/bge-reranker-large"
    # Google Generative AI (optional)
    GOOGLE_EMBEDDING_MODEL: str = "text-embedding-004"
    GOOGLE_LLM_MODEL: str = "gemini-1.5-flash"
    
    # Retrieval Parameters
    DENSE_K: int = 10
    SPARSE_K: int = 10
    RERANK_TOP_K: int = 20
    CONTEXT_TOP_N: int = 8
    
    # Chunking Parameters
    CHUNK_TOK_AVG_RANGE: Tuple[int, int] = (250, 500)  # avg target
    CHUNK_TOK_MAX: int = 800                           # hard cap for tables/figures
    OVERLAP_SIZE: int = 50                             # token overlap between chunks
    
    # Data Validation
    MIN_PAGES: int = 10                                # minimum pages for document ingestion
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    
    # Evaluation Thresholds
    MIN_CONTEXT_PRECISION: float = 0.75
    MIN_RECALL: float = 0.70
    MIN_FAITHFULNESS: float = 0.85
    MIN_TABLE_QA_ACCURACY: float = 0.90
    
    # Gear Analysis Integration
    GEAR_IMAGES_DIR: Path = Path("gear_images")
    VIBRATION_DATA_DIR: Path = Path("vibration_data")
    
    # Supported File Types
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".docx", ".doc", ".txt")

# Global settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.INDEX_DIR, settings.REPORTS_DIR]:
    directory.mkdir(exist_ok=True)
