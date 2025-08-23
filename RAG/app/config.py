from dataclasses import dataclass
from pathlib import Path
import os

# Import PDF extraction settings
from .pdf_extractions_settings import pdf_settings

# If the user save the API key in a text file, read it from the file
def read_api_key_from_file(key_file_path: str) -> str:
    """Read API key from a text file."""
    try:
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r') as f:
                api_key = f.read().strip()
                # Set as environment variable for Google API
                if "GOOGLE" in key_file_path.upper():
                    os.environ["GOOGLE_API_KEY"] = api_key
                elif "OPENAI" in key_file_path.upper() or "CHATGPT" in key_file_path.upper():
                    os.environ["OPENAI_API_KEY"] = api_key
                return api_key
        return ""
    except Exception:
        return ""

@dataclass(frozen=True)
class Settings:
    # Data directories
    DATA_DIR: Path = Path("C:/Users/amitl/Documents/AI Developers/PracticeBase/Final_project")  # Project root directory
    INDEX_DIR: Path = Path("index")
    
    # ============================================================================
    # 🚀 LLM CONFIGURATION SETTINGS - MODIFY THESE TO CHANGE LLM BEHAVIOR
    # ============================================================================
    PREFERRED_LLM: str = "openai"      # Preferred LLM: "openai" or "google"
    
    # API Keys - Read from text files in PracticeBase folder
    GOOGLE_API_KEY: str = read_api_key_from_file("C:/Users/amitl/Documents/AI Developers/PracticeBase/Google API key.txt")
    OPENAI_API_KEY: str = read_api_key_from_file("C:/Users/amitl/Documents/AI Developers/PracticeBase/ChatGPT API key.txt")
    
    # OpenAI Settings
    OPENAI_MODEL: str = "gpt-4o-mini"  # OpenAI model to use
    OPENAI_TEMPERATURE: float = 0.0    # OpenAI temperature setting
    
    # Google Settings
    GOOGLE_MODEL: str = "gemini-1.5-flash"  # Google model to use
    GOOGLE_TEMPERATURE: float = 0.0    # Google temperature setting
    
    # Embedding models
    EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small"
    EMBEDDING_MODEL_GOOGLE: str = "models/text-embedding-004"
    
    # Retrieval settings
    DENSE_K: int = 20          # Number of dense vector results (increased from 15)
    SPARSE_K: int = 40         # Number of sparse vector results (increased from 15) - let BM25 sweep broader
    RERANK_TOP_K: int = 50     # Top candidates for reranking (increased from 30)
    CONTEXT_TOP_N: int = 12    # Final context chunks to use
    
    # Chunking settings
    CHUNK_TOK_AVG_RANGE: tuple[int, int] = (250, 500)  # Target chunk size range
    CHUNK_TOK_MAX: int = 800   # Maximum chunk size for tables/figures
    
    # Validation settings
    MIN_PAGES: int = 10        # Minimum pages required for processing

    # ============================================================================
    # 🔧 RETRIEVAL CONSTANTS - CENTRALIZED FOR CONSISTENCY
    # ============================================================================
    MIN_CASE_ID_LENGTH: int = 3  # Minimum characters for case/ID to avoid filtering on trivial digits
    MAX_KEYWORDS: int = 10  # Maximum number of keywords to extract
    METADATA_BOOST_FACTOR: float = 0.2  # Boost factor for metadata matching
    DEFAULT_TOP_N: int = 8  # Default number of top candidates for reranking
    HYBRID_RETRIEVER_WEIGHTS: list = None  # Weights for ensemble retriever - favor sparse BM25 for exact tokens/symbols
    
    def __post_init__(self):
        """Set default values for lists."""
        if self.HYBRID_RETRIEVER_WEIGHTS is None:
            object.__setattr__(self, 'HYBRID_RETRIEVER_WEIGHTS', [0.7, 0.3])


settings = Settings()
