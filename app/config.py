from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
	DATA_DIR: Path = Path("data")
	INDEX_DIR: Path = Path("index")
	EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small"
	EMBEDDING_MODEL_GOOGLE: str = "models/text-embedding-004"
	DENSE_K: int = 10
	SPARSE_K: int = 10
	K_TOP_K: int = 20
	RERANK_TOP_K: int = 20
	CONTEXT_TOP_N: int = 8
	CHUNK_TOK_AVG_RANGE: tuple[int, int] = (250, 500)
	CHUNK_TOK_MAX: int = 800
	MIN_PAGES: int = 10
	
	# Default extraction settings for optimal performance
	USE_TABULA: bool = True
	USE_PDFPLUMBER: bool = False
	USE_CAMELOT: bool = False
	USE_NORMALIZED_RAG: bool = True
	USE_GRAPH_DB: bool = True
	ENABLE_LLAMA_INDEX: bool = False  # Optional advanced feature
	
	# Ground truth files
	GROUND_TRUTH_FILE: Path = DATA_DIR / "gear_wear_ground_truth_context_free.json"
	QA_FILE: Path = DATA_DIR / "gear_wear_qa_context_free.jsonl"


# Set environment defaults for clean runs
def setup_default_env():
	"""Set optimal environment variables for clean, fast runs."""
	env_defaults = {
		"RAG_USE_TABULA": "1",
		"RAG_USE_PDFPLUMBER": "0", 
		"RAG_USE_CAMELOT": "0",
		"RAG_USE_NORMALIZED": "1",
		"RAG_HEADLESS": "0",  # Keep UI for testing
		"RAG_EVAL": "0",  # Skip evaluation for speed
		"RAG_ENABLE_LLAMAINDEX": "0",
		"CE_ENABLED": "1",
		"CE_PROVIDER": "auto",
	}
	
	for key, value in env_defaults.items():
		if key not in os.environ:
			os.environ[key] = value


settings = Settings()
