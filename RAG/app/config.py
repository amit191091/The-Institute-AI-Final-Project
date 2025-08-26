import os
from dataclasses import dataclass
from pathlib import Path


def load_api_keys_from_files():
    """Load API keys from txt files if environment variables are not set."""
    # Load OpenAI API key from file if not in environment
    if not os.getenv("OPENAI_API_KEY"):
        openai_key_file = Path("openai_api_key.txt")
        if openai_key_file.exists():
            try:
                with open(openai_key_file, "r") as f:
                    openai_key = f.read().strip()
                    if openai_key:
                        os.environ["OPENAI_API_KEY"] = openai_key
                        print("[API Keys] Loaded OpenAI API key from file")
            except Exception as e:
                print(f"[API Keys] Error loading OpenAI API key from file: {e}")
    
    # Load Google API key from file if not in environment
    if not os.getenv("GOOGLE_API_KEY"):
        google_key_file = Path("google_api_key.txt")
        if google_key_file.exists():
            try:
                with open(google_key_file, "r") as f:
                    google_key = f.read().strip()
                    if google_key:
                        os.environ["GOOGLE_API_KEY"] = google_key
                        print("[API Keys] Loaded Google API key from file")
            except Exception as e:
                print(f"[API Keys] Error loading Google API key from file: {e}")


@dataclass(frozen=True)
class Settings:
	DATA_DIR: Path = Path("RAG/data")
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


settings = Settings()
