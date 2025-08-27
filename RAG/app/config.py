from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
	DATA_DIR: Path = Path("RAG/data")
	INDEX_DIR: Path = Path("index")
	EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small"
	EMBEDDING_MODEL_GOOGLE: str = "models/text-embedding-004"
	DENSE_K: int = 20  # Increased from 10
	SPARSE_K: int = 20  # Increased from 10
	K_TOP_K: int = 40   # Increased from 20
	RERANK_TOP_K: int = 40  # Increased from 20
	CONTEXT_TOP_N: int = 8
	CHUNK_TOK_AVG_RANGE: tuple[int, int] = (250, 500)
	CHUNK_TOK_MAX: int = 800
	MIN_PAGES: int = 10


settings = Settings()
