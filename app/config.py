from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
	DATA_DIR: Path = Path("data")
	INDEX_DIR: Path = Path("index")
	EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small"
	EMBEDDING_MODEL_GOOGLE: str = "models/text-embedding-004"
	# Defaults: prefer Google providers; avoid OpenAI unless explicitly forced by env
	DENSE_K: int = 10 # Number of dense vectors to retrieve
	SPARSE_K: int = 10 # Number of sparse vectors to retrieve
	K_TOP_K: int = 20 # Number of top k results to retrieve
	RERANK_TOP_K: int = 20 # Number of top k results to rerank
	CONTEXT_TOP_N: int = 8 # Number of top n results to retrieve
	CONTEXT_LOW_N: int = 6 # Number of low n results to retrieve
	CHUNK_TOK_AVG_RANGE: tuple[int, int] = (250, 500) # Average range of tokens to chunk
	CHUNK_TOK_MAX: int = 800 # Maximum number of tokens to chunk
	MIN_PAGES: int = 10 # Minimum number of pages to chunk


settings = Settings()
