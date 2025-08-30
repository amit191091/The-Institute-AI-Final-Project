from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List, Tuple, Dict

# This file is used to store the settings for the RAG system

@dataclass(frozen=True)
class PathSettings:
    """Configuration for file paths and directories."""
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent 
    DATA_DIR: Path = PROJECT_ROOT / "RAG" / "data" # For data directory
    INDEX_DIR: Path = PROJECT_ROOT / "RAG" / "index" # For index directory
    LOGS_DIR: Path = PROJECT_ROOT / "RAG" / "logs" # For logs directory
    DEEPEVAL_DIR: Path = PROJECT_ROOT / "RAG" / "deepeval" # For DeepEval working directory
    GEAR_IMAGES_DIR: Path = PROJECT_ROOT / "gear_images" # For gear images directory


@dataclass(frozen=True)
class EmbeddingSettings:
    """Configuration for embedding models and retrieval."""
    EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small" # For OpenAI embedding model
    EMBEDDING_MODEL_GOOGLE: str = "models/text-embedding-004" # For Google embedding model
    DENSE_K: int = 10 # For dense retrieval
    SPARSE_K: int = 10 # For sparse retrieval
    K_TOP_K: int = 20 # For top k retrieval
    RERANK_TOP_K: int = 20 # For reranking
    CONTEXT_TOP_N: int = 8 # For context retrieval


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for LLM models and parameters."""
    # Primary LLM model selection - change this to switch between providers
    PRIMARY_LLM_PROVIDER: str = "openai"  # Options: "openai", "google", "auto"
    
    # Model names
    GOOGLE_MODEL: str = "gemini-1.5-flash" # For Google Gemini model
    OPENAI_MODEL: str = "gpt-4o-mini" # For OpenAI model (default)
    
    # Temperature settings
    TEMPERATURE: float = 0.0    # For response creativity (0.0 = deterministic, 1.0 = creative)
    MAX_TOKENS: int = 150       # For response length limit (reduced for precision)
    TOP_P: float = 0.1          # For response diversity (reduced for precision)
    
    # Environment variable names
    FORCE_OPENAI_ENV: str = "FORCE_OPENAI_ONLY" # Environment variable to force OpenAI
    GOOGLE_API_KEY_ENV: str = "GOOGLE_API_KEY" # Environment variable for Google API key
    OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY" # Environment variable for OpenAI API key
    OPENAI_CHAT_MODEL_ENV: str = "OPENAI_CHAT_MODEL" # Environment variable for OpenAI model override
    
    # Fallback settings
    FALLBACK_EMBEDDING_SIZE: int = 1536 # For FakeEmbeddings fallback size
    ERROR_PROMPT_LENGTH: int = 400 # For error message prompt truncation


@dataclass(frozen=True)
class ChunkingSettings:
    """Configuration for document chunking."""
    # Basic chunking settings
    CHUNK_TOK_AVG_RANGE: Tuple[int, int] = (250, 500) # For text chunks
    CHUNK_TOK_MAX: int = 800 # For Table and Figure chunks	
    MIN_PAGES: int = 10 # For minimum number of pages in a document
    
    # Advanced semantic chunking settings
    USE_SEMANTIC_CHUNKING: bool = True # For AI-powered semantic chunking (default ON)
    USE_HEADING_DETECTION: bool = True # For heading detection and hierarchy (default ON)
    USE_DYNAMIC_TOKENS: bool = True # For content-type specific token limits (default ON)
    TEXT_SPLIT_MULTI: bool = True # For multi-element text splitting (default ON)
    SEMANTIC_CHUNKING: bool = True # For semantic chunking (default ON)
    
    # Semantic chunking parameters
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.7 # For sentence similarity threshold
    SEMANTIC_BATCH_SIZE: int = 32 # For sentence transformer batch size
    SEMANTIC_MODEL_NAME: str = "all-MiniLM-L6-v2" # For sentence transformer model
    
    # Heading detection parameters
    HEADING_MIN_LENGTH: int = 3 # For minimum heading length
    HEADING_MAX_LENGTH: int = 100 # For maximum heading length
    HEADING_PATTERNS: List[str] = field(default_factory=lambda: [
        r"^\d+\.\s+[A-Z]",  # 1. Heading
        r"^\d+\.\d+\s+[A-Z]",  # 1.1 Subheading
        r"^[A-Z][A-Z\s]{2,}$",  # ALL CAPS HEADING
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$",  # Title Case Heading
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:$",  # Title Case with colon
    ])
    
    # Dynamic token management
    TEXT_TARGET_TOK: int = 500 # For text chunk target tokens (increased from 375)
    TEXT_MAX_TOK: int = 700 # For text chunk maximum tokens (increased from 500)
    FIGURE_TABLE_MAX_TOK: int = 1000 # For figure/table chunk maximum tokens (increased from 800)
    CONTEXT_LOW_N: int = 8 # For context window size (increased from 6)
    
    # Additional token settings for backward compatibility
    MIN_CHUNK_TOKENS: int = 50 # For minimum chunk tokens
    TEXT_TARGET_TOKENS: int = 375 # For text chunk target tokens (alias)
    TEXT_MAX_TOKENS: int = 500 # For text chunk maximum tokens (alias)
    TEXT_OVERLAP_SENTENCES: int = 2 # For text overlap sentences
    DISTILL_RATIO: float = 0.8 # For distillation ratio
    
    # Chunking quality settings
    MIN_CHUNK_LENGTH: int = 50 # For minimum chunk length in characters
    MAX_CHUNK_LENGTH: int = 2000 # For maximum chunk length in characters
    PREFER_SEMANTIC_BREAKS: bool = True # For preferring semantic over length breaks
    
    # Debug and logging settings
    CHUNKING_DEBUG: bool = False # For chunking debug output
    LOG_CHUNK_STATS: bool = True # For logging chunk statistics (default ON)
    EXPORT_CHUNK_METADATA: bool = True # For exporting chunk metadata (default ON)


@dataclass(frozen=True)
class QueryAnalysisSettings:
    """Configuration for query analysis and keyword extraction."""
    TECHNICAL_TERMS: List[str] = (
        "mg-5025a", "ins haifa", "rps", "khz", "mv/g", "μm", "micron",
        "dytran", "honeywell", "accelerometer", "tachometer", "sensitivity",
        "wear depth", "rms", "fft", "spectrogram", "sideband", "meshing frequency",
        "threshold", "escalation", "baseline", "criterion"
    )
    
    # Modular wear case generation (1 to 35)
    WEAR_CASE_RANGE: Tuple[int, int] = (1, 35)
    
    # Configurable wear case patterns (can be customized for any wear case set)
    WEAR_CASE_PATTERNS: Dict[str, List[int]] = field(default_factory=lambda: {
        "main": [1, 15, 25, 35],  # Main reference cases (can be W1, W9, W15, W23, etc.)
        "extended": [1, 11, 14, 15, 23, 25, 32, 35],  # Extended set
        "measurements": [1, 15, 25, 35]  # Cases with measurement data
    })
    
    @property
    def WEAR_CASES(self) -> List[str]:
        """Generate wear cases from 1 to 35 in modular way."""
        return [f"w{i}" for i in range(self.WEAR_CASE_RANGE[0], self.WEAR_CASE_RANGE[1] + 1)]
    
    @property
    def WEAR_CASES_MAIN(self) -> List[str]:
        """Generate main reference cases dynamically from configured pattern."""
        pattern = self.WEAR_CASE_PATTERNS.get("main", [1, 15, 25, 35])
        return [f"w{i}" for i in pattern if self.WEAR_CASE_RANGE[0] <= i <= self.WEAR_CASE_RANGE[1]]
    
    @property
    def WEAR_CASES_EXTENDED(self) -> List[str]:
        """Generate extended set dynamically from configured pattern."""
        pattern = self.WEAR_CASE_PATTERNS.get("extended", [1, 11, 14, 15, 23, 25, 32, 35])
        return [f"w{i}" for i in pattern if self.WEAR_CASE_RANGE[0] <= i <= self.WEAR_CASE_RANGE[1]]
    
    @property
    def WEAR_CASES_WITH_MEASUREMENTS(self) -> List[str]:
        """Generate wear cases that have measurement data."""
        pattern = self.WEAR_CASE_PATTERNS.get("measurements", [1, 15, 25, 35])
        return [f"w{i}" for i in pattern if self.WEAR_CASE_RANGE[0] <= i <= self.WEAR_CASE_RANGE[1]]
    
    FIGURE_REFS: List[str] = (
        "figure 1", "figure 2", "figure 3", "figure 4", "fig 1", "fig 2", "fig 3", "fig 4"
    )
    
    UNITS: List[str] = (
        "rps", "khz", "mv/g", "μm", "micron", "mm", "db"
    )
    
    EQUIPMENT: List[str] = (
        "gear", "gearbox", "transmission", "shaft", "starboard", "port"
    )
    
    TABLE_QUESTION_KEYWORDS: List[str] = (
        "table", "data", "value", "measurement", "wear depth",
        "accelerometer", "tachometer", "sensitivity", "threshold", "criterion", "module value"
    )
    
    FIGURE_QUESTION_KEYWORDS: List[str] = (
        "figure", "fig", "plot", "graph", "rms", "fft", "spectrogram", "sideband",
        "normalized fme", "face view", "dashed line"
    )
    
    THRESHOLD_QUESTION_KEYWORDS: List[str] = (
        "alert threshold", "rms", "baseline", "rolling average"
    )
    
    ESCALATION_QUESTION_KEYWORDS: List[str] = (
        "escalation criterion", "immediate inspection", "high-amplitude", "impact trains", "multiple records"
    )


@dataclass(frozen=True)
class RerankingSettings:
    """Configuration for document reranking and scoring."""
    MIN_SCORE_THRESHOLD_RATIO: float = 0.25 # For minimum score threshold ratio (increased for precision)
    MAX_DOCS_PER_SECTION: int = 1 # For maximum number of documents per section (reduced for precision)
    MAX_DOCS_PER_FILE: int = 2 # For maximum number of documents per file (reduced for precision)
    MAX_DOCS_PER_PAGE: int = 1 # For maximum number of documents per page (reduced for precision)
    
    # Scoring weights
    LEXICAL_OVERLAP_WEIGHT: float = 100.0 # For lexical overlap weight
    SECTION_BONUS: float = 200.0 # For section bonus
    TABLE_CONTENT_BONUS: float = 150.0 # For table content bonus
    FIGURE_CONTENT_BONUS: float = 150.0 # For figure content bonus
    WEAR_CASE_BONUS: float = 300.0 # For wear case bonus
    THRESHOLD_BONUS: float = 250.0 # For threshold bonus
    ESCALATION_BONUS: float = 200.0 # For escalation bonus
    MODULE_VALUE_BONUS: float = 250.0 # For module value bonus
    RECOMMENDATION_BONUS: float = 80.0 # For recommendation bonus


@dataclass(frozen=True)
class FallbackSettings:
    """Configuration for fallback data retrieval."""
    MAX_TABLE_DOCS: int = 3 # For maximum number of table documents
    MAX_MEASUREMENT_DOCS: int = 2 # For maximum number of measurement documents
    MAX_SPEED_DOCS: int = 2 # For maximum number of speed documents
    MAX_ACCELEROMETER_DOCS: int = 2 # For maximum number of accelerometer documents
    MAX_THRESHOLD_DOCS: int = 2 # For maximum number of threshold documents	
    
    # Configurable wear case patterns for measurements (can be customized for any wear case set)
    # Default values are from the actual table data (Table 1: Wear severities dimensions)
    WEAR_MEASUREMENT_PATTERNS: Dict[str, int] = field(default_factory=lambda: {
        "healthy": 0, "w1": 40, "w2": 81, "w3": 115, "w4": 159, "w5": 175, "w6": 195, "w7": 227, "w8": 256,
        "w9": 276, "w10": 294, "w11": 305, "w12": 323, "w13": 344, "w14": 378, "w15": 400, "w16": 417, "w17": 436,
        "w18": 450, "w19": 466, "w20": 488, "w21": 510, "w22": 524, "w23": 557, "w24": 579, "w25": 608, "w26": 637,
        "w27": 684, "w28": 720, "w29": 744, "w30": 769, "w31": 797, "w32": 825, "w33": 853, "w34": 890, "w35": 932
    })
    
    # Speed data patterns (can be customized for any speed values)
    SPEED_PATTERNS: List[str] = field(default_factory=lambda: ["15", "45"])
    
    # Speed query terms (generic, not hardcoded to specific values)
    SPEED_QUERY_TERMS: List[str] = (
        "speed", "rps", "rpm"
    )
    
    # Accelerometer data
    ACCELEROMETER_DATA: List[str] = (
        "dytran", "3053b", "9.47", "9.35", "mv/g"
    )
    
    # Accelerometer query terms
    ACCELEROMETER_QUERY_TERMS: List[str] = (
        "accelerometer", "dytran", "3053b", "sensor"
    )
    
    # Threshold data
    THRESHOLD_DATA: List[str] = (
        "6 db", "25%", "7-day", "rolling average", "high-amplitude", "immediate inspection"
    )
    
    # Threshold query terms
    THRESHOLD_QUERY_TERMS: List[str] = (
        "alert threshold", "rms", "crest factor", "escalation criterion", "immediate inspection"
    )
    
    @property
    def WEAR_MEASUREMENTS(self) -> List[str]:
        """Generate wear measurements dynamically from configured patterns."""
        return [f"{case}, {value}" for case, value in self.WEAR_MEASUREMENT_PATTERNS.items()]
    
    @property
    def WEAR_MEASUREMENT_CASES(self) -> List[str]:
        """Get just the wear case names from measurements."""
        return list(self.WEAR_MEASUREMENT_PATTERNS.keys())
    
    @property
    def SPEED_VALUES(self) -> List[str]:
        """Generate speed data dynamically from configured patterns."""
        return [f"{speed} rps" for speed in self.SPEED_PATTERNS] + [f"{speed} [rps]" for speed in self.SPEED_PATTERNS]


@dataclass(frozen=True)
class LoggingSettings:
    """Configuration for logging and tracing."""
    RAG_TRACE_ENV: str = "RAG_TRACE" # For tracing environment variable
    RAG_TRACE_RETRIEVAL_ENV: str = "RAG_TRACE_RETRIEVAL" # For retrieval tracing environment variable
    RAG_SPARSE_WEIGHT_ENV: str = "RAG_SPARSE_WEIGHT" # For sparse weight environment variable
    RAG_DENSE_WEIGHT_ENV: str = "RAG_DENSE_WEIGHT" # For dense weight environment variable


@dataclass(frozen=True)
class LoaderSettings:
    """Configuration for document loading and parsing."""
    # OCR and PDF processing settings
    OCR_LANG: str = "eng" # For OCR language setting
    PDF_HI_RES: bool = False # For high-resolution PDF parsing (slowest)
    USE_TABULA: bool = False # For Tabula table extraction (requires Java)
    USE_CAMELOT: bool = False # For Camelot table extraction (slow)
    USE_PDFPLUMBER: bool = True # For PDFPlumber table extraction (moderate) - default ON
    USE_PYMUPDF: bool = True # For PyMuPDF image extraction (fast and reliable) - default ON
    USE_LLAMA_PARSE: bool = False # For LlamaParse enhanced table parsing (requires API key)
    EXTRACT_IMAGES: bool = True # For image extraction (slow) - default ON
    SYNTH_TABLES: bool = True # For table synthesis (slow) - default ON
    
    # Table extraction thresholds
    MIN_TABLES_TARGET: int = 2 # For minimum tables target
    TABLES_PER_PAGE: int = 3 # For maximum tables per page
    LINE_COUNT_MIN: int = 20 # For minimum line count for table detection
    TEXT_BLOCKS_MIN: int = 2 # For minimum text blocks for table detection
    
    # Camelot specific settings
    CAMELOT_FLAVORS: str = "lattice,stream" # For Camelot extraction flavors
    CAMELOT_PAGES: str = "all" # For Camelot page selection
    CAMELOT_MIN_ROWS: int = 3 # For minimum rows in Camelot tables
    CAMELOT_MIN_COLS: int = 2 # For minimum columns in Camelot tables
    CAMELOT_MIN_ACCURACY: float = 70.0 # For minimum accuracy (0-100)
    CAMELOT_MAX_EMPTY_COL_RATIO: float = 0.6 # For maximum empty column ratio
    CAMELOT_MIN_NUMERIC_RATIO: float = 0.0 # For minimum numeric ratio (0-1)
    
    # Export and debugging settings
    EXPORT_TABLES: bool = True # For exporting tables to files (default ON)
    DUMP_ELEMENTS: bool = True # For dumping elements for debugging (default ON)
    USE_PYMUPDF_TEXT: bool = True # For PyMuPDF text extraction (default ON)
    
    # Debug flags for individual extractors
    PDFPLUMBER_DEBUG: bool = False # For pdfplumber debug output
    CAMELOT_DEBUG: bool = False # For camelot debug output
    TABULA_DEBUG: bool = False # For tabula debug output
    
    # Exclusive extractor mode
    EXCLUSIVE_EXTRACTOR: str = "" # For running only one extractor (useful for testing)
    
    # Logging and debugging
    LOG_LEVEL: str = "ERROR" # For logging level


@dataclass(frozen=True)
class EvaluationSettings:
    """Configuration for RAG evaluation metrics and targets."""
    
    # Answer correctness evaluation weights for different similarity metrics
    # These weights determine the importance of each metric in calculating answer correctness
    ANSWER_CORRECTNESS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "sequence_similarity": 0.25,  # Overall text similarity (highest priority)
        "token_overlap": 0.15,        # Word-level similarity (medium priority)
        "exact_match": 0.25,          # Perfect match bonus (highest priority)
        "number_accuracy": 0.10,      # Numerical accuracy (medium-low priority)
        "date_accuracy": 0.05,        # Date accuracy (lowest priority)
        "technical_terms": 0.15,      # Technical terminology (medium priority)
        "citations": 0.05             # Citation accuracy (lowest priority)
    })
    
    # Evaluation target thresholds for performance assessment
    # These targets define the minimum acceptable scores for each metric
    EVALUATION_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        "answer_correctness": 0.85,  # For correct answers ≥ 0.85
        "context_precision": 0.75,   # For precise answers ≥ 0.75
        "context_recall": 0.70,      # For relevant context ≥ 0.70
        "faithfulness": 0.85,        # For accurate answers ≥ 0.85
        "table_qa_accuracy": 0.90,   # For table-based questions ≥ 0.90
    })
    
    @property
    def ANSWER_CORRECTNESS_WEIGHT_LIST(self) -> List[float]:
        """Get weights as a list for easy use in calculations."""
        return list(self.ANSWER_CORRECTNESS_WEIGHTS.values())


# Data Source Classification
DATA_SOURCES = {
    "report": [
        "Gear wear Failure.pdf",
        "Gear wear Failure-table-01.csv",
        "Gear wear Failure-table-02.csv", 
        "Gear wear Failure-table-03.csv"
    ],
    "database": [
        # Removed "Database figures and tables.pdf" - 
        # "Database figures and tables-table-*.csv",
        # "Database figures and tables-table-*.md"
    ],
    "other": [
        # No other files currently needed
    ]
}

# Source priority for ambiguous queries
SOURCE_PRIORITY = ["report", "database"]


@dataclass(frozen=True)
class Settings:
    """Main settings class that combines all configuration components."""
    paths: PathSettings = PathSettings() # For file paths and directories
    embedding: EmbeddingSettings = EmbeddingSettings() # For embedding models and retrieval
    llm: LLMSettings = LLMSettings() # For LLM models and parameters
    chunking: ChunkingSettings = ChunkingSettings() # For document chunking
    query_analysis: QueryAnalysisSettings = QueryAnalysisSettings() # For query analysis and keyword extraction
    reranking: RerankingSettings = RerankingSettings() # For document reranking and scoring
    fallback: FallbackSettings = FallbackSettings() # For fallback data retrieval
    logging: LoggingSettings = LoggingSettings() # For logging and tracing
    loader: LoaderSettings = LoaderSettings() # For document loading and parsing
    evaluation: EvaluationSettings = EvaluationSettings() # For evaluation metrics and targets


# Create the main settings instance
settings = Settings()