"""
PDF Extraction Settings
Controls all PDF extraction features including tables, images, graphs, and text
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PDFSettings:
    # ============================================================================
    # 📊 TABLE EXTRACTION SETTINGS - ENABLE BETTER PDF PROCESSING
    # ============================================================================
    RAG_USE_PDFPLUMBER: bool = True     # Enable PDFPlumber for table extraction
    RAG_SYNTH_TABLES: bool = True       # Enable synthetic table detection
    RAG_USE_TABULA: bool = False        # Disable Tabula (requires Java)
    RAG_USE_CAMELOT: bool = False       # Disable Camelot (requires Ghostscript)
    
    # ============================================================================
    # 🖼️ IMAGE EXTRACTION SETTINGS - EXTRACT PICTURES AND GRAPHS
    # ============================================================================
    RAG_EXTRACT_IMAGES: bool = False    # Enable/disable image extraction
    RAG_IMAGE_FORMATS: list = None      # Supported image formats (png, jpg, etc.)
    RAG_IMAGE_QUALITY: int = 85         # Image quality (1-100)
    RAG_SAVE_IMAGES: bool = True        # Save extracted images to disk
    
    # ============================================================================
    # 📈 GRAPH AND CHART EXTRACTION SETTINGS
    # ============================================================================
    RAG_EXTRACT_GRAPHS: bool = False    # Enable graph/chart detection
    RAG_GRAPH_DETECTION: bool = False   # Use AI to detect graphs in images
    RAG_CHART_ANALYSIS: bool = False    # Analyze chart data when possible
    
    # ============================================================================
    # 📄 GENERAL PDF PROCESSING SETTINGS
    # ============================================================================
    RAG_PDF_HI_RES: bool = True         # Enable high-resolution PDF parsing
    RAG_PDF_DPI: int = 300              # PDF processing DPI
    RAG_EXTRACT_FONTS: bool = False     # Extract font information
    RAG_EXTRACT_LAYOUT: bool = True     # Preserve document layout
    
    # ============================================================================
    # 🔧 PROCESSING OPTIMIZATION SETTINGS
    # ============================================================================
    RAG_MAX_PAGE_SIZE: int = 50         # Maximum pages to process
    RAG_TIMEOUT: int = 300              # Processing timeout in seconds
    RAG_MEMORY_LIMIT: str = "2GB"       # Memory limit for processing
    
    def __post_init__(self):
        """Set default values for lists."""
        if self.RAG_IMAGE_FORMATS is None:
            object.__setattr__(self, 'RAG_IMAGE_FORMATS', ['png', 'jpg', 'jpeg', 'tiff'])


# Create the PDF settings instance
pdf_settings = PDFSettings()