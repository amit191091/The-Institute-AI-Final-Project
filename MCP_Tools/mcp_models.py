#!/usr/bin/env python3
"""
MCP Tool Input Models
====================

Pydantic models for validating MCP tool inputs with proper type checking
and documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, List, Dict, Any
from pathlib import Path


# ============================================================================
# RAG TOOL MODELS
# ============================================================================

class RAGIndexInput(BaseModel):
    """Input model for RAG indexing operations."""
    
    path: str = Field(
        ..., 
        description="Path to documents directory or file to index",
        examples=["documents/", "Gear wear Failure.pdf"]
    )
    clear: bool = Field(
        False, 
        description="Whether to clear existing index before indexing"
    )
    
    @validator('path')
    def validate_path(cls, v):
        """Validate that the path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


class RAGQueryInput(BaseModel):
    """Input model for RAG query operations."""
    
    question: str = Field(
        ..., 
        description="User question to query the RAG system",
        min_length=1,
        examples=["What is gear wear?", "How is wear measured?", "Show me wear depth data"]
    )
    top_k: int = Field(
        8, 
        description="Number of top documents to retrieve",
        ge=1,
        le=50
    )


class RAGEvaluateInput(BaseModel):
    """Input model for RAG evaluation operations."""
    
    eval_set: str = Field(
        ..., 
        description="Path to evaluation dataset or evaluation type",
        examples=["gear_wear_qa.jsonl", "default", "custom_eval.json"]
    )


# ============================================================================
# VISION TOOL MODELS
# ============================================================================

class VisionAlignInput(BaseModel):
    """Input model for image alignment operations."""
    
    image_path: str = Field(
        ..., 
        description="Path to image file to align",
        examples=["gear_image.jpg", "worn_gear.png", "healthy_gear.tiff"]
    )
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate that the image file exists and has valid extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image format. Supported: {valid_extensions}")
        
        return v


class VisionMeasureInput(BaseModel):
    """Input model for image measurement operations."""
    
    image_path: str = Field(
        ..., 
        description="Path to image file to measure",
        examples=["worn_gear.jpg", "gear_measurement.png"]
    )
    healthy_ref: Optional[str] = Field(
        None, 
        description="Path to healthy reference image for comparison",
        examples=["healthy_gear.jpg", "baseline_gear.png"]
    )
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate that the image file exists and has valid extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image format. Supported: {valid_extensions}")
        
        return v
    
    @validator('healthy_ref')
    def validate_healthy_ref(cls, v):
        """Validate that the healthy reference image exists if provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Healthy reference image does not exist: {v}")
            
            valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            if path.suffix.lower() not in valid_extensions:
                raise ValueError(f"Invalid image format. Supported: {valid_extensions}")
        
        return v


# ============================================================================
# VIBRATION TOOL MODELS
# ============================================================================

class VibrationFeaturesInput(BaseModel):
    """Input model for vibration feature extraction."""
    
    file: str = Field(
        ..., 
        description="Path to vibration CSV file",
        examples=["vibration_data.csv", "RMS15.csv", "FME_Values.csv"]
    )
    fs: int = Field(
        50000, 
        description="Sampling frequency in Hz",
        ge=1000,
        le=1000000
    )
    
    @validator('file')
    def validate_file(cls, v):
        """Validate that the vibration file exists and has valid extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Vibration file does not exist: {v}")
        
        valid_extensions = {'.csv', '.txt', '.dat'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid file format. Supported: {valid_extensions}")
        
        return v


# ============================================================================
# TIMELINE TOOL MODELS
# ============================================================================

class TimelineSummarizeInput(BaseModel):
    """Input model for timeline summarization."""
    
    doc_path: str = Field(
        ..., 
        description="Path to document to summarize and extract timeline",
        examples=["Gear wear Failure.pdf"]
    )
    mode: Literal["mapreduce", "refine"] = Field(
        "mapreduce", 
        description="Processing mode: 'mapreduce' for chunk-based processing, 'refine' for full document processing"
    )
    
    @validator('doc_path')
    def validate_doc_path(cls, v):
        """Validate that the document file exists and has valid extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Document file does not exist: {v}")
        
        valid_extensions = {'.pdf'} # {'.pdf', '.txt', '.md', '.docx', '.doc'} Optional
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid document format. Supported: {valid_extensions}")
        
        return v


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class MCPResponse(BaseModel):
    """Base response model for all MCP tools."""
    
    ok: bool = Field(..., description="Whether the operation was successful")
    run_id: str = Field(..., description="Unique run ID for tracking")
    timings: Dict[str, Any] = Field(..., description="Execution timing information")
    error: Optional[str] = Field(None, description="Error message if operation failed")


class RAGIndexResponse(MCPResponse):
    """Response model for RAG indexing operations."""
    
    indexed: int = Field(..., description="Number of documents indexed")
    snapshot: str = Field(..., description="Path to index snapshot file")


class RAGQueryResponse(MCPResponse):
    """Response model for RAG query operations."""
    
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    retrieved: List[str] = Field(..., description="Retrieved document contents")


class RAGEvaluateResponse(MCPResponse):
    """Response model for RAG evaluation operations."""
    
    scores: Dict[str, float] = Field(..., description="Evaluation scores")
    n: int = Field(..., description="Number of evaluation questions")


class VisionAlignResponse(MCPResponse):
    """Response model for image alignment operations."""
    
    aligned_path: str = Field(..., description="Path to aligned image file")
    transform: Dict[str, Any] = Field(..., description="Transformation parameters")


class VisionMeasureResponse(MCPResponse):
    """Response model for image measurement operations."""
    
    depth_um: float = Field(..., description="Wear depth in micrometers")
    area_um2: float = Field(..., description="Wear area in square micrometers")
    scale_um_per_px: float = Field(..., description="Scale conversion factor")
    method: str = Field(..., description="Measurement method used")


class VibrationFeaturesResponse(MCPResponse):
    """Response model for vibration feature extraction."""
    
    rms: float = Field(..., description="Root mean square value")
    bands: Dict[str, float] = Field(..., description="Frequency band values")
    peaks: List[float] = Field(..., description="Peak values detected")
    fs: int = Field(..., description="Sampling frequency used")
    units: Dict[str, str] = Field(..., description="Units for measurements")


class TimelineSummarizeResponse(MCPResponse):
    """Response model for timeline summarization."""
    
    timeline: List[Dict[str, str]] = Field(..., description="Extracted timeline events")
    mode: str = Field(..., description="Processing mode used")


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_MODELS = {
    "rag_index": {"input": RAGIndexInput, "response": RAGIndexResponse},
    "rag_query": {"input": RAGQueryInput, "response": RAGQueryResponse},
    "rag_evaluate": {"input": RAGEvaluateInput, "response": RAGEvaluateResponse},
    "vision_align": {"input": VisionAlignInput, "response": VisionAlignResponse},
    "vision_measure": {"input": VisionMeasureInput, "response": VisionMeasureResponse},
    "vib_features": {"input": VibrationFeaturesInput, "response": VibrationFeaturesResponse},
    "timeline_summarize": {"input": TimelineSummarizeInput, "response": TimelineSummarizeResponse},
}


def get_tool_model(tool_name: str) -> tuple[type[BaseModel], type[MCPResponse]]:
    """Get input and response models for a tool."""
    if tool_name not in TOOL_MODELS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    models = TOOL_MODELS[tool_name]
    return models["input"], models["response"]


def validate_tool_input(tool_name: str, arguments: dict) -> BaseModel:
    """Validate tool input arguments."""
    input_model, _ = get_tool_model(tool_name)
    return input_model(**arguments)


def create_tool_response(tool_name: str, result: dict) -> MCPResponse:
    """Create a validated tool response."""
    _, response_model = get_tool_model(tool_name)
    return response_model(**result)
