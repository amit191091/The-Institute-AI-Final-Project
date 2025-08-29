#!/usr/bin/env python3
"""
MCP Server for Gear Wear Analysis Tools
======================================

Model Context Protocol server that exposes all gear wear analysis tools
with proper validation, error handling, and auditability.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path (go up one level to main project)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp.server import Server
from mcp.types import Tool
from mcp import stdio_server

# Import our tool implementations and models
from MCP_Tools.tool_implementations import (
    rag_index, rag_query, rag_evaluate,
    vision_align, vision_measure,
    vib_features, timeline_summarize
)
from MCP_Tools.mcp_models import validate_tool_input, create_tool_response, TOOL_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("gear-wear-analysis")

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

TOOL_DEFINITIONS = {
    "rag_index": {
        "name": "rag_index",
        "description": "Index documents for RAG system. Processes PDFs, Word documents, and other files to create a searchable index for question answering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to documents directory or file to index",
                    "examples": ["documents/", "Gear wear Failure.pdf"]
                },
                "clear": {
                    "type": "boolean",
                    "description": "Whether to clear existing index before indexing",
                    "default": False
                }
            },
            "required": ["path"]
        }
    },
    
    "rag_query": {
        "name": "rag_query",
        "description": "Query the RAG system with questions about gear wear analysis, measurements, and technical data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "User question to query the RAG system",
                    "examples": ["What is gear wear?", "How is wear measured?", "Show me wear depth data"]
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top documents to retrieve",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": ["question"]
        }
    },
    
    "rag_evaluate": {
        "name": "rag_evaluate",
        "description": "Evaluate the RAG system performance using RAGAS metrics on test questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "eval_set": {
                    "type": "string",
                    "description": "Path to evaluation dataset or evaluation type",
                    "examples": ["gear_wear_qa.jsonl", "default", "custom_eval.json"]
                }
            },
            "required": ["eval_set"]
        }
    },
    
    "vision_align": {
        "name": "vision_align",
        "description": "Align gear images for analysis by detecting gear centers and applying transformations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to image file to align",
                    "examples": ["gear_image.jpg", "worn_gear.png", "healthy_gear.tiff"]
                }
            },
            "required": ["image_path"]
        }
    },
    
    "vision_measure": {
        "name": "vision_measure",
        "description": "Measure wear depth and area from gear images using computer vision techniques.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to image file to measure",
                    "examples": ["worn_gear.jpg", "gear_measurement.png"]
                },
                "healthy_ref": {
                    "type": "string",
                    "description": "Path to healthy reference image for comparison",
                    "examples": ["healthy_gear.jpg", "baseline_gear.png"]
                }
            },
            "required": ["image_path"]
        }
    },
    
    "vib_features": {
        "name": "vib_features",
        "description": "Extract vibration features from CSV files including RMS values, frequency bands, and peak detection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to vibration CSV file",
                    "examples": ["vibration_data.csv", "RMS15.csv", "FME_Values.csv"]
                },
                "fs": {
                    "type": "integer",
                    "description": "Sampling frequency in Hz",
                    "default": 50000,
                    "minimum": 1000,
                    "maximum": 1000000
                }
            },
            "required": ["file"]
        }
    },
    
    "timeline_summarize": {
        "name": "timeline_summarize",
        "description": "Summarize documents and extract timeline events using mapreduce or refine processing modes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_path": {
                    "type": "string",
                    "description": "Path to document to summarize and extract timeline",
                    "examples": ["Gear wear Failure.pdf"]
                },
                "mode": {
                    "type": "string",
                    "enum": ["mapreduce", "refine"],
                    "description": "Processing mode: 'mapreduce' for chunk-based processing, 'refine' for full document processing",
                    "default": "mapreduce"
                }
            },
            "required": ["doc_path"]
        }
    }
}

# ============================================================================
# MCP SERVER HANDLERS
# ============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools."""
    tools = []
    for tool_name, definition in TOOL_DEFINITIONS.items():
        tool = Tool(
            name=definition["name"],
            description=definition["description"],
            inputSchema=definition["inputSchema"]
        )
        tools.append(tool)
    
    logger.info(f"Registered {len(tools)} tools: {list(TOOL_DEFINITIONS.keys())}")
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call a specific tool with validated arguments."""
    try:
        logger.info(f"Calling tool '{name}' with arguments: {arguments}")
        
        # Validate input arguments
        validated_args = validate_tool_input(name, arguments)
        
        # Map tool names to functions
        tool_functions = {
            "rag_index": rag_index,
            "rag_query": rag_query,
            "rag_evaluate": rag_evaluate,
            "vision_align": vision_align,
            "vision_measure": vision_measure,
            "vib_features": vib_features,
            "timeline_summarize": timeline_summarize
        }
        
        if name not in tool_functions:
            raise ValueError(f"Unknown tool: {name}")
        
        # Call the tool function
        tool_func = tool_functions[name]
        result = tool_func(**validated_args.dict())
        
        # Validate and return response
        validated_response = create_tool_response(name, result)
        
        logger.info(f"Tool '{name}' completed successfully")
        return validated_response.dict()
        
    except Exception as e:
        logger.error(f"Error in tool '{name}': {str(e)}")
        
        # Return error response
        error_response = {
            "ok": False,
            "error": str(e),
            "run_id": f"error_{name}_{asyncio.get_event_loop().time()}",
            "timings": {
                "error": True,
                "function": name
            }
        }
        
        return error_response


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the MCP server."""
    try:
        logger.info("🚀 Starting Gear Wear Analysis MCP Server...")
        
        # Verify all tools are available
        tool_functions = {
            "rag_index": rag_index,
            "rag_query": rag_query,
            "rag_evaluate": rag_evaluate,
            "vision_align": vision_align,
            "vision_measure": vision_measure,
            "vib_features": vib_features,
            "timeline_summarize": timeline_summarize
        }
        
        for tool_name, func in tool_functions.items():
            if not callable(func):
                raise RuntimeError(f"Tool function '{tool_name}' is not callable")
        
        logger.info("✅ MCP Server initialized successfully")
        logger.info(f"📋 Available tools: {list(tool_functions.keys())}")
        
        # Run the server using stdio_server
        stdio_server(server)
        
    except KeyboardInterrupt:
        logger.info("🛑 MCP Server stopped by user")
    except Exception as e:
        logger.error(f"❌ MCP Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
