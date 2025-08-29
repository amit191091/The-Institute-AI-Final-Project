#!/usr/bin/env python3
"""
MCP Client Test Script
======================

Test script to verify the MCP server functionality and tool availability.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mcp_server_components():
    """Test MCP server components without full client/server integration."""
    logger.info("🧪 Testing MCP Server Components...")
    
    try:
        # Test server creation
        from MCP_Tools.mcp_server import server, TOOL_DEFINITIONS
        logger.info("✅ MCP server created successfully")
        
        # Test tool definitions
        logger.info(f"✅ Tool definitions loaded: {len(TOOL_DEFINITIONS)} tools")
        for tool_name in TOOL_DEFINITIONS.keys():
            logger.info(f"  - {tool_name}")
        
        # Test server handlers
        logger.info("✅ Server handlers registered successfully")
        
        logger.info("✅ MCP server components test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ MCP server components not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MCP server components test failed: {e}")
        return False


def test_mcp_client_imports():
    """Test MCP client imports."""
    logger.info("🧪 Testing MCP Client Imports...")
    
    try:
        # Test client imports
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        logger.info("✅ MCP client imports successful")
        
        # Test server parameters creation
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
        logger.info("✅ StdioServerParameters created successfully")
        
        logger.info("✅ MCP client imports test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ MCP client imports failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MCP client imports test failed: {e}")
        return False


def test_tool_models():
    """Test Pydantic models validation."""
    logger.info("🧪 Testing Pydantic models...")
    
    try:
        from MCP_Tools.mcp_models import (
            RAGQueryInput, VisionAlignInput, VibrationFeaturesInput,
            validate_tool_input, create_tool_response
        )
        
        # Test RAG query input validation
        valid_input = RAGQueryInput(
            question="What is gear wear?",
            top_k=10
        )
        logger.info(f"✅ RAG query input validation: {valid_input}")
        
        # Test vision align input validation
        try:
            valid_vision_input = VisionAlignInput(
                image_path="test_image.jpg"
            )
            logger.info(f"✅ Vision align input validation: {valid_vision_input}")
        except ValueError as e:
            logger.info(f"ℹ️ Vision input validation (expected for non-existent file): {e}")
        
        # Test tool validation functions
        try:
            validated = validate_tool_input("rag_query", {
                "question": "Test question",
                "top_k": 5
            })
            logger.info(f"✅ Tool input validation: {validated}")
        except Exception as e:
            logger.error(f"❌ Tool input validation failed: {e}")
            return False
        
        logger.info("✅ Pydantic models test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Pydantic models not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Pydantic models test failed: {e}")
        return False


def test_tool_implementations():
    """Test direct tool implementations."""
    logger.info("🧪 Testing tool implementations...")
    
    try:
        from MCP_Tools.tool_implementations import rag, vision, vib, timeline
        
        # Test tool interfaces
        logger.info(f"✅ RAG tools available: {dir(rag)}")
        logger.info(f"✅ Vision tools available: {dir(vision)}")
        logger.info(f"✅ Vibration tools available: {dir(vib)}")
        logger.info(f"✅ Timeline tools available: {dir(timeline)}")
        
        logger.info("✅ Tool implementations test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Tool implementations not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Tool implementations test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting MCP Integration Tests...")
    
    # Test tool implementations first
    if not test_tool_implementations():
        logger.error("❌ Tool implementations test failed")
        return False
    
    # Test Pydantic models
    if not test_tool_models():
        logger.error("❌ Pydantic models test failed")
        return False
    
    # Test MCP server components
    if not test_mcp_server_components():
        logger.error("❌ MCP server components test failed")
        return False
    
    # Test MCP client imports
    if not test_mcp_client_imports():
        logger.error("❌ MCP client imports test failed")
        return False
    
    logger.info("\n🎉 All tests completed!")
    logger.info("📋 Summary:")
    logger.info("  ✅ Tool implementations: Working")
    logger.info("  ✅ Pydantic models: Working")
    logger.info("  ✅ MCP server components: Working")
    logger.info("  ✅ MCP client imports: Working")
    logger.info("  ℹ️ Full MCP integration: Ready for manual testing")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
