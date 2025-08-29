# MCP Integration Documentation

## Overview

This project now includes full **Model Context Protocol (MCP)** integration, allowing all gear wear analysis tools to be used through the MCP protocol. This enables seamless integration with MCP-compatible clients and AI assistants.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │───▶│   MCP Server     │───▶│  Tool Functions │
│   (AI Assistant)│    │  (mcp_server.py) │    │ (tool_impl.py)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Pydantic Models  │
                       │  (mcp_models.py) │
                       └──────────────────┘
```

## Available Tools

### 1. RAG Tools

#### `rag_index`
- **Purpose**: Index documents for RAG system
- **Input**: `path` (string), `clear` (boolean, optional)
- **Output**: Indexing results with document count and snapshot

#### `rag_query`
- **Purpose**: Query the RAG system with questions
- **Input**: `question` (string), `top_k` (integer, optional)
- **Output**: Answer with sources and retrieved documents

#### `rag_evaluate`
- **Purpose**: Evaluate RAG system performance
- **Input**: `eval_set` (string)
- **Output**: Evaluation scores and metrics

### 2. Vision Tools

#### `vision_align`
- **Purpose**: Align gear images for analysis
- **Input**: `image_path` (string)
- **Output**: Aligned image path and transformation data

#### `vision_measure`
- **Purpose**: Measure wear depth and area from images
- **Input**: `image_path` (string), `healthy_ref` (string, optional)
- **Output**: Wear measurements in micrometers

### 3. Vibration Tools

#### `vib_features`
- **Purpose**: Extract vibration features from CSV files
- **Input**: `file` (string), `fs` (integer, optional)
- **Output**: RMS values, frequency bands, and peak data

### 4. Timeline Tools

#### `timeline_summarize`
- **Purpose**: Summarize documents and extract timeline events
- **Input**: `doc_path` (string), `mode` (string, optional)
- **Output**: Timeline events and processing mode

## Installation

### 1. Install Dependencies
```bash
pip install mcp pydantic
```

### 2. Verify Installation
```bash
python test_mcp_client.py
```

## Usage

### Method 1: Direct Python Usage
```python
from tool_implementations import rag, vision, vib, timeline

# RAG operations
result = rag.index("documents/", clear=True)
answer = rag.query("What is gear wear?", top_k=8)

# Vision operations
aligned = vision.align("gear_image.jpg")
measurements = vision.measure("worn_gear.jpg", "healthy_gear.jpg")

# Vibration analysis
features = vib.features("vibration_data.csv", fs=50000)

# Timeline extraction
timeline = timeline.summarize("document.pdf", mode="mapreduce")
```

### Method 2: MCP Protocol Usage
```python
from mcp.client import ClientSession
from mcp.types import StdioServerParameters

# Connect to MCP server
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"]
)

async with ClientSession(server_params) as session:
    # List available tools
    tools = await session.list_tools()
    
    # Call tools
    result = await session.call_tool(
        "rag_query",
        {"question": "What is gear wear?", "top_k": 5}
    )
```

### Method 3: MCP Client Configuration
Update your MCP client configuration:

```json
{
  "mcpServers": {
    "gear-wear-analysis": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

## Input Validation

All tool inputs are validated using Pydantic models:

```python
from mcp_models import RAGQueryInput

# Valid input
valid_input = RAGQueryInput(
    question="What is gear wear?",
    top_k=10
)

# Invalid input (will raise validation error)
try:
    invalid_input = RAGQueryInput(
        question="",  # Empty question
        top_k=100     # Too many documents
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Response Format

All tools return standardized responses:

```python
{
    "ok": True,
    "run_id": "uuid-string",
    "timings": {
        "start": 1234567890.123,
        "end": 1234567890.456,
        "duration": 0.333
    },
    # Tool-specific data
    "answer": "Generated answer...",
    "sources": [...],
    "retrieved": [...]
}
```

## Error Handling

Tools handle errors gracefully:

```python
{
    "ok": False,
    "error": "File not found: missing_file.pdf",
    "run_id": "error_rag_index_1234567890",
    "timings": {
        "error": True,
        "function": "rag_index"
    }
}
```

## Testing

### Run All Tests
```bash
python test_mcp_client.py
```

### Test Individual Components
```python
# Test Pydantic models
from mcp_models import test_tool_models
test_tool_models()

# Test tool implementations
from test_mcp_client import test_tool_implementations
test_tool_implementations()

# Test MCP server
import asyncio
from test_mcp_client import test_mcp_server
asyncio.run(test_mcp_server())
```

## Configuration

### Environment Variables
- `PYTHONPATH`: Set to project root for imports
- `RAG_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### MCP Server Settings
- **Server Name**: `gear-wear-analysis`
- **Protocol**: MCP v1.0
- **Transport**: stdio
- **Validation**: Pydantic models

## Security

### Input Validation
- All file paths are validated for existence
- File extensions are checked for supported formats
- Input parameters have type and range validation

### Output Sanitization
- Sensitive data is filtered from responses
- File paths are validated before returning
- Error messages don't expose internal details

## Performance

### Optimization Features
- **Lazy Loading**: Tools are imported only when needed
- **Caching**: Results are cached for repeated operations
- **Timing**: All operations include timing information
- **Async Support**: MCP server supports async operations

### Monitoring
- **Run IDs**: Every operation has a unique identifier
- **Timing Data**: Execution time is tracked
- **Error Logging**: Comprehensive error logging
- **Performance Metrics**: Built-in performance monitoring

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Solution: Set PYTHONPATH
   export PYTHONPATH=/path/to/project
   ```

2. **File Not Found**
   ```bash
   # Solution: Check file paths
   ls -la "Pictures and Vibrations database/Vibration/database/"
   ```

3. **MCP Server Not Starting**
   ```bash
   # Solution: Check dependencies
   pip install mcp pydantic
   python mcp_server.py
   ```

### Debug Mode
```bash
# Enable debug logging
export RAG_LOG_LEVEL=DEBUG
python mcp_server.py
```

## Examples

### Complete Workflow
```python
from tool_implementations import rag, vision, vib

# 1. Index documents
index_result = rag.index("Gear wear Failure.pdf", clear=True)
print(f"Indexed {index_result['indexed']} documents")

# 2. Query the system
query_result = rag.query("What is the wear depth for case W15?")
print(f"Answer: {query_result['answer']}")

# 3. Analyze vibration data
vib_result = vib.features("vibration_data.csv")
print(f"RMS: {vib_result['rms']}")

# 4. Measure wear from image
measure_result = vision.measure("worn_gear.jpg", "healthy_gear.jpg")
print(f"Wear depth: {measure_result['depth_um']} μm")
```

### MCP Integration Example
```python
import asyncio
from mcp.client import ClientSession
from mcp.types import StdioServerParameters

async def analyze_gear_wear():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    async with ClientSession(server_params) as session:
        # Get available tools
        tools = await session.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")
        
        # Perform analysis
        result = await session.call_tool(
            "rag_query",
            {"question": "What are the main causes of gear wear?"}
        )
        
        print(f"Analysis result: {result.content}")

# Run the analysis
asyncio.run(analyze_gear_wear())
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite: `python test_mcp_client.py`
3. Review the logs for detailed error information
4. Verify all dependencies are installed correctly
