# Gradio Script Comparison

## Overview
This document compares two different approaches to building a Gradio interface for a RAG (Retrieval-Augmented Generation) system.

## Script 1: Monolithic Approach
**File**: Single file with `build_ui()` function

### Characteristics:
- **Structure**: All code in one monolithic function
- **Organization**: UI building, event handlers, and logic mixed together
- **Functionality**: Basic RAG system with simple exit button
- **Exit Handling**: Simple exit with `gr.close_all()` and `sys.exit(0)`
- **Evaluation**: Basic RAGAS evaluation
- **Code Reusability**: Hard to reuse components

## Script 2: Modular Approach
**Files**: Multiple focused modules

### Characteristics:
- **Structure**: Modular approach with separate files for different components
- **Organization**: Separated into focused modules:
  - `ui_tabs.py` - Tab-specific UI builders
  - `ui_qa_handlers.py` - Question & answer event handlers
  - `ui_graph_handlers.py` - Graph and database handlers
  - `ui_evaluation.py` - Evaluation functionality
  - `ui_data_loader.py` - Data loading utilities
  - `ui_components.py` - Reusable UI components
- **Functionality**: Advanced features with enhanced evaluation system
- **Exit Handling**: Robust exit handling with multiple shutdown options
- **Evaluation**: Enhanced evaluation with auto-evaluator and synthetic ground truth
- **Code Reusability**: Modular design for easy reuse and maintenance

## Tabs in the Functional Gradio Interface

The modular Gradio interface includes the following tabs:

### 1. **Ask Tab**
- **Purpose**: Main Q&A interface for the RAG system
- **Features**:
  - Question input textbox
  - Ground truth input (optional)
  - Debug checkbox for retrieval trace
  - Ask and Clear buttons
  - Shutdown and Exit buttons
  - Answer display with markdown formatting
  - Evaluation metrics display
  - Inline figure preview
  - System info panel
  - Debug accordion with detailed retrieval information

### 2. **Figures Tab**
- **Purpose**: Display extracted figures from documents
- **Features**:
  - Gallery view of extracted figures
  - Sorted by figure number/order
  - Automatic figure extraction when `RAG_EXTRACT_IMAGES=true`

### 3. **Agent Tab**
- **Purpose**: Agent trace and tool visibility
- **Features**:
  - Question selection from file
  - Question loading functionality
  - Agent execution with tool visibility
  - Trace display in JSON format
  - Maintenance tools for figure auditing
  - Planner for fixing database issues

### 4. **Inspect Tab**
- **Purpose**: Browse indexed documents
- **Features**:
  - Sample of top indexed documents
  - Document context preview
  - Limited to 12 sample documents

### 5. **Graph Tab**
- **Purpose**: Knowledge graph visualization
- **Features**:
  - Auto-built knowledge graph
  - Multiple graph sources (docs co-mention, normalized graph, Neo4j)
  - Graph generation and refresh
  - Neo4j Cypher query interface
  - Interactive graph visualization

### 6. **Evaluation Tab**
- **Purpose**: RAGAS evaluation tools
- **Features**:
  - Google API testing
  - RAGAS testing
  - Ground truth generation
  - RAG system evaluation
  - Configurable number of questions for evaluation

### 7. **DB Explorer Tab**
- **Purpose**: Browse and filter indexed documents
- **Features**:
  - Section-based filtering
  - Text search in documents and metadata
  - Interactive data table
  - Document preview with file, page, and section information
  - Refresh functionality

## Key Differences Summary

| Aspect | Monolithic Script | Modular Script |
|--------|------------------|----------------|
| **Structure** | Single file | Multiple modules |
| **Maintainability** | Low | High |
| **Reusability** | Low | High |
| **Features** | Basic | Advanced |
| **Tabs** | 7 tabs with basic functionality | 7 tabs with enhanced features |
| **Evaluation** | Basic RAGAS | Enhanced with synthetic ground truth |
| **Error Handling** | Basic | Robust |
| **Debug Information** | Limited | Comprehensive |

## Conclusion

The modular approach (Script 2) represents a more mature and maintainable implementation of the Gradio interface, with better separation of concerns, enhanced functionality, and improved user experience through comprehensive tabbed interface.
