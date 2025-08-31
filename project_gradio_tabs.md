# Project Gradio Interface Tabs

## Overview
This document describes the actual tabs implemented in the project's Gradio interface for the Hybrid RAG system focused on failure reports analysis.

## Interface Title
**"Hybrid RAG – Failure Reports"**
*Subtitle: "Router + Summary / Needle / Table QA"*

## Tabs in the Current Implementation

### 1. **Ask Tab**
**Purpose**: Main Q&A interface for the RAG system

**Components**:
- **Question Input**: Textbox with placeholder "Ask about figures, tables, procedures, conclusions…"
- **Debug Toggle**: Checkbox to show/hide retrieval debug information
- **Ask Button**: Primary button to submit questions
- **Answer Display**: Markdown area for displaying responses
- **Metrics Display**: Textbox showing evaluation metrics (3 lines)
- **Figure Preview**: Image component for displaying relevant figures
- **Debug Accordion**: Collapsible section with detailed debug information:
  - Router information (Markdown)
  - Filters (JSON)
  - Dense retrieval results (Markdown)
  - Sparse retrieval results (Markdown)
  - Hybrid retrieval results (Markdown)
  - Top documents (DataFrame)
  - Answer vs Reference comparison (JSON)
  - Reasoning trace (JSON)

### 2. **Figures Tab**
**Purpose**: Display extracted figures from documents

**Components**:
- **Figure Gallery**: Grid display of extracted figures
- **Sorting**: Figures are automatically sorted by:
  - Figure number
  - Figure order
  - Page number
  - Anchor text
- **Fallback Message**: Shows "(No extracted figures. Enable RAG_EXTRACT_IMAGES=true and rerun.)" if no figures are found

### 3. **Agent Tab**
**Purpose**: Agent trace and tool visibility for debugging and maintenance

**Components**:
- **Question Input**: Textbox for agent questions
- **Run Agent Button**: Execute agent with tools
- **Trace Display**: JSON output showing agent execution steps
- **Result Display**: Markdown area for agent results

**Maintenance Tools**:
- **Figure Audit**: Button to audit and fix missing figure numbers/orders
- **Audit Summary**: JSON output showing audit results
- **Planner**: Text input for observations and button to generate fix plans
- **Plan Output**: Markdown area for generated plans

### 4. **Inspect Tab**
**Purpose**: Browse indexed documents for inspection

**Components**:
- **Sample Documents**: Display of top 12 indexed documents
- **Document Preview**: Textbox showing document contexts and metadata
- **Format**: Shows file name, page, section, and content preview

### 5. **Graph Tab**
**Purpose**: Knowledge graph visualization and database interaction

**Components**:
- **Graph Source Dropdown**: Choose from:
  - "Docs co-mention (default)"
  - "Normalized graph.json"
  - "Normalized chunks"
  - "Neo4j (live)"
- **Generate/Refresh Button**: Build or update the knowledge graph
- **Graph Display**: HTML iframe showing the generated graph
- **Status Display**: Markdown showing graph generation status

**Neo4j Integration**:
- **Cypher Query Input**: Textbox for database queries
- **Run Cypher Button**: Execute queries against Neo4j
- **Query Results**: JSON output showing query results

### 6. **DB Explorer Tab**
**Purpose**: Browse and filter indexed documents with advanced search

**Components**:
- **Section Filter**: Dropdown to filter by document sections
- **Text Search**: Textbox for searching in document content and metadata
- **Refresh Button**: Update the document table
- **Document Table**: Interactive DataFrame showing:
  - File name
  - Page number
  - Section type
  - Anchor text
  - Word count
  - Figure number
  - Figure order
  - Table markdown path
  - Table CSV path
  - Image path
  - Content preview

## Key Features

### Debug Capabilities
- Comprehensive debug information in the Ask tab
- Agent trace visibility
- Retrieval process transparency
- Performance metrics display

### Document Management
- Multiple document viewing modes (Inspect, DB Explorer)
- Advanced filtering and search capabilities
- Figure extraction and display
- Table and figure metadata tracking

### Knowledge Graph
- Multiple graph generation sources
- Interactive graph visualization
- Neo4j database integration
- Cypher query interface

### Agent Tools
- Question analysis tools
- Retrieval candidate tools
- Figure listing and auditing
- Maintenance planning capabilities

## Technical Implementation

### File Structure
- **Main UI File**: `app/ui_gradio.py`
- **UI Building Function**: `build_ui(docs, hybrid, llm, debug=None)`
- **Event Handler**: `on_ask(q, debug_toggle)`

### Dependencies
- **Gradio**: For the web interface
- **NetworkX**: For graph operations
- **PyVis**: For graph visualization
- **Neo4j**: For graph database operations (optional)

### Configuration
- **RAG_EXTRACT_IMAGES**: Environment variable to enable figure extraction
- **Logs Directory**: Stores query logs and generated graphs
- **Normalized Data**: Supports normalized graph and chunk data formats

## Usage Workflow

1. **Start with Ask Tab**: Primary interface for Q&A
2. **Use Figures Tab**: Browse extracted visual content
3. **Debug with Agent Tab**: Understand system behavior
4. **Inspect Documents**: Review indexed content
5. **Explore Graph**: Visualize knowledge relationships
6. **Search DB**: Advanced document filtering and search

This interface provides a comprehensive toolkit for working with the Hybrid RAG system, from basic Q&A to advanced debugging and document exploration.
