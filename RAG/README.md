# RAG Folder

## Overview
The `RAG` folder contains the core Retrieval Augmented Generation (RAG) system for gear wear analysis. It provides advanced document processing, intelligent question answering, and comprehensive evaluation capabilities.

## Purpose
- **Document Processing**: Load, chunk, and index technical documents
- **Intelligent Q&A**: Answer questions using document knowledge
- **Agent Routing**: Automatically route questions to specialized agents
- **Evaluation**: Assess system performance with RAGAS metrics
- **Modular Architecture**: Clean separation of concerns

## Contents

### Core Application (`app/`)

#### Main Components

##### `rag_service.py` (4.6KB, 159 lines)
- **Purpose**: Service layer for RAG operations
- **Features**:
  - Pipeline execution
  - Query processing
  - Agent integration
  - Error handling
  - Clean business logic separation

##### `pipeline.py` (9.5KB, 258 lines)
- **Purpose**: Main RAG pipeline orchestration
- **Features**:
  - Document loading and processing
  - Index building
  - Query routing
  - Result generation
  - Performance optimization

##### `config.py` (14KB, 304 lines)
- **Purpose**: Centralized configuration management
- **Features**:
  - Environment settings
  - Model configurations
  - Path management
  - Feature flags
  - Performance tuning

##### `rag_cli.py` (6.0KB, 197 lines)
- **Purpose**: Command-line interface
- **Features**:
  - Build pipeline
  - Query system
  - Evaluate performance
  - Status checking
  - Clean output formatting

### Modular Components

#### `Data_Management/`
- **Document Processing**: Loading, chunking, metadata attachment
- **Indexing**: Dense and sparse index creation
- **Normalization**: Document structure standardization
- **Storage**: Efficient data persistence

#### `Evaluation_Analysis/`
- **RAGAS Integration**: Performance evaluation metrics
- **Validation**: Input and output validation
- **Metrics**: Comprehensive performance analysis
- **Reporting**: Detailed evaluation reports

#### `Agent_Components/`
- **Question Routing**: Intelligent question classification
- **Specialized Agents**: Table, figure, summary agents
- **Answer Generation**: Context-aware responses
- **Agent Orchestration**: Multi-agent coordination

#### `Gradio_apps/`
- **Web Interface**: User-friendly web UI
- **Interactive Q&A**: Real-time question answering
- **Visualization**: Results display and charts
- **Debug Tools**: System monitoring and debugging

#### `Services/`
- **Orchestration**: High-level service coordination
- **Integration**: Component interaction management
- **Error Handling**: Comprehensive error management
- **Performance**: Optimization and caching

#### `Performance_Optimization/`
- **Caching**: In-memory and persistent caching
- **Batching**: Efficient batch processing
- **Memory Management**: Optimized memory usage
- **Profiling**: Performance monitoring

#### `retrieve_modules/`
- **Retrieval Strategies**: Multiple retrieval methods
- **Filtering**: Content-based filtering
- **Reranking**: Result reranking algorithms
- **Hybrid Search**: Combined dense/sparse search

#### `chunking_modules/`
- **Document Chunking**: Intelligent text segmentation
- **Overlap Management**: Chunk overlap strategies
- **Metadata Preservation**: Context preservation
- **Quality Control**: Chunk quality validation

#### `loader_modules/`
- **Document Loading**: Multi-format document support
- **Table Processing**: Table extraction and processing
- **Figure Processing**: Image and figure handling
- **Format Conversion**: Document format conversion

#### `pipeline_modules/`
- **Pipeline Stages**: Modular pipeline components
- **Stage Coordination**: Pipeline stage management
- **Error Recovery**: Pipeline error handling
- **Progress Tracking**: Pipeline progress monitoring

### Data Storage

#### `data/`
- **Document Storage**: Processed document storage
- **Index Data**: Vector index storage
- **Evaluation Data**: Test datasets and results
- **Configuration**: System configuration files

#### `index/`
- **Vector Database**: ChromaDB vector storage
- **Index Metadata**: Index configuration and metadata
- **Cache Storage**: Performance cache storage
- **Snapshot Storage**: Index snapshots for recovery

#### `logs/`
- **Application Logs**: System operation logs
- **Error Logs**: Error tracking and debugging
- **Performance Logs**: Performance monitoring data
- **Audit Logs**: System audit trail

## Architecture

### Service Layer Pattern
```
CLI/UI Layer
    ↓
Service Layer (rag_service.py)
    ↓
Orchestration Layer (pipeline.py)
    ↓
Component Layer (Data_Management, Agents, etc.)
    ↓
Infrastructure Layer (Storage, Logging, Config)
```

### Data Flow
```
Documents → Loaders → Chunking → Indexing → Storage
    ↓
Query → Routing → Retrieval → Reranking → Answer Generation
    ↓
Evaluation → Metrics → Reporting
```

## Key Features

### 1. Intelligent Document Processing
- **Multi-format Support**: PDF, Word, text files
- **Table Extraction**: Automatic table detection and processing
- **Figure Handling**: Image and diagram processing
- **Metadata Preservation**: Context and structure preservation

### 2. Advanced Question Answering
- **Agent Routing**: Automatic question classification
- **Hybrid Retrieval**: Dense and sparse search combination
- **Context Awareness**: Intelligent context selection
- **Answer Generation**: High-quality answer synthesis

### 3. Performance Optimization
- **Caching**: Multi-level caching system
- **Batching**: Efficient batch processing
- **Memory Management**: Optimized memory usage
- **Parallel Processing**: Concurrent operations

### 4. Comprehensive Evaluation
- **RAGAS Metrics**: Industry-standard evaluation
- **Custom Metrics**: Project-specific measurements
- **Performance Monitoring**: Real-time performance tracking
- **Quality Assurance**: Automated quality checks

## Usage

### Command Line Interface
```bash
# Build the RAG pipeline
python RAG/rag_cli.py build

# Query the system
python RAG/rag_cli.py query "What is gear wear?"

# Evaluate performance
python RAG/rag_cli.py evaluate

# Check system status
python RAG/rag_cli.py status
```

### Programmatic Usage
```python
from RAG.app.rag_service import RAGService

# Initialize service
service = RAGService()

# Run pipeline
result = service.run_pipeline()

# Query system
answer = service.query("What is the wear depth for case W15?")
```

### Web Interface
```bash
# Start Gradio interface
python RAG/app/ui_gradio.py
```

## Configuration

### Environment Variables
- **OPENAI_API_KEY**: OpenAI API key for LLM operations
- **GOOGLE_API_KEY**: Google API key for additional services
- **PROJECT_ROOT**: Project root directory path

### Configuration Files
- **config.py**: Centralized configuration management
- **settings.json**: User-specific settings
- **logging.conf**: Logging configuration

## Performance

### Optimization Features
- **Lazy Loading**: Components loaded on demand
- **Caching**: Multi-level caching for performance
- **Batching**: Efficient batch processing
- **Memory Management**: Optimized memory usage

### Monitoring
- **Performance Metrics**: Real-time performance tracking
- **Resource Usage**: Memory and CPU monitoring
- **Error Tracking**: Comprehensive error logging
- **Audit Trail**: Complete operation audit trail

## Security

### Data Protection
- **Input Validation**: Strict input validation
- **Path Security**: Secure file path handling
- **API Security**: Secure API key management
- **Error Handling**: No sensitive data exposure

### Access Control
- **File Permissions**: Secure file access
- **API Limits**: Rate limiting and quotas
- **Audit Logging**: Complete access audit trail

## Maintenance

### Code Quality
- **Modularity**: Clean component separation
- **Documentation**: Comprehensive inline documentation
- **Testing**: Extensive test coverage
- **Type Hints**: Full type annotation

### Scalability
- **Component Isolation**: Independent components
- **Interface Contracts**: Clear component interfaces
- **Dependency Management**: Clean dependency structure
- **Extension Points**: Easy feature addition

## Status: ✅ Production Ready
The RAG folder provides a comprehensive, production-ready RAG system with advanced features, excellent performance, and robust error handling.
