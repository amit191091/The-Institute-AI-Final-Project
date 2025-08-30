# 🤖 **RAG SYSTEM (Retrieval-Augmented Generation)**

## 📋 **OVERVIEW**

The RAG system is an advanced AI-powered document analysis and question answering system specifically designed for gear wear analysis. It combines document processing, vector search, and AI agents to provide intelligent answers to technical questions.

---

## 🗂️ **FOLDER STRUCTURE**

```
RAG/
├── app/                        # Main application code
├── data/                       # Training data and datasets
├── logs/                       # System logs and metrics
├── rag_cli.py                  # Command-line interface
└── README.md                   # RAG system documentation
```

---

## 🧠 **RAG/APP - APPLICATION CORE**

### **Purpose**
Core RAG system implementation with modular, scalable architecture

### **Location**
`RAG/app/`

### **Key Components**

#### **📁 Services/**
- **`rag_orchestrator.py`**: Main service coordination and orchestration
- **`document_service.py`**: Document processing operations
- **`query_service.py`**: Query processing and routing
- **`indexing_service.py`**: Vector index management
- **`embedding_service.py`**: Text embedding operations
- **`llm_service.py`**: Language model integration
- **`evaluation_service.py`**: System evaluation and metrics

#### **📁 Agent_Components/**
- **`agents.py`**: AI agent implementations
- **`agent_tools.py`**: Agent utility functions
- **`prompts.py`**: Prompt management and templates
- **`answer_generation.py`**: Answer generation service
- **`query_processing.py`**: Query analysis and processing

#### **📁 Data_Management/**
- **`normalize_snapshot.py`**: Data normalization and processing
- **`indexing.py`**: Index operations and management
- **`metadata.py`**: Metadata extraction and classification
- **`chunk_generators.py`**: Document chunking strategies
- **`date_parsers.py`**: Date extraction and parsing
- **`measurement_extractors.py`**: Measurement data extraction
- **`text_processors.py`**: Text processing utilities
- **`io_handlers.py`**: Input/output operations
- **`normalized_loader.py`**: Normalized data loading
- **`graph_builders.py`**: Knowledge graph construction

#### **📁 Performance_Optimization/**
- **`dependency_injection.py`**: DI container and service management
- **`caching.py`**: Performance caching mechanisms
- **`batching.py`**: Batch processing utilities

#### **📁 chunking_modules/**
- **`chunking_config.py`**: Chunking configuration
- **`advanced_chunking.py`**: Advanced chunking strategies
- **`chunking_utils.py`**: Chunking utilities
- **`chunking_text.py`**: Text-specific chunking
- **`chunking_table.py`**: Table chunking
- **`chunking_figure.py`**: Figure chunking
- **`heading_detection.py`**: Heading detection algorithms

#### **📁 loader_modules/**
- **`pdf_loaders.py`**: PDF document loading
- **`excel_loaders.py`**: Excel file loading
- **`table_extractors.py`**: Table extraction utilities
- **`table_utils.py`**: Table processing utilities
- **`loader_utils.py`**: General loading utilities
- **`element_types.py`**: Element type definitions
- **`llamaparse_extractor.py`**: LlamaParse integration

#### **📁 retrieve_modules/**
- **`retrieve_query_analyzer.py`**: Query analysis
- **`retrieve_filters.py`**: Document filtering
- **`retrieve_hybrid.py`**: Hybrid retrieval strategies
- **`retrieve_fallbacks.py`**: Fallback mechanisms
- **`retrieve_wear_range.py`**: Wear range retrieval
- **`query_intent.py`**: Query intent analysis
- **`reranker_ce.py`**: Cross-encoder reranking

#### **📁 pipeline_modules/**
- **`pipeline_core.py`**: Core pipeline implementation
- **`pipeline_ingestion.py`**: Document ingestion
- **`pipeline_query.py`**: Query processing pipeline
- **`pipeline_utils.py`**: Pipeline utilities
- **`integrated_rag_pipeline.py`**: Integrated pipeline
- **`rag_pipeline.py`**: RAG pipeline implementation
- **`router_chain.py`**: Router chain implementation
- **`graph.py`**: Graph operations
- **`graphdb.py`**: Graph database operations
- **`graphdb_import_normalized.py`**: Graph import utilities
- **`clean_table_extract.py`**: Table extraction cleaning
- **`llamaindex_export.py`**: LlamaIndex export
- **`llamaindex_compare.py`**: LlamaIndex comparison

#### **📁 Evaluation_Analysis/**
- **`evaluation_utils.py`**: Evaluation utilities
- **`validate.py`**: Validation functions
- **`answer_evaluators.py`**: Answer evaluation
- **`system_evaluators.py`**: System evaluation
- **`compliance_checkers.py`**: Compliance checking
- **`progress_tracking.py`**: Progress tracking
- **`deepeval_integration.py`**: DeepEval integration

#### **📁 Gradio_apps/**
- **`ui_handlers.py`**: UI handlers
- **`ui_data_loader.py`**: Data loading UI
- **`ui_tabs.py`**: Tab components
- **`UI_Handlers/`**: UI handler modules

### **Core Files**
- **`config.py`**: Centralized configuration management
- **`interfaces.py`**: Service interface definitions
- **`types.py`**: Type definitions
- **`rag_service.py`**: Main RAG service
- **`agent_orchestrator.py`**: Agent orchestration
- **`pipeline.py`**: Pipeline orchestration
- **`chunking.py`**: Chunking orchestration
- **`retrieve.py`**: Retrieval orchestration
- **`loaders.py`**: Loader orchestration
- **`logger.py`**: Centralized logging
- **`utils.py`**: Utility functions
- **`table_ops.py`**: Table operations
- **`fact_miner.py`**: Fact mining utilities
- **`ui_gradio.py`**: Gradio UI integration

---

## 🚀 **FEATURES**

### **Document Processing**
- ✅ **PDF Processing**: Advanced PDF text and table extraction
- ✅ **Table Extraction**: Intelligent table detection and parsing
- ✅ **Image Processing**: Figure and diagram analysis
- ✅ **Multi-format Support**: PDF, DOC, Excel, and more

### **AI-Powered Q&A**
- ✅ **Natural Language**: Human-like question understanding
- ✅ **Context-Aware**: Intelligent context retrieval
- ✅ **Multi-source Answers**: Information from multiple documents
- ✅ **Source Attribution**: Clear source references

### **Vector Search**
- ✅ **Semantic Search**: Meaning-based document retrieval
- ✅ **Hybrid Retrieval**: Combining dense and sparse methods
- ✅ **Reranking**: Advanced result ranking
- ✅ **Filtering**: Intelligent document filtering

### **Evaluation Framework**
- ✅ **RAGAS Integration**: Automated evaluation metrics
- ✅ **DeepEval Support**: Comprehensive evaluation tools
- ✅ **Performance Monitoring**: Real-time system metrics
- ✅ **Quality Assessment**: Answer quality evaluation

---

## 🔄 **SYSTEM ARCHITECTURE**

### **Data Flow**
1. **Document Ingestion**: Load and process documents
2. **Chunking**: Break documents into manageable pieces
3. **Embedding**: Convert text to vector representations
4. **Indexing**: Store vectors for fast retrieval
5. **Query Processing**: Analyze user questions
6. **Retrieval**: Find relevant document chunks
7. **Answer Generation**: Generate AI-powered answers
8. **Evaluation**: Assess answer quality

### **Component Interaction**
- **Service Layer**: Business logic and orchestration
- **Data Layer**: Document processing and storage
- **AI Layer**: Language models and embeddings
- **UI Layer**: User interfaces and interactions
- **Evaluation Layer**: Quality assessment and metrics

---

## 🎯 **USE CASES**

### **Technical Documentation**
- **Manual Analysis**: Extract information from technical manuals
- **Specification Review**: Analyze product specifications
- **Procedure Understanding**: Understand complex procedures
- **Troubleshooting**: Find solutions to technical problems

### **Gear Wear Analysis**
- **Wear Assessment**: Analyze wear patterns and measurements
- **Trend Analysis**: Monitor wear progression over time
- **Comparison Studies**: Compare different gear conditions
- **Predictive Analysis**: Predict future wear patterns

### **Research & Development**
- **Literature Review**: Analyze research papers and reports
- **Data Mining**: Extract insights from large document collections
- **Knowledge Discovery**: Find hidden patterns and relationships
- **Documentation Analysis**: Understand complex documentation

---

## 📊 **PERFORMANCE**

### **Scalability**
- **Modular Design**: Easy to extend and modify
- **Performance Optimization**: Caching and batching
- **Resource Management**: Efficient memory and CPU usage
- **Horizontal Scaling**: Support for distributed deployment

### **Accuracy**
- **Advanced Retrieval**: Hybrid search methods
- **Context Understanding**: Deep context analysis
- **Source Verification**: Reliable source attribution
- **Quality Metrics**: Continuous quality monitoring

---

## 🔧 **CONFIGURATION**

### **Environment Variables**
- **API Keys**: OpenAI, Google, and other service keys
- **Model Settings**: Language model configurations
- **Performance Settings**: Caching and optimization
- **Evaluation Settings**: RAGAS and DeepEval configuration

### **Configuration Files**
- **`config.py`**: Centralized configuration management
- **Environment Files**: `.env` for sensitive settings
- **Model Configs**: Language model parameters
- **System Settings**: Performance and optimization

---

## 🚀 **USAGE**

### **Starting the System**
```bash
# Start RAG system
python Main.py start

# Start with headless mode
python Main.py start --headless

# Build pipeline
python Main.py build

# Run evaluation
python Main.py evaluate
```

### **CLI Interface**
```bash
# Query the system
python Main.py query "What is the wear depth for case W15?"

# Build pipeline
python Main.py build --normalized

# Run evaluation
python Main.py evaluate --questions 50
```

### **API Usage**
```python
from RAG.app.rag_service import RAGService

# Initialize service
service = RAGService()

# Run pipeline
result = service.run_pipeline()

# Query system
answer = service.query("What causes gear wear?")
```

---

## 📈 **OUTPUTS**

### **Answer Generation**
- **Structured Responses**: Well-formatted answers
- **Source References**: Clear source attribution
- **Confidence Scores**: Answer confidence metrics
- **Multiple Formats**: Text, JSON, and structured outputs

### **Evaluation Results**
- **RAGAS Metrics**: Automated evaluation scores
- **Performance Reports**: System performance analysis
- **Quality Assessment**: Answer quality evaluation
- **Comparative Analysis**: System comparison results

---

## 🔍 **MONITORING**

### **Logs**
- **Application Logs**: System operation logs
- **Query Logs**: User query tracking
- **Performance Logs**: System performance metrics
- **Error Logs**: Error tracking and debugging

### **Metrics**
- **Response Time**: Query response times
- **Accuracy Metrics**: Answer accuracy measurements
- **Usage Statistics**: System usage patterns
- **Resource Utilization**: CPU and memory usage

---

## 🎯 **CONCLUSION**

The RAG system provides **advanced AI-powered document analysis** with:

- **Comprehensive Document Processing**: Multi-format support
- **Intelligent Question Answering**: Natural language understanding
- **High-Performance Retrieval**: Fast and accurate search
- **Quality Evaluation**: Continuous quality assessment
- **Scalable Architecture**: Production-ready design

**🏆 This is a state-of-the-art RAG system for technical document analysis and gear wear research!**
