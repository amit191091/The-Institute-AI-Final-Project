# 🔧 Hybrid RAG for Gear & Bearing Failure Analysis

**Metadata-Driven Document Analysis with Multi-Agent Intelligence**

A sophisticated hybrid retrieval-augmented generation (RAG) system specifically designed for analyzing gear and bearing failure reports, integrating seamlessly with existing gear analysis capabilities.

## 🌟 Features

### 🧠 Advanced RAG Capabilities
- **Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25) + Reranking for optimal context retrieval
- **Multi-Agent System**: Router + 3 specialized agents (Summary, Needle, Table)
- **Metadata-Driven**: ≥5 metadata fields per chunk with intelligent filtering
- **Structure-Aware Chunking**: Respects document structure (tables, figures, sections)
- **Token Budget Management**: Adaptive chunking with 250-500 token average, 800 max for tables

### 📄 Document Processing
- **Multi-Format Support**: PDF, DOCX, DOC, TXT
- **Intelligent Parsing**: Preserves page numbers, anchors, and table structure
- **Content Distillation**: Extracts core 5% information from each section
- **Validation System**: Ensures documents meet quality criteria (≥10 pages, proper structure)

### 🔍 Specialized Agents
1. **Summary Agent**: Comprehensive overviews and multi-document synthesis
2. **Needle Agent**: Precise information extraction with exact citations
3. **Table Agent**: Quantitative data analysis and calculations
4. **Integration Agent**: Combines document insights with live gear analysis

### 🔬 Gear Analysis Integration
- **Picture Analysis**: Visual wear detection and measurement
- **Vibration Analysis**: Signal processing for fault detection
- **Cross-Reference**: Historical document patterns vs. current conditions
- **Comprehensive Assessment**: Unified reporting across all analysis types

### 🎛️ User Interfaces
- **Gradio Web Interface**: Interactive document upload, validation, and querying
- **Command Line Interface**: Batch processing and scripted workflows
- **API Integration**: Programmatic access for advanced use cases

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd The-Institute-AI-Final-Project

# Run setup script
python setup.py
```

### 2. Environment Configuration

Edit the `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Optional for reranking
```

### 3. Launch the System

**Option A: Interactive Menu**
```bash
python main.py
```

**Option B: Gradio Web Interface**
```bash
python -m rag_app.ui_gradio
```

**Option C: Full Pipeline**
```bash
python -m rag_app.pipeline
```

## 📁 Project Structure

```
The-Institute-AI-Final-Project/
├── rag_app/                    # Hybrid RAG System
│   ├── config.py              # Configuration settings
│   ├── loaders.py             # Document loaders (PDF/DOCX/TXT)
│   ├── chunking.py            # Structure-aware chunking
│   ├── metadata.py            # Metadata extraction (≥5 fields)
│   ├── indexing.py            # Dense + Sparse indexing
│   ├── retrieve.py            # Hybrid retrieval + reranking
│   ├── agents.py              # Multi-agent system
│   ├── prompts.py             # LLM prompt templates
│   ├── validate.py            # Document validation
│   ├── utils.py               # Utility functions
│   ├── ui_gradio.py           # Gradio interface
│   └── pipeline.py            # Main orchestrator
├── gear_images/               # Gear image analysis
├── vibration_data/           # Vibration analysis data
├── data/                     # Document storage
├── index/                    # Vector store indices
├── reports/                  # Generated reports
├── main.py                   # Enhanced main entry point
├── requirements.txt          # Python dependencies
└── setup.py                  # Setup script
```

## 🔧 Usage Examples

### Document Upload & Querying

```python
from rag_app.pipeline import create_pipeline

# Initialize pipeline
pipeline = create_pipeline()
pipeline.initialize()

# Ingest documents
documents = [Path("MG-5025A_Gearbox_Wear_Investigation_Report.pdf")]
result = pipeline.ingest_documents(documents)

# Query documents
response = pipeline.query("What caused the gearbox failure?")
print(response["response"])
```

### Integrated Analysis

```python
# Combine document analysis with live gear analysis
result = pipeline.integrated_query(
    "Compare documented wear patterns with current gear condition",
    include_gear_analysis=True
)
```

### Gradio Interface

1. Launch: `python -m rag_app.ui_gradio`
2. Upload documents via the web interface
3. Query using natural language
4. View responses with source citations

## 📊 System Specifications

### Retrieval Configuration
- **Dense Retrieval**: K=10 (semantic similarity)
- **Sparse Retrieval**: K=10 (keyword matching)
- **Reranking**: Top 20 → Final 6-8 contexts
- **Embedding Model**: text-embedding-3-small
- **LLM Model**: gpt-4o-mini

### Quality Thresholds
- **Context Precision**: ≥ 0.75
- **Recall**: ≥ 0.70
- **Faithfulness**: ≥ 0.85
- **Table QA Accuracy**: ≥ 0.90

### Document Requirements
- **Minimum Pages**: 10 pages
- **Supported Formats**: PDF, DOCX, DOC, TXT
- **Required Metadata**: Page numbers, section types, anchors
- **Content Types**: Tables, figures, technical text

## 🎯 Use Cases

### Failure Analysis
- **Root Cause Investigation**: "What caused the bearing failure in case MG-5025A?"
- **Pattern Recognition**: "Compare wear patterns across multiple cases"
- **Timeline Analysis**: "Show the sequence of events leading to failure"

### Technical Documentation
- **Measurement Extraction**: "What were the wear depth measurements?"
- **Specification Lookup**: "Find the gear specifications and tolerances"
- **Procedure Verification**: "What maintenance procedures were followed?"

### Comparative Analysis
- **Historical Comparison**: "How does this failure compare to previous cases?"
- **Cross-Reference**: "Correlate document findings with current analysis"
- **Trend Analysis**: "Identify patterns in failure modes over time"

## 🔬 Advanced Features

### Multi-Agent Routing
The system automatically routes queries to the most appropriate agent:
- **Summary queries** → Summary Agent
- **Specific questions** → Needle Agent  
- **Data questions** → Table Agent

### Metadata Filtering
Queries can be filtered by:
- Client ID / Case ID
- Date ranges
- Failure types
- Severity levels
- Document sections

### Integration Capabilities
- **Live Gear Analysis**: Real-time picture and vibration analysis
- **Cross-Platform**: Works with existing analysis tools
- **Export Options**: JSON, CSV, PDF reports

## 🛠️ Configuration

### Environment Variables
```env
# Core API Keys
OPENAI_API_KEY=required
GOOGLE_API_KEY=optional
COHERE_API_KEY=optional

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
RERANK_MODEL=BAAI/bge-reranker-large

# Retrieval Parameters
DENSE_K=10
SPARSE_K=10
CONTEXT_TOP_N=8
```

### Directory Configuration
```python
# In rag_app/config.py
DATA_DIR = Path("data")           # Input documents
INDEX_DIR = Path("index")         # Vector indices
REPORTS_DIR = Path("reports")     # Generated reports
```

## 📈 Performance & Evaluation

### Metrics Tracked
- **Answer Correctness**: Context-aware evaluation
- **Context Precision**: Relevance of retrieved contexts
- **Faithfulness**: Response adherence to sources
- **Retrieval Recall**: Coverage of relevant information

### Evaluation Tools
- **RAGAS Integration**: Automated evaluation metrics
- **Manual Validation**: Expert review capabilities
- **Performance Monitoring**: Response time and accuracy tracking

## 🔧 Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Check environment
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

**2. Document Validation Failures**
- Ensure documents have ≥10 pages
- Check file format support
- Verify document structure

**3. Index Loading Issues**
```bash
# Clear and rebuild index
rm -rf index/
python -m rag_app.pipeline
```

**4. Gear Analysis Integration**
- Ensure gear analysis modules are available
- Check file paths for images/vibration data
- Verify dependencies are installed

### Performance Optimization
- Use GPU for embedding models when available
- Adjust chunk sizes for your document types
- Tune retrieval parameters for your use case
- Monitor memory usage for large document sets

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black rag_app/
```

### Architecture Guidelines
- Follow the modular design pattern
- Maintain separation between RAG components
- Add comprehensive logging
- Include validation for all inputs

## 📝 License

This project is developed for educational and research purposes. Please ensure compliance with all API terms of service and data usage policies.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in the console output
3. Validate your environment setup
4. Ensure all dependencies are installed

## 🎉 Acknowledgments

Built using:
- **LangChain**: RAG framework and document processing
- **OpenAI**: Embeddings and language models
- **FAISS**: Dense vector search
- **Gradio**: Interactive web interface
- **Sentence Transformers**: Reranking models

---

**Ready to analyze failure reports with cutting-edge AI? Let's have a blast! 🚀**
