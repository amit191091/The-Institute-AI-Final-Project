# RAG Document Analysis System Integration

## 📋 Overview

The RAG (Retrieval-Augmented Generation) system has been successfully integrated into your Gear Wear Diagnosis project. This system allows you to ask questions about your "Gear wear Failure.pdf" document and get AI-powered answers.

## 🗂️ File Structure

```
Final_project/
├── RAG/                    # RAG system folder
│   ├── Main_RAG.py        # Main RAG application
│   └── app/               # RAG modules
│       ├── config.py      # Configuration settings
│       ├── loaders.py     # Document loading & image extraction
│       ├── chunking.py    # Text chunking
│       ├── indexing.py    # Vector database indexing
│       ├── retrieve.py    # Information retrieval
│       ├── agents.py      # AI agents
│       ├── ui_gradio.py   # Web interface
│       └── ...
├── run_rag.py             # RAG runner script
├── requirements.txt       # Unified dependencies
└── Main.py               # Main gear wear system (with RAG option)
```

## 🚀 How to Use

### Option 1: From Main Menu (Recommended)
1. Run your main system: `python Main.py`
2. Select option `6. RAG Document Analysis`
3. The system will automatically start the RAG application

### Option 2: Direct RAG Execution
1. Run the RAG system directly: `python run_rag.py`
2. Or navigate to RAG folder: `cd RAG && python Main_RAG.py`

## 📦 Dependencies

All RAG dependencies have been added to the main `requirements.txt` file:

### Core RAG Dependencies:
- `langchain` - Core RAG framework
- `langchain-community` - Community components
- `langchain-openai` - OpenAI integration
- `langchain-google-genai` - Google AI integration

### Vector Databases:
- `chromadb` - Vector database
- `faiss-cpu` - Fast similarity search
- `rank-bm25` - Sparse retrieval

### Document Processing:
- `unstructured[all-docs]` - Document parsing
- `pypdf` - PDF processing
- `PyMuPDF` - Advanced PDF features

### Web Interface:
- `gradio` - Web UI framework

## 🔧 Installation

Install all dependencies (including RAG):
```bash
pip install -r requirements.txt
```

## 📄 Supported Documents

The RAG system can process:
- **PDF files** (primary: "Gear wear Failure.pdf")
- **DOCX files** 
- **Text files**

## 🎯 Features

### Document Analysis:
- **Text Extraction** - Extracts text from PDF/DOCX
- **Image Extraction** - Extracts figures and diagrams
- **Table Detection** - Identifies and processes tables
- **Metadata Extraction** - Captures document structure

### AI-Powered Q&A:
- **Semantic Search** - Find relevant content
- **Hybrid Retrieval** - Combines dense and sparse search
- **Context-Aware Answers** - Provides detailed responses
- **Source Citations** - Shows where answers come from

### Web Interface:
- **Gradio UI** - User-friendly web interface
- **Real-time Q&A** - Ask questions and get instant answers
- **Document Upload** - Support for multiple documents

## 🔑 API Keys Required

The RAG system supports multiple AI providers:

### OpenAI (Recommended):
```bash
export OPENAI_API_KEY="your-openai-key"
```

### Google AI (Alternative):
```bash
export GOOGLE_API_KEY="your-google-key"
```

## 🎮 Usage Examples

### Starting the System:
```bash
# From main menu
python Main.py
# Select option 6

# Or directly
python run_rag.py
```

### Sample Questions:
- "What are the main causes of gear wear?"
- "How is vibration analysis used in gear diagnosis?"
- "What are the different wear patterns described?"
- "Show me the measurement procedures"

## 🔍 System Architecture

1. **Document Loading** - Processes PDF/DOCX files
2. **Text Chunking** - Breaks documents into searchable chunks
3. **Vector Indexing** - Creates searchable embeddings
4. **Query Processing** - Analyzes user questions
5. **Information Retrieval** - Finds relevant content
6. **Answer Generation** - Generates AI responses
7. **Web Interface** - Provides user interaction

## 🛠️ Troubleshooting

### Common Issues:

**Missing Dependencies:**
```bash
pip install -r requirements.txt
```

**API Key Issues:**
- Set your API keys as environment variables
- Check API key validity and quotas

**Document Not Found:**
- Ensure "Gear wear Failure.pdf" is in the project root
- Check file permissions

**Memory Issues:**
- Large documents may require more RAM
- Consider processing smaller sections

## 📈 Performance Tips

1. **First Run**: Initial indexing may take time
2. **Subsequent Runs**: Much faster due to cached indexes
3. **Large Documents**: Consider chunking for better performance
4. **API Limits**: Monitor your AI provider usage

## 🔄 Integration Benefits

- **Unified System**: Both gear wear analysis and document Q&A in one place
- **Shared Dependencies**: No duplicate package installations
- **Consistent Interface**: Same menu-driven approach
- **Modular Design**: Easy to maintain and extend

## 📞 Support

For issues with:
- **Gear Wear Analysis**: Check existing documentation
- **RAG System**: Review this README and error messages
- **Integration**: Ensure all files are in correct locations

---

**Note**: The RAG system is designed to work alongside your existing gear wear analysis, providing complementary AI-powered document analysis capabilities.
