# 🔧 Gear Wear Diagnosis System

A comprehensive gear wear diagnosis system that combines **Picture Analysis**, **Vibration Analysis**, and **RAG Document Analysis** to provide accurate gear condition assessment and maintenance recommendations.

## 📋 Features

### 🖼️ Picture Analysis
- **Visual Wear Detection**: Analyzes gear images to detect wear patterns
- **Tooth-by-Tooth Analysis**: Individual tooth wear measurement
- **Wear Level Classification**: Categorizes wear from Healthy to Severe (W1-W35)
- **Automated Image Processing**: Handles orientation, scaling, and alignment
- **Statistical Analysis**: Provides detailed wear statistics and progression scores
- **MCP Wear Depth Analysis**: Advanced wear depth measurement with calibration
- **Quality Assurance**: Built-in verification and quality assessment

### 📊 Vibration Analysis
- **Plot Viewing System**: Opens pre-generated vibration analysis plots
- **High Speed Analysis**: 45 RPS vibration signal plots (4 plots)
- **Low Speed Analysis**: 15 RPS vibration signal plots (4 plots)
- **RMS Features**: Root Mean Square analysis plots
- **FME Features**: Frequency Modulated Energy analysis plots
- **Cross-Platform Support**: Automatic image opening (Windows, macOS, Linux)

### 🤖 RAG Document Analysis
- **AI-Powered Analysis**: Uses AI to analyze gear wear failure documents
- **Question-Answering**: Ask questions about document content
- **Intelligent Retrieval**: Retrieves relevant information from documents
- **Web Interface**: User-friendly web-based interface

### 🔍 Diagnosis System
- **Comprehensive Assessment**: Combines multiple analysis types
- **Condition Scoring**: Numerical assessment of gear health
- **Maintenance Recommendations**: Actionable maintenance advice
- **Report Generation**: Detailed technical and summary reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/gear-wear-diagnosis-system.git
cd gear-wear-diagnosis-system

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the main application
python Main.py
```

## 📁 Project Structure

```
Final_project/
├── Main.py                          # Main application entry point
├── utility_functions.py             # Utility functions
├── write_summary_menu.py            # Report generation menu
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── Pictures and Vibrations database/
│   ├── Picture/                     # Picture analysis modules
│   │   ├── database/                # Gear image database
│   │   ├── Picture Tools/           # Picture analysis tools
│   │   │   ├── gear_parameters.py   # Gear specifications
│   │   │   ├── config.py            # Configuration management
│   │   │   ├── data_loader.py       # Data loading utilities
│   │   │   ├── visualization.py     # Plotting functions
│   │   │   ├── wear_analyzer.py     # Wear analysis engine
│   │   │   ├── image_processor.py   # Image processing
│   │   │   ├── tooth1_analyzer.py   # Tooth1 analysis
│   │   │   └── [other analysis tools]
│   │   ├── picture_analysis_menu.py # Picture analysis menu
│   │   ├── plot_results.py          # Results plotting
│   │   ├── calibration_factors.py   # Calibration utilities
│   │   ├── hybrid_scale_bar_analyzer.py
│   │   ├── image_orientation_normalizer.py
│   │   ├── image_quality.py
│   │   ├── reporting.py
│   │   ├── targeted_wear_analyzer.py
│   │   ├── tooth_alignment.py
│   │   ├── tooth_curve_analyzer.py
│   │   ├── tooth_matching.py
│   │   ├── wear_analysis.py
│   │   ├── wear_measurement_system.py
│   │   ├── wear_depth_qa_verification.py
│   │   ├── wear_depth_quality_assessment.py
│   │   └── mcp_assistant_scripts/    # MCP integration
│   │       ├── __init__.py
│   │       ├── agent_wear_depth_interface.py
│   │       ├── mcp_wear_interface.py
│   │       └── mcp_integration_example.py
│   └── Vibration/                   # Vibration analysis
│       ├── vibration_analysis_menu.py # Vibration menu (plot viewing)
│       ├── database/                # Vibration data files
│       └── [vibration plots and data]
├── RAG/                             # RAG document analysis system
│   ├── Main_RAG.py                  # RAG main application
│   ├── app/                         # RAG application modules
│   │   ├── agents.py
│   │   ├── chunking.py
│   │   ├── config.py
│   │   ├── indexing.py
│   │   ├── loaders.py
│   │   ├── metadata.py
│   │   ├── retrieve.py
│   │   ├── ui_gradio.py
│   │   └── validate.py
│   └── [RAG data and logs]
├── Gear wear Failure.pdf            # Sample document for RAG analysis
├── Gear wear Failure.docx           # Sample document for RAG analysis
├── RAG_README.md                    # RAG system documentation
└── README.md                        # This file
```

## 🎯 Main Menu Options

1. **Extract Database** - Access picture and vibration analysis
   - Picture Analysis: Visual wear detection and measurement
   - Vibration Analysis: Plot viewing for vibration signals
2. **RAG Document Analysis** - AI-powered document analysis
3. **Write Summary** - Generate detailed reports
4. **View System Information** - System status and file information
5. **Exit** - Close the application

## 📊 Analysis Types

### Picture Analysis Features
- **Healthy Baseline Creation**: Establishes reference for comparison
- **Wear Level Analysis**: Processes images from different wear levels
- **Individual Tooth Analysis**: Detailed tooth-by-tooth assessment
- **Statistical Reporting**: Comprehensive wear statistics
- **Visualization**: Generated plots and analysis images
- **MCP Wear Depth Analysis**: Advanced wear depth measurement
- **Quality Assurance**: Built-in verification systems

### Vibration Analysis Features
- **Plot Viewing System**: Opens pre-generated vibration analysis plots
- **High Speed Analysis**: 45 RPS vibration signal plots (4 plots)
- **Low Speed Analysis**: 15 RPS vibration signal plots (4 plots)
- **RMS Features**: Root Mean Square analysis plots
- **FME Features**: Frequency Modulated Energy analysis plots

### RAG Document Analysis Features
- **Document Processing**: Load and process gear wear documents
- **Intelligent Search**: AI-powered information retrieval
- **Question Answering**: Ask questions about document content
- **Web Interface**: User-friendly Gradio interface

## 📈 Output Reports

The system generates several types of reports:

- **Comprehensive Diagnosis Report**: Complete analysis summary
- **Picture Analysis Report**: Visual wear analysis results
- **Vibration Analysis Report**: Plot viewing summary
- **Technical Report**: Detailed technical analysis
- **JSON Results**: Structured data for further processing
- **PNG Plots**: High-quality analysis visualizations

## 🔧 Configuration

### Data Organization
- **Gear Images**: Place gear images in `Pictures and Vibrations database/Picture/database/`
- **Vibration Data**: Place vibration files in `Pictures and Vibrations database/Vibration/database/`
- **RAG Documents**: Place documents in `RAG/data/`
- **Results**: Analysis results are saved in the project root

### File Formats
- **Images**: JPG, PNG, BMP formats supported
- **Vibration Data**: CSV files for processed data
- **Documents**: PDF, DOCX, TXT formats for RAG analysis
- **Output**: CSV, JSON, PNG, and text report formats

## 📋 Requirements

### Core Dependencies
- `numpy` - Numerical computing
- `opencv-python` - Image processing
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `pandas` - Data manipulation
- `scikit-image` - Image processing
- `pillow` - Image handling

### RAG Dependencies
- `chromadb` - Vector database
- `sentence-transformers` - Text embeddings
- `gradio` - Web interface
- `langchain` - AI framework

### Optional Dependencies
- `scikit-learn` - Machine learning (for advanced analysis)
- `seaborn` - Statistical visualization

## 🚀 Advanced Features

### MCP Wear Depth Analysis
- **Calibration Factors**: Automatic scale bar detection and calibration
- **Hybrid Analysis**: Combines multiple measurement techniques
- **Quality Assessment**: Built-in quality verification
- **Comprehensive Reporting**: Detailed wear depth analysis

### Modular Architecture
- **Picture Tools**: Organized analysis modules
- **Configuration Management**: Centralized configuration system
- **Data Loading**: Flexible data loading utilities
- **Visualization**: Advanced plotting capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Image Processing**: OpenCV and scikit-image libraries
- **Signal Processing**: SciPy and NumPy for vibration analysis
- **Data Visualization**: Matplotlib for plotting and analysis
- **Scientific Computing**: Python scientific computing ecosystem
- **AI Framework**: LangChain and ChromaDB for RAG system

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the code comments
- Review the example outputs in the project

## 🔄 Version History

- **v1.0** - Initial release with picture and vibration analysis
- **v1.1** - Enhanced diagnosis system and reporting
- **v1.2** - Improved image processing and wear detection
- **v1.3** - Streamlined system (removed noise analysis)
- **v1.4** - Added RAG document analysis system
- **v1.5** - Modular architecture and MCP integration
- **v1.6** - Vibration analysis converted to plot viewing mode
- **v1.7** - Complete system integration and optimization

## 🎉 System Status

**✅ ALL MODULES SUCCESSFULLY INTEGRATED**

The system is fully functional with:
- ✅ Complete picture analysis with MCP integration
- ✅ Vibration analysis plot viewing system
- ✅ RAG document analysis with AI capabilities
- ✅ Comprehensive reporting and visualization
- ✅ Modular architecture for easy maintenance

---

**Note**: This system is designed for research and educational purposes. Always verify results with professional equipment and expertise for critical applications. 