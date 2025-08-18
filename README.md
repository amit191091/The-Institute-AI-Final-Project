# 🔧 Gear Wear Diagnosis System

A comprehensive gear wear diagnosis system that combines **Picture Analysis** and **Vibration Analysis** to provide accurate gear condition assessment and maintenance recommendations.

## 📋 Features

### 🖼️ Picture Analysis
- **Visual Wear Detection**: Analyzes gear images to detect wear patterns
- **Tooth-by-Tooth Analysis**: Individual tooth wear measurement
- **Wear Level Classification**: Categorizes wear from Healthy to Severe (W1-W35)
- **Automated Image Processing**: Handles orientation, scaling, and alignment
- **Statistical Analysis**: Provides detailed wear statistics and progression scores

### 📊 Vibration Analysis
- **Mechanical Response Analysis**: Analyzes vibration signals for fault detection
- **Bearing Fault Detection**: Identifies bearing-related issues
- **Gear Mesh Analysis**: Detects gear mesh frequency patterns
- **Signal Processing**: Advanced signal analysis with FFT and filtering

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
├── gear_images/                     # Picture analysis modules
│   ├── database/                    # Gear image database
│   ├── wear_measurement.py          # Core wear measurement logic
│   └── picture_analysis_menu.py     # Picture analysis menu
├── vibration_data/                  # Vibration analysis data
├── vibration_analysis.py            # Vibration analysis module
├── diagnosis_menu.py                # Diagnosis system menu
├── write_summary_menu.py            # Report generation menu
├── utility_functions.py             # Utility functions
└── README.md                        # This file
```

## 🎯 Main Menu Options

1. **Picture Analysis** - Visual wear detection and measurement
2. **Vibration Analysis** - Mechanical fault detection
3. **Diagnosis** - Comprehensive gear condition assessment
4. **Write Summary** - Generate detailed reports
5. **View System Information** - System status and file information
6. **Exit** - Close the application

## 📊 Analysis Types

### Picture Analysis Features
- **Healthy Baseline Creation**: Establishes reference for comparison
- **Wear Level Analysis**: Processes images from different wear levels
- **Individual Tooth Analysis**: Detailed tooth-by-tooth assessment
- **Statistical Reporting**: Comprehensive wear statistics
- **Visualization**: Generated plots and analysis images

### Vibration Analysis Features
- **Signal Processing**: Advanced signal analysis techniques
- **Frequency Domain Analysis**: FFT-based frequency analysis
- **Bearing Fault Detection**: Identifies bearing-related issues
- **Gear Mesh Analysis**: Detects gear mesh patterns
- **Statistical Metrics**: RMS, peak, frequency analysis

## 📈 Output Reports

The system generates several types of reports:

- **Comprehensive Diagnosis Report**: Complete analysis summary
- **Picture Analysis Report**: Visual wear analysis results
- **Vibration Analysis Report**: Mechanical analysis results
- **Technical Report**: Detailed technical analysis
- **JSON Results**: Structured data for further processing

## 🔧 Configuration

### Data Organization
- **Gear Images**: Place gear images in `gear_images/database/`
- **Vibration Data**: Place vibration files in `vibration_data/database/`
- **Results**: Analysis results are saved in the project root

### File Formats
- **Images**: JPG, PNG, BMP formats supported
- **Vibration Data**: MATLAB .mat files (v7.3 format)
- **Output**: CSV, JSON, and text report formats

## 📋 Requirements

### Core Dependencies
- `numpy` - Numerical computing
- `opencv-python` - Image processing
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `pandas` - Data manipulation
- `scikit-image` - Image processing
- `h5py` - HDF5 file handling

### Optional Dependencies
- `scikit-learn` - Machine learning (for advanced analysis)
- `seaborn` - Statistical visualization

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

---

**Note**: This system is designed for research and educational purposes. Always verify results with professional equipment and expertise for critical applications. 