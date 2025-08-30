# 📊 **PICTURES AND VIBRATIONS DATABASE**

## 📋 **OVERVIEW**

This folder contains comprehensive tools for gear wear analysis using both visual (pictures) and vibration data. It provides specialized analysis capabilities for gear wear detection, measurement, and monitoring.

---

## 🗂️ **FOLDER STRUCTURE**

```
Pictures and Vibrations database/
├── Picture/                    # Image analysis tools
├── Vibration/                  # Vibration analysis tools
├── Database figures and tables.pdf  # Reference documentation
└── README.md                   # Detailed documentation
```

---

## 🖼️ **PICTURE ANALYSIS**

### **Purpose**
Advanced image processing and gear wear analysis from photographs

### **Location**
`Pictures and Vibrations database/Picture/`

### **Key Components**

#### **📁 Picture Tools/**
- **`Analyze_tooth1.py`**: Single tooth analysis and measurement
- **`Analyze_all_teeth.py`**: Complete gear analysis for all teeth
- **`picture_service.py`**: Core image processing service
- **`data_loader.py`**: Image data loading utilities
- **`config.py`**: Image analysis configuration
- **`visualization.py`**: Chart and graph generation
- **`picture_cli.py`**: Command-line interface

#### **Main Files**
- **`picture_analysis_menu.py`**: User interface for image analysis
- **`plot_results.py`**: Results visualization and reporting

### **Features**
- ✅ **Wear Detection**: Automated gear wear measurement
- ✅ **Image Processing**: Advanced image analysis algorithms
- ✅ **Data Export**: CSV and visualization outputs
- ✅ **Batch Processing**: Multiple image analysis
- ✅ **Measurement Tools**: Precise gear wear quantification
- ✅ **Visualization**: Charts, graphs, and trend analysis

### **Usage**
```bash
# Navigate to Picture folder
cd "Pictures and Vibrations database/Picture"

# Run image analysis menu
python picture_analysis_menu.py

# Run single tooth analysis
python "Picture Tools/Analyze_tooth1.py"

# Run complete gear analysis
python "Picture Tools/Analyze_all_teeth.py"
```

---

## 📈 **VIBRATION ANALYSIS**

### **Purpose**
Vibration signal analysis and frequency domain processing for gear wear detection

### **Location**
`Pictures and Vibrations database/Vibration/`

### **Key Components**

#### **Core Files**
- **`vibration_service.py`**: Core vibration analysis service
- **`vibration_analysis_menu.py`**: User interface for vibration analysis
- **`vibration_cli.py`**: Command-line interface

#### **📁 Database/**
- **CSV files**: Vibration measurements and data
- **Excel files**: Analysis results and reports
- **Compressed archives**: Data storage and backup

### **Features**
- ✅ **Signal Processing**: FFT, RMS, and spectral analysis
- ✅ **Frequency Analysis**: Sideband detection and meshing frequency
- ✅ **Data Visualization**: Time and frequency domain plots
- ✅ **Trend Analysis**: Wear progression monitoring
- ✅ **Real-time Processing**: Live vibration data analysis
- ✅ **Statistical Analysis**: Advanced statistical methods

### **Usage**
```bash
# Navigate to Vibration folder
cd "Pictures and Vibrations database/Vibration"

# Run vibration analysis menu
python vibration_analysis_menu.py

# Run CLI interface
python vibration_cli.py
```

---

## 🔄 **INTEGRATION**

### **Data Flow**
1. **Input**: Images and vibration data from gear systems
2. **Processing**: Specialized analysis for each data type
3. **Analysis**: Wear detection and measurement
4. **Output**: Reports, visualizations, and data exports
5. **Integration**: Results can be used with RAG system

### **Cross-Component Features**
- **Unified Data Format**: Consistent data structures
- **Shared Configuration**: Common settings and parameters
- **Integrated Reporting**: Combined analysis reports
- **Data Export**: Standardized output formats

---

## 📊 **ANALYSIS CAPABILITIES**

### **Picture Analysis**
- **Gear Wear Measurement**: Precise wear depth quantification
- **Image Enhancement**: Advanced image processing algorithms
- **Batch Processing**: Multiple image analysis
- **Quality Assessment**: Image quality evaluation
- **Automated Detection**: AI-powered wear detection

### **Vibration Analysis**
- **Frequency Domain**: FFT and spectral analysis
- **Time Domain**: RMS and peak analysis
- **Sideband Detection**: Gear meshing frequency analysis
- **Trend Monitoring**: Wear progression tracking
- **Alert Systems**: Threshold-based alerts

---

## 🎯 **USE CASES**

### **Industrial Applications**
- **Gearbox Monitoring**: Continuous gear wear monitoring
- **Predictive Maintenance**: Early wear detection
- **Quality Control**: Manufacturing quality assurance
- **Research & Development**: Gear design optimization

### **Analysis Scenarios**
- **Single Gear Analysis**: Individual gear examination
- **Batch Processing**: Multiple gear analysis
- **Trend Analysis**: Long-term wear monitoring
- **Comparative Analysis**: Different gear comparisons

---

## 📈 **OUTPUTS**

### **Data Files**
- **CSV Reports**: Measured data and statistics
- **Excel Reports**: Comprehensive analysis results
- **Image Files**: Processed images and visualizations
- **JSON Data**: Structured analysis data

### **Visualizations**
- **Wear Trend Charts**: Wear progression over time
- **Frequency Spectra**: Vibration frequency analysis
- **Comparison Plots**: Before/after analysis
- **Statistical Graphs**: Data distribution and trends

---

## 🔧 **CONFIGURATION**

### **Picture Analysis Settings**
- **Image Quality**: Resolution and format settings
- **Processing Parameters**: Analysis algorithm settings
- **Output Formats**: Export file formats
- **Batch Settings**: Multiple file processing

### **Vibration Analysis Settings**
- **Sampling Rate**: Data acquisition settings
- **Filter Parameters**: Signal processing filters
- **Threshold Values**: Alert and detection thresholds
- **Analysis Windows**: Time and frequency windows

---

## 📚 **DOCUMENTATION**

- **`README.md`**: Detailed documentation for the database
- **`Database figures and tables.pdf`**: Reference documentation
- **Inline Comments**: Code documentation
- **User Guides**: Step-by-step usage instructions

---

## 🚀 **GETTING STARTED**

1. **Setup**: Ensure all dependencies are installed
2. **Data Preparation**: Organize images and vibration data
3. **Configuration**: Set analysis parameters
4. **Analysis**: Run picture or vibration analysis
5. **Review Results**: Examine outputs and visualizations
6. **Export Data**: Save results for further processing

---

**🏆 This database provides comprehensive gear wear analysis capabilities for both visual and vibration data!**
