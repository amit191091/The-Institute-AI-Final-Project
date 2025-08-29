# Pictures and Vibrations Database

## Overview
The `Pictures and Vibrations database` folder contains comprehensive gear wear analysis tools for both visual (image) and vibration signal analysis. It provides specialized analysis capabilities for gear wear detection, measurement, and monitoring.

## Purpose
- **Visual Analysis**: Analyze gear images for wear patterns and measurements
- **Vibration Analysis**: Process vibration signals for wear detection
- **Data Processing**: Handle large datasets of gear wear measurements
- **Result Visualization**: Generate plots and reports for analysis results
- **Integrated Analysis**: Combine visual and vibration data for comprehensive assessment

## Contents

### Picture Analysis (`Picture/`)

#### Core Components

##### `picture_analysis_menu.py` (8.9KB, 227 lines)
- **Purpose**: User interface for picture analysis operations
- **Features**:
  - Interactive menu system
  - Database processing options
  - Results display
  - Service integration
  - Error handling

##### `plot_results.py` (8.1KB, 199 lines)
- **Purpose**: Results visualization and plotting
- **Features**:
  - Data visualization
  - Chart generation
  - Report creation
  - Export functionality
  - Custom plotting options

#### Picture Tools (`Picture Tools/`)
- **Image Processing**: Advanced image analysis algorithms
- **Wear Detection**: Automatic wear pattern recognition
- **Measurement Tools**: Precise wear depth and area measurement
- **Alignment Tools**: Image alignment and normalization
- **Quality Control**: Image quality assessment and validation

#### Database (`database/`)
- **Reference Images**: Healthy and worn gear images
- **Measurement Data**: Wear depth measurements and analysis results
- **Processed Images**: Aligned and processed gear images
- **Analysis Results**: Comprehensive analysis outputs

#### Results Files
- **`all_teeth_results.csv`** (15KB, 1227 lines): Complete analysis results for all teeth
- **`single_tooth_results.csv`** (365B, 37 lines): Single tooth analysis results
- **`all_teeth_analysis_plot.png`** (320KB): Visualization of all teeth analysis
- **`single_tooth_analysis_plot.png`** (200KB): Single tooth analysis visualization

### Vibration Analysis (`Vibration/`)

#### Core Components

##### `vibration_service.py` (12KB, 323 lines)
- **Purpose**: Service layer for vibration analysis operations
- **Features**:
  - Signal processing
  - Feature extraction
  - Data analysis
  - Result generation
  - Error handling

##### `vibration_analysis_menu.py` (10KB, 252 lines)
- **Purpose**: User interface for vibration analysis
- **Features**:
  - Interactive menu system
  - Analysis options
  - Result display
  - File management
  - Service integration

##### `vibration_cli.py` (11KB, 251 lines)
- **Purpose**: Command-line interface for vibration analysis
- **Features**:
  - CLI commands
  - Batch processing
  - Result export
  - Configuration options
  - Automation support

#### Database (`database/`)
- **Signal Data**: Raw vibration signal data
- **Processed Data**: Feature-extracted vibration data
- **Analysis Results**: Vibration analysis outputs
- **Configuration**: Analysis parameters and settings

#### Visualization Files
- **High Speed Analysis**: 45 RPS vibration analysis plots
- **Low Speed Analysis**: 15 RPS vibration analysis plots
- **RMS Features**: Root Mean Square feature plots
- **FME Features**: Frequency Modulation Error feature plots

### Documentation Files

#### `Database figures and tables.docx` (32MB)
- **Purpose**: Comprehensive database documentation
- **Content**: Figures, tables, and analysis results
- **Format**: Microsoft Word document

#### `Database figures and tables.pdf` (5.5MB)
- **Purpose**: Portable database documentation
- **Content**: Same as DOCX but in PDF format
- **Format**: Portable Document Format

## Analysis Capabilities

### 1. Visual Analysis
- **Gear Alignment**: Automatic gear image alignment
- **Wear Detection**: Pattern recognition for wear identification
- **Depth Measurement**: Precise wear depth quantification
- **Area Calculation**: Wear area measurement and analysis
- **Quality Assessment**: Image quality and measurement accuracy

### 2. Vibration Analysis
- **Signal Processing**: Raw vibration signal analysis
- **Feature Extraction**: Statistical and frequency features
- **Wear Correlation**: Vibration-wear relationship analysis
- **Trend Analysis**: Wear progression over time
- **Anomaly Detection**: Unusual vibration patterns

### 3. Integrated Analysis
- **Multi-Modal**: Combine visual and vibration data
- **Cross-Validation**: Validate results across methods
- **Comprehensive Assessment**: Complete wear evaluation
- **Predictive Analysis**: Wear progression prediction

## Usage

### Picture Analysis
```bash
# Run picture analysis menu
python "Pictures and Vibrations database/Picture/picture_analysis_menu.py"

# Run specific analysis
python "Pictures and Vibrations database/Picture/Picture Tools/picture_service.py"
```

### Vibration Analysis
```bash
# Run vibration analysis menu
python "Pictures and Vibrations database/Vibration/vibration_analysis_menu.py"

# Run CLI interface
python "Pictures and Vibrations database/Vibration/vibration_cli.py"
```

### Programmatic Usage
```python
# Picture analysis
from Picture_Tools.picture_service import PictureAnalysisService
service = PictureAnalysisService()
result = service.run_complete_analysis()

# Vibration analysis
from Vibration.vibration_service import VibrationAnalysisService
service = VibrationAnalysisService()
result = service.analyze_vibration_data()
```

## Data Structure

### Picture Database
```
database/
├── Healthy.png              # Reference healthy gear
├── aligned_Healthy.png      # Aligned healthy gear
├── Wear1.png - Wear35.png   # Worn gear images
└── Wear depth measurments/  # Measurement data
```

### Vibration Database
```
database/
├── Records.csv              # Raw vibration records
├── RMS15.csv               # 15 RPS RMS data
├── RMS45.csv               # 45 RPS RMS data
├── FME Values.csv          # FME feature data
└── Vibration signals/      # Signal data files
```

## Analysis Workflows

### 1. Visual Analysis Workflow
```
Input Image → Alignment → Wear Detection → Measurement → Results
```

### 2. Vibration Analysis Workflow
```
Raw Signal → Preprocessing → Feature Extraction → Analysis → Results
```

### 3. Integrated Workflow
```
Visual Data + Vibration Data → Cross-Analysis → Comprehensive Results
```

## Performance

### Optimization Features
- **Batch Processing**: Efficient processing of multiple images/signals
- **Memory Management**: Optimized memory usage for large datasets
- **Parallel Processing**: Concurrent analysis operations
- **Caching**: Result caching for repeated operations

### Quality Assurance
- **Validation**: Input data validation and quality checks
- **Calibration**: Measurement system calibration
- **Reproducibility**: Consistent and reproducible results
- **Documentation**: Comprehensive result documentation

## Integration

### MCP Integration
- **Tool Exposure**: Available through MCP tools
- **Remote Access**: External tool access capabilities
- **Standardized Interface**: Consistent API for all operations
- **Error Handling**: Robust error management

### RAG Integration
- **Document Processing**: Analysis results as documents
- **Knowledge Base**: Analysis methods and results
- **Query Support**: Answer questions about analysis
- **Result Storage**: Persistent result storage

## Configuration

### Environment Setup
- **Python Dependencies**: Required analysis libraries
- **Image Processing**: OpenCV and image analysis tools
- **Signal Processing**: Vibration analysis libraries
- **Visualization**: Plotting and charting tools

### Analysis Parameters
- **Measurement Precision**: Configurable measurement accuracy
- **Processing Options**: Analysis method selection
- **Output Formats**: Result format configuration
- **Quality Thresholds**: Quality control parameters

## Maintenance

### Data Management
- **Backup**: Regular data backup procedures
- **Versioning**: Analysis result versioning
- **Cleanup**: Automated cleanup of temporary files
- **Archiving**: Long-term result archiving

### Code Quality
- **Modularity**: Clean component separation
- **Documentation**: Comprehensive inline documentation
- **Testing**: Automated testing for analysis functions
- **Error Handling**: Robust error management

## Status: ✅ Production Ready
The Pictures and Vibrations database provides comprehensive gear wear analysis capabilities with advanced visualization, robust processing, and excellent integration with the overall system.
