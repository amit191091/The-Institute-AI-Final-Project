# Vibration to Noise Conversion System

## Overview

This system provides comprehensive tools to convert vibration signals into noise signals for gear wear analysis. The conversion is based on physical relationships between mechanical vibrations and acoustic noise generation in gear systems.

## 🎯 Key Features

- **Multiple Conversion Methods**: Transfer function, empirical relationship, and acoustic radiation models
- **Batch Processing**: Convert entire vibration databases to noise databases
- **Integrated Analysis**: Seamlessly integrated with existing noise analysis system
- **Characteristic Comparison**: Compare original vibration vs converted noise characteristics
- **Visualization**: Generate comparison plots and analysis reports

## 🔧 Conversion Methods

### 1. Transfer Function Method (Recommended)
- **Principle**: Uses frequency domain transfer function to model acoustic characteristics
- **Advantages**: 
  - Realistic frequency response modeling
  - Captures gear mesh frequency characteristics
  - Accounts for acoustic radiation efficiency
- **Best for**: Gear wear analysis and fault detection

### 2. Empirical Relationship Method
- **Principle**: Simple mathematical relationship with nonlinear effects
- **Advantages**:
  - Fast computation
  - Includes nonlinear gear effects
  - Good for quick approximations
- **Best for**: Rapid prototyping and initial analysis

### 3. Acoustic Radiation Model
- **Principle**: Based on acoustic radiation theory
- **Advantages**:
  - Most physically accurate
  - Considers surface area and distance
  - Frequency-dependent radiation efficiency
- **Best for**: Detailed acoustic analysis

## 📁 File Structure

```
project_root/
├── noise_data/
│   ├── __init__.py                    # Package initialization
│   ├── vibration_to_noise_converter.py # Main converter class
│   ├── convert_vibration_to_noise.py   # Standalone conversion script
│   ├── noise_analysis.py              # Enhanced noise analysis with conversion
│   └── database/                      # Original noise data
├── noise_analysis_menu.py             # Menu system with conversion options
├── vibration_data/
│   └── database/
│       ├── Healthy/
│       ├── W1/
│       ├── W2/
│       └── ...
└── noise_data/
    └── converted_from_vibration/      # Converted noise data
```

## 🚀 Quick Start

### Option 1: Standalone Script
```bash
python convert_vibration_to_noise.py
```

### Option 2: Integrated Menu System
```bash
python Main.py
# Navigate to: Noise Analysis → Convert Vibration to Noise
```

### Option 3: Direct API Usage
```python
from noise_data.vibration_to_noise_converter import VibrationToNoiseConverter

# Initialize converter
converter = VibrationToNoiseConverter(sampling_rate=50000)

# Convert single signal
noise_signal = converter.convert_vibration_to_noise(
    vibration_signal, 
    method='transfer_function'
)

# Batch convert database
stats = converter.batch_convert_vibration_database(
    noise_data_path="vibration_data/database",
    output_path="noise_data/converted_from_vibration",
    method='transfer_function'
)
```

## 📊 Usage Examples

### Example 1: Single Signal Conversion with Gear Specifications
```python
import numpy as np
import scipy.io as sio
from noise_data.vibration_to_noise_converter import VibrationToNoiseConverter

# Load vibration data
mat_data = sio.loadmat('vibration_data/sample.mat')
vibration_signal = mat_data['signal'].flatten()

# Initialize converter
converter = VibrationToNoiseConverter(sampling_rate=50000)

# Define gear specifications
gear_specs = {
    'driven_teeth': 35,      # Checked gear (driven)
    'driving_teeth': 18,     # Driving gear
    'driven_speed': 15,      # Hz (R15)
    'driving_speed': 45      # Hz (R45)
}

# Calculate gear mesh frequency
mesh_freq, harmonics = converter.calculate_gear_mesh_frequency(gear_specs)
print(f"Gear mesh frequency: {mesh_freq:.1f} Hz")
print(f"Harmonics: {[f'{h:.1f}' for h in harmonics]} Hz")

# Convert to noise with gear-specific parameters
noise_signal = converter.convert_vibration_to_noise(
    vibration_signal, 
    method='transfer_function',
    gear_specs=gear_specs
)

# Compare characteristics
comparison = converter.compare_vibration_noise_characteristics(
    vibration_signal, noise_signal
)

print(f"Vibration RMS: {comparison['vibration']['rms']:.4f}")
print(f"Noise RMS: {comparison['noise']['rms']:.4f}")
```

### Example 2: Batch Database Conversion
```python
from noise_data.vibration_to_noise_converter import VibrationToNoiseConverter

# Initialize converter
converter = VibrationToNoiseConverter(sampling_rate=50000)

# Convert entire database
stats = converter.batch_convert_vibration_database(
    noise_data_path="vibration_data/database",
    output_path="noise_data/converted_from_vibration",
    method='transfer_function'
)

print(f"Converted {stats['converted_files']} out of {stats['total_files']} files")
```

### Example 3: Integrated Analysis with Automatic Gear Spec Extraction
```python
from noise_data.noise_analysis import NoiseAnalysis
from noise_data.vibration_to_noise_converter import VibrationToNoiseConverter

# Initialize enhanced noise analyzer
analyzer = NoiseAnalysis(sampling_rate=50000)

# The system automatically extracts gear specifications from folder names
# For example, folder "R45_L10" will be interpreted as:
# - Speed: 45 Hz
# - Load: 10 Nm
# - Gear mesh frequency: 18 × 45 = 810 Hz

# Convert and analyze
stats = analyzer.convert_vibration_to_noise_database(
    noise_data_path="vibration_data/database",
    output_path="noise_data/converted_from_vibration",
    method='transfer_function'
)

# Analyze converted data
results = analyzer.analyze_converted_noise_data(
    "noise_data/converted_from_vibration"
)

# Test gear specification extraction
converter = VibrationToNoiseConverter(sampling_rate=50000)
test_folders = ['R45_L10', 'R30_L15', 'R60_L5']

for folder in test_folders:
    gear_specs = converter._extract_gear_specs_from_folder(folder)
    mesh_freq, harmonics = converter.calculate_gear_mesh_frequency(gear_specs)
    print(f"Folder {folder}: Mesh frequency = {mesh_freq:.1f} Hz")
```

## 🔍 Analysis Features

### Signal Characteristics
- **RMS Value**: Root mean square amplitude
- **Peak Value**: Maximum absolute amplitude
- **Crest Factor**: Peak-to-RMS ratio
- **Kurtosis**: Statistical measure of signal shape
- **Skewness**: Statistical measure of signal asymmetry

### Frequency Analysis
- **Dominant Frequencies**: Peak frequencies in spectrum
- **Spectral Centroid**: Center of mass of frequency spectrum
- **Frequency Energy**: Total energy in frequency domain

### Wear Pattern Classification
- **Impact-like**: High crest factor (>4) - potential gear damage
- **Moderate Variation**: Medium crest factor (2.5-4) - wear progression
- **Normal Operation**: Low crest factor (<2.5) - healthy condition

## 📈 Visualization

### Comparison Plots
```python
# Generate comparison plot
fig = converter.plot_conversion_comparison(
    vibration_signal, 
    noise_signal,
    title="Vibration to Noise Conversion"
)

# Save plot
fig.savefig('conversion_comparison.png', dpi=300, bbox_inches='tight')
```

### Analysis Reports
```python
# Generate noise analysis report
analyzer = NoiseAnalysis()
report = analyzer.generate_noise_report("noise_analysis_report.txt")
```

## ⚙️ Configuration

### Sampling Rate and Record Time
```python
# Default: 50 kHz (50,000 Hz) sampling rate
# Record time: 60 seconds
converter = VibrationToNoiseConverter(sampling_rate=50000)

# Signal length calculation:
# signal_length = sampling_rate * record_time
# signal_length = 50000 * 60 = 3,000,000 samples
```

### Gear Specifications
```python
# Define gear specifications for accurate mesh frequency calculation
gear_specs = {
    'driven_teeth': 35,      # Checked gear (driven)
    'driving_teeth': 18,     # Driving gear
    'driven_speed': 15,      # Hz (R15)
    'driving_speed': 45      # Hz (R45)
}

# Calculate gear mesh frequency
mesh_freq, harmonics = converter.calculate_gear_mesh_frequency(gear_specs)
print(f"Mesh frequency: {mesh_freq:.1f} Hz")
print(f"Harmonics: {[f'{h:.1f}' for h in harmonics]} Hz")
```

### Transfer Function Parameters
```python
# Custom transfer function parameters with gear specifications
noise_signal = converter.convert_vibration_to_noise(
    vibration_signal,
    method='transfer_function',
    gear_specs=gear_specs,   # Use specific gear specifications
    low_freq_cutoff=50,      # Hz
    high_freq_cutoff=2000    # Hz
)
```

### Empirical Method Parameters
```python
# Custom empirical parameters
noise_signal = converter.convert_vibration_to_noise(
    vibration_signal,
    method='empirical',
    sensitivity_factor=0.15,
    nonlinearity=0.1
)
```

### Acoustic Method Parameters
```python
# Custom acoustic parameters
noise_signal = converter.convert_vibration_to_noise(
    vibration_signal,
    method='acoustic',
    surface_area=0.01,  # m²
    distance=1.0        # m
)
```

## 🔬 Technical Details

### Physical Relationship
The conversion is based on the fundamental relationship between vibration and noise:

1. **Vibration** → Mechanical oscillations of gear components
2. **Acoustic Radiation** → Conversion of mechanical energy to acoustic energy
3. **Noise** → Acoustic pressure waves in air

### Gear Specifications
The system is configured for the specific gear system:
- **Driven Gear (Checked)**: 35 teeth
- **Driving Gear**: 18 teeth
- **Gear Ratio**: 18/35 = 0.514
- **Speed Range**: 15-60 Hz (extracted from folder names like R45_L10)
- **Load Range**: 5-15 Nm (extracted from folder names like R45_L10)

### Gear Mesh Frequency Calculation
```
Gear Mesh Frequency = Driving Teeth × Driving Speed
```

**Example for R45_L10:**
- Driving speed: 45 Hz
- Gear mesh frequency: 18 × 45 = 810 Hz
- Harmonics: 1620 Hz, 2430 Hz, 3240 Hz

### Transfer Function Model
```
H(f) = Radiation_Efficiency(f) × Acoustic_Filter(f) × Distance_Attenuation(f)
```

Where:
- `Radiation_Efficiency(f)`: Frequency-dependent efficiency of acoustic radiation
- `Acoustic_Filter(f)`: Frequency response of the acoustic path
- `Distance_Attenuation(f)`: Distance-dependent attenuation

### Frequency Characteristics
- **Low Frequency (<50 Hz)**: Reduced acoustic radiation
- **Gear Mesh Frequency**: Enhanced based on calculated mesh frequency (e.g., 810 Hz for R45)
- **Harmonics**: Enhanced with decreasing emphasis (1620, 2430, 3240 Hz)
- **High Frequency (>2000 Hz)**: Roll-off due to acoustic filtering
- **Nyquist Frequency**: 25 kHz (half of 50 kHz sampling rate)
- **Maximum Detectable Frequency**: 25 kHz

## 📋 Requirements

### Python Dependencies
```
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
pathlib (built-in)
```

### Installation
```bash
pip install numpy scipy matplotlib
```

## 🎯 Best Practices

### 1. Method Selection
- **Gear Wear Analysis**: Use Transfer Function method
- **Quick Prototyping**: Use Empirical method
- **Detailed Acoustic Analysis**: Use Acoustic Radiation method

### 2. Parameter Tuning
- Adjust transfer function parameters based on your gear system
- Calibrate sensitivity factors with measured data
- Consider environmental conditions for acoustic parameters

### 3. Validation
- Compare with actual noise measurements when available
- Validate frequency characteristics against known gear mesh frequencies
- Check conversion ratios for physical consistency

### 4. Performance
- Use batch processing for large databases
- Consider memory usage for large signals
- Optimize sampling rate based on frequency content

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
2. **File Not Found**: Check file paths and directory structure
3. **Memory Error**: Reduce batch size or signal length
4. **Conversion Failure**: Verify input signal format and parameters

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Run conversion with debug info
converter = VibrationToNoiseConverter(sampling_rate=1000)
stats = converter.batch_convert_vibration_database(...)
```

## 📚 References

1. **Gear Noise Analysis**: Fundamentals of gear noise generation and analysis
2. **Acoustic Radiation**: Theory of acoustic radiation from vibrating surfaces
3. **Signal Processing**: Digital signal processing techniques for vibration analysis
4. **Transfer Functions**: Frequency domain modeling of acoustic systems

## 🤝 Contributing

To contribute to this system:

1. Follow the existing code structure
2. Add comprehensive documentation
3. Include unit tests for new features
4. Validate with real gear data when possible

## 📄 License

This system is part of the Gear Wear Diagnosis Project and follows the same licensing terms.

---

**Note**: This conversion system provides a mathematical approximation of the vibration-to-noise relationship. For critical applications, validate results against actual acoustic measurements.
