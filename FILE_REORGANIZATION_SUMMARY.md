# File Reorganization Summary

## Overview
Successfully moved vibration-to-noise conversion files to the `noise_data` folder and updated all import paths.

## Files Moved
- `vibration_to_noise_converter.py` → `noise_data/vibration_to_noise_converter.py`
- `convert_vibration_to_noise.py` → `noise_data/convert_vibration_to_noise.py`
- `noise_analysis.py` → `noise_data/noise_analysis.py`

## Files Created
- `noise_data/__init__.py` - Package initialization file

## Files Updated

### 1. Main.py
- Updated import: `from noise_analysis import NoiseAnalysis` → `from noise_data.noise_analysis import NoiseAnalysis`

### 2. noise_data/convert_vibration_to_noise.py
- Updated import: `from vibration_to_noise_converter import VibrationToNoiseConverter` → `from .vibration_to_noise_converter import VibrationToNoiseConverter`
- Added fallback import for direct script execution

### 3. noise_data/noise_analysis.py
- Updated import: `from vibration_to_noise_converter import VibrationToNoiseConverter` → `from .vibration_to_noise_converter import VibrationToNoiseConverter`
- Added fallback import for direct script execution

### 4. VIBRATION_TO_NOISE_CONVERSION_README.md
- Updated file structure documentation
- Updated all import examples to use new paths:
  - `from vibration_to_noise_converter import VibrationToNoiseConverter` → `from noise_data.vibration_to_noise_converter import VibrationToNoiseConverter`
  - `from noise_analysis import NoiseAnalysis` → `from noise_data.noise_analysis import NoiseAnalysis`

## New Package Structure
```
noise_data/
├── __init__.py                    # Package initialization
├── vibration_to_noise_converter.py # Main converter class
├── convert_vibration_to_noise.py   # Standalone conversion script
├── noise_analysis.py              # Enhanced noise analysis with conversion
└── database/                      # Original noise data
```

## Import Methods

### 1. Package Import (Recommended)
```python
from noise_data import NoiseAnalysis, VibrationToNoiseConverter
```

### 2. Direct Module Import
```python
from noise_data.noise_analysis import NoiseAnalysis
from noise_data.vibration_to_noise_converter import VibrationToNoiseConverter
```

### 3. Standalone Script Execution
The scripts in `noise_data/` can still be run directly:
```bash
python noise_data/convert_vibration_to_noise.py --help
```

## Testing Results
✅ All imports work correctly
✅ All basic functionality tests pass
✅ Standalone scripts work with fallback imports
✅ Main application works with updated imports

## Benefits of Reorganization
1. **Better Organization**: Related files are grouped together
2. **Cleaner Root Directory**: Reduces clutter in the main project folder
3. **Proper Package Structure**: Makes the code more modular and maintainable
4. **Flexible Imports**: Supports both package and direct imports
5. **Backward Compatibility**: Existing functionality is preserved

## Usage After Reorganization

### Running the Main Application
```bash
python Main.py
```

### Running Standalone Conversion
```bash
python noise_data/convert_vibration_to_noise.py
```

### Using in Your Own Code
```python
from noise_data import NoiseAnalysis, VibrationToNoiseConverter

# Initialize
analyzer = NoiseAnalysis()
converter = VibrationToNoiseConverter()

# Use as before
```

## Notes
- All existing functionality is preserved
- No breaking changes to the API
- Paths in configuration files remain the same (relative to project root)
- The reorganization improves code organization without affecting functionality

## Updated Configuration (Latest)
- **Sampling Rate**: Updated from 1 kHz to 50 kHz (50,000 Hz)
- **Record Time**: 60 seconds
- **Signal Length**: 3,000,000 samples (50,000 Hz × 60 seconds)
- **Nyquist Frequency**: 25 kHz (half of sampling rate)
- **Maximum Detectable Frequency**: 25 kHz

## Gear Specifications (Latest)
- **Driven Gear (Checked)**: 35 teeth
- **Driving Gear**: 18 teeth
- **Gear Ratio**: 18/35 = 0.514
- **Speed Range**: 15-60 Hz (extracted from folder names like R45_L10)
- **Load Range**: 5-15 Nm (extracted from folder names like R45_L10)
- **Gear Mesh Frequency**: Automatically calculated (e.g., 810 Hz for R45)
- **Harmonics**: Automatically calculated (1620, 2430, 3240 Hz for R45)
