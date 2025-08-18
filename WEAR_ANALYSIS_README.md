# Wear Experiment Analysis System

This system is designed to analyze gear tooth wear from hand-drawn curves in wear experiment images. It focuses on measuring wear depth and progression based on the healthy tooth profile as a baseline.

## 🎯 Key Features

- **Hand-drawn curve analysis**: Detects and analyzes manually drawn tooth curves
- **Healthy baseline establishment**: Uses healthy tooth profiles as reference
- **Wear depth measurement**: Calculates wear percentages based on area and perimeter changes
- **Equal wear assumption**: Initially assumes all teeth are worn equally
- **Interactive curve drawing**: Manual annotation capability for precise measurements
- **Comprehensive reporting**: Generates detailed analysis reports and visualizations

## 📁 Image Requirements

The system expects images from the path:
```
C:\Users\amitl\Documents\AI Developers\PracticeBase\Final_project\gear_images\database\Wear depth measurments
```

### Image Naming Convention
The system automatically categorizes images based on filenames:
- **Wear cases**: Files with patterns like "W1", "W6", "W15" (primary format for wear analysis)
- **Alternative wear patterns**: "wear 10", "W10", "10 microns", "damage 10"
- **Healthy baseline**: Files containing keywords like "healthy", "baseline", "normal", "good", "h"
- **Scale information**: Extracts micrometer scale from filenames

## 🚀 Quick Start

### 1. Test the System
First, test if the system can access your images:
```bash
python test_wear_analysis.py
```

### 2. Run Automatic Analysis
For automatic curve detection and analysis:
```bash
python wear_measurement_system.py
```

### 3. Interactive Analysis (Recommended)
For precise hand-drawn curve analysis:
```python
from wear_measurement_system import WearMeasurementSystem

# Initialize system
system = WearMeasurementSystem()

# Run interactive analysis
results = system.run_complete_analysis(interactive_mode=True)
```

## 🎨 Interactive Curve Drawing

When using interactive mode, you can manually draw tooth curves:

### Instructions:
- **Left click**: Start/continue drawing a curve
- **Right click**: Finish the current curve
- **'r'**: Reset all curves
- **'s'**: Save and exit
- **'q'**: Quit without saving

### Best Practices:
1. **Draw along the tooth profile**: Follow the hand-drawn curves in your images
2. **Complete the curve**: Ensure the curve forms a closed shape
3. **Be consistent**: Use the same drawing style for all teeth
4. **Focus on wear areas**: Pay special attention to worn sections

## 📊 Analysis Output

The system generates several outputs:

### 1. JSON Results
- `wear_measurement_results.json`: Comprehensive analysis data
- Individual curve analysis files for each image

### 2. Visualization Plots
- `wear_progression.png`: Wear progression across different levels
- `wear_by_case.png`: Average wear by case
- `wear_distributions.png`: Histograms of wear distributions

### 3. Analysis Summary
- Total wear cases analyzed
- Average wear percentages
- Wear progression trends
- Recommendations

## 🔬 Analysis Methodology

### Wear Calculation
The system calculates wear using two methods:

1. **Area-based wear**:
   ```
   Wear (%) = ((Healthy_Area - Current_Area) / Healthy_Area) × 100
   ```

2. **Perimeter-based wear**:
   ```
   Wear (%) = ((Healthy_Perimeter - Current_Perimeter) / Healthy_Perimeter) × 100
   ```

### Baseline Establishment
- Uses healthy tooth profiles as reference
- Calculates average area and perimeter from healthy curves
- Establishes standard deviation for quality control

### Equal Wear Assumption
- Initially assumes all teeth in a wear case are worn equally
- Allows for statistical analysis of wear distribution
- Can identify uneven wear patterns

## 📈 Understanding Results

### Wear Levels
- **0-5%**: Minimal wear - normal operation
- **5-15%**: Moderate wear - monitor closely
- **15-30%**: Significant wear - consider maintenance
- **>30%**: Critical wear - immediate replacement recommended

### Key Metrics
- **Average wear**: Mean wear across all teeth
- **Wear distribution**: Spread of wear values
- **Wear progression**: Trend across different wear levels
- **Uneven wear**: Standard deviation indicating alignment issues

## 🛠️ Customization

### Adjusting Analysis Parameters
You can modify analysis parameters in the code:

```python
# In wear_experiment_analysis.py
# Adjust contour detection thresholds
if cv2.contourArea(contour) > 100:  # Minimum contour area
    # Process contour

# In tooth_curve_analyzer.py
# Adjust edge detection parameters
edges = cv2.Canny(blurred, 50, 150)  # Canny thresholds
```

### Adding New Wear Patterns
To add new wear case patterns:

```python
# In _extract_wear_case method
wear_patterns = [
    r'wear\s*(\d+)',  # "wear 10"
    r'W(\d+)',        # "W10"
    r'damage\s*(\d+)', # "damage 10"
    r'(\d+)\s*microns?', # "10 microns"
    r'(\d+)\s*μm',    # "10 μm"
    # Add your custom pattern here
    r'your_pattern_(\d+)',  # Custom pattern
]
```

## 🔧 Troubleshooting

### Common Issues

1. **No images found**:
   - Check the image path is correct
   - Ensure images have supported extensions (.jpg, .png, etc.)

2. **No curves detected**:
   - Try interactive mode for manual drawing
   - Adjust edge detection parameters
   - Check image quality and contrast

3. **Incorrect wear categorization**:
   - Review filename patterns
   - Add custom patterns if needed
   - Check for special characters in filenames

### Performance Tips

1. **Large image sets**: Process in batches
2. **Memory issues**: Close other applications
3. **Interactive mode**: Use for critical measurements only

## 📋 File Structure

```
Final_project/
├── wear_experiment_analysis.py      # Main wear analyzer
├── tooth_curve_analyzer.py          # Curve detection and analysis
├── wear_measurement_system.py       # Complete measurement system
├── test_wear_analysis.py           # System testing
├── wear_measurement_results.json    # Analysis results
├── wear_measurement_plots/          # Generated plots
└── WEAR_ANALYSIS_README.md         # This file
```

## 🤝 Integration with Existing System

The new wear analysis system can be integrated with your existing gear diagnosis system:

```python
# In Main.py, add:
from wear_measurement_system import WearMeasurementSystem

class GearWearDiagnosisAgent:
    def __init__(self):
        # ... existing code ...
        self.wear_measurement_system = WearMeasurementSystem()
    
    def run_wear_experiment_analysis(self):
        """Run wear experiment analysis"""
        return self.wear_measurement_system.run_complete_analysis()
```

## 📞 Support

For issues or questions:
1. Run the test script first: `python test_wear_analysis.py`
2. Check the troubleshooting section
3. Review the generated error messages
4. Ensure all dependencies are installed

## 📚 Dependencies

Required Python packages:
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `pathlib` (built-in)
- `json` (built-in)
- `re` (built-in)

Install with:
```bash
pip install opencv-python numpy matplotlib
```
