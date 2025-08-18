"""
Gear Parameters Configuration
============================

This file contains all the key gear specifications and parameters used in the wear analysis system.
Modify these values to match your specific gear configuration.
"""

import math

# =============================================================================
# KHK SS3-35 GEAR SPECIFICATIONS (from datasheet)
# =============================================================================

# Core gear specifications (CONSTANTS - verified correct from KHK datasheet)
GEAR_MODULE = 3.0  # mm (from KHK datasheet)
TOOTH_COUNT = 35      # Number of teeth (from datasheet)
PRESSURE_ANGLE = 20.0  # degrees (standard)

# Material specifications (from datasheet)
GEAR_MATERIAL = "S45C"

BACKLASH_RANGE = (0.14, 0.32)  # mm


# =============================================================================
# STANDARD GEAR PARAMETERS (using verified standard spur gear formulas)
# =============================================================================

# Reference diameter (d) = zm 
# d = number of teeth × module
REFERENCE_DIAMETER = TOOTH_COUNT * GEAR_MODULE  # mm

# Tip diameter (da) = d + 2m 
# da = reference diameter + 2 × module
TIP_DIAMETER = REFERENCE_DIAMETER + 2 * GEAR_MODULE  # mm

# Root diameter (df) = d - 2.5m 
# df = reference diameter - 2.5 × module
ROOT_DIAMETER = REFERENCE_DIAMETER - 2.5 * GEAR_MODULE  # mm

# Addendum (ha) = 1.00m 
# ha = distance between reference line and tooth tip
STANDARD_ADDENDUM = 1.00 * GEAR_MODULE  # mm

# Dedendum (hf) = 1.25m 
# hf = distance between reference line and tooth root
STANDARD_DEDENDUM = 1.25 * GEAR_MODULE  # mm

# Tooth thickness (s) = πm / 2 
# s = half the value of pitch (p), where pitch (p) = πm
STANDARD_TOOTH_THICKNESS = math.pi * GEAR_MODULE / 2  # mm

# Total tooth height = Addendum + Dedendum
STANDARD_TOOTH_HEIGHT = STANDARD_ADDENDUM + STANDARD_DEDENDUM  # mm

# Circular pitch (p) = πm
CIRCULAR_PITCH = math.pi * GEAR_MODULE  # mm

# Base circle diameter = Reference diameter × cos(pressure angle)
BASE_CIRCLE_DIAMETER = REFERENCE_DIAMETER * math.cos(math.radians(PRESSURE_ANGLE))  # mm

# =============================================================================
# TOOTH THICKNESS DEFINITIONS (using verified formulas)
# =============================================================================

# Tooth thickness at pitch circle (verified formula)
# According to standard gear formula: s = πm / 2 
PITCH_CIRCLE_TOOTH_THICKNESS = math.pi * GEAR_MODULE / 2  # mm

# NOTE: Multi-position tooth thickness values (addendum_circle, base_circle, root_circle)
# are not available from standard formulas and would need specific gear manufacturer data.
# For wear analysis, we use the pitch circle tooth thickness as the reference.

# NOTE: Measurement heights for multi-position thickness analysis are not available
# from standard gear formulas or the KHK datasheet. These would need to be measured
# or obtained from specific gear manufacturer data.

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_standard_parameters():
    """Get all standard gear parameters as a dictionary"""
    return {
        'module': GEAR_MODULE,
        'tooth_count': TOOTH_COUNT,
        'pressure_angle': PRESSURE_ANGLE,
        'reference_diameter': REFERENCE_DIAMETER,
        'tip_diameter': TIP_DIAMETER,
        'root_diameter': ROOT_DIAMETER,
        'addendum': STANDARD_ADDENDUM,
        'dedendum': STANDARD_DEDENDUM,
        'tooth_thickness': STANDARD_TOOTH_THICKNESS,
        'tooth_height': STANDARD_TOOTH_HEIGHT,
        'circular_pitch': CIRCULAR_PITCH,
        'base_diameter': BASE_CIRCLE_DIAMETER,
        'material': GEAR_MATERIAL,

    }

def get_tooth_thickness_positions():
    """Get tooth thickness at pitch circle (only verified value available)"""
    return {
        'pitch_circle': PITCH_CIRCLE_TOOTH_THICKNESS
    }

def get_gear_specifications():
    """Get complete gear specifications from datasheet"""
    return {
        'brand': 'KHK',
        'model': 'SS3-35',
        'module': GEAR_MODULE,
        'tooth_count': TOOTH_COUNT,
        'pressure_angle': PRESSURE_ANGLE,
        'material': GEAR_MATERIAL,
        'backlash_range': BACKLASH_RANGE
    }

def validate_gear_parameters():
    """Validate that all gear parameters are reasonable"""
    issues = []
    
    if GEAR_MODULE <= 0:
        issues.append("Gear module must be positive")
    
    if TOOTH_COUNT <= 0:
        issues.append("Tooth count must be positive")
    
    if PRESSURE_ANGLE <= 0 or PRESSURE_ANGLE >= 90:
        issues.append("Pressure angle must be between 0 and 90 degrees")
    
    if STANDARD_TOOTH_HEIGHT <= 0:
        issues.append("Standard tooth height must be positive")
    
    return issues

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Validate parameters when file is run directly
    issues = validate_gear_parameters()
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All gear parameters are valid")
        print(f"🏭 Gear Brand: KHK")
        print(f"📋 Model: SS3-35")
        print(f"📏 Gear Module: {GEAR_MODULE} mm")
        print(f"⚙️ Tooth Count: {TOOTH_COUNT}")
        print(f"📐 Standard Tooth Height: {STANDARD_TOOTH_HEIGHT:.2f} mm")
        print(f"📐 Standard Tooth Thickness: {STANDARD_TOOTH_THICKNESS:.2f} mm")
        print(f"🔧 Material: {GEAR_MATERIAL}")
        print(f"📏 Backlash Range: {BACKLASH_RANGE[0]}-{BACKLASH_RANGE[1]} mm")
        
        print("\n📐 Calculated Parameters (using standard formulas):")
        print(f"   Reference Diameter: {REFERENCE_DIAMETER:.2f} mm")
        print(f"   Tip Diameter: {TIP_DIAMETER:.2f} mm")
        print(f"   Root Diameter: {ROOT_DIAMETER:.2f} mm")
        print(f"   Base Circle Diameter: {BASE_CIRCLE_DIAMETER:.2f} mm")
        print(f"   Addendum: {STANDARD_ADDENDUM:.2f} mm")
        print(f"   Dedendum: {STANDARD_DEDENDUM:.2f} mm")
        print(f"   Circular Pitch: {CIRCULAR_PITCH:.2f} mm")
        print(f"   Pitch Circle Tooth Thickness: {PITCH_CIRCLE_TOOTH_THICKNESS:.2f} mm")
