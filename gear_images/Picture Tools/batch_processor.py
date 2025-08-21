#!/usr/bin/env python3
"""
Batch Processing Functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from wear_analyzer import *

def analyze_all_teeth_wear_from_images(image_folder: str) -> List[Dict]:
    """
    Analyze all teeth wear from a folder of gear images
    """
    print("📊 ANALYZING ALL TEETH FROM GEAR IMAGES")
    print("=" * 50)
    
    # Find all gear images in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"❌ No image files found in {image_folder}")
        return []
    
    # Sort images by wear case number (numerical order)
    def get_wear_case_for_sorting(filepath):
        filename = os.path.basename(filepath)
        return extract_wear_case_from_filename(filename, 999)  # High default for sorting
    
    image_files.sort(key=get_wear_case_for_sorting)
    print(f"📁 Found {len(image_files)} gear images")
    print("📋 Processing order:")
    for i, img_file in enumerate(image_files[:10]):  # Show first 10
        wear_case = extract_wear_case_from_filename(os.path.basename(img_file), 0)
        print(f"   {i+1}. {os.path.basename(img_file)} -> Wear Case {wear_case}")
    if len(image_files) > 10:
        print(f"   ... and {len(image_files)-10} more files")
    
    all_results = []
    
    for i, image_path in enumerate(image_files):
        # Extract wear case from filename or use index
        filename = os.path.basename(image_path)
        wear_case = extract_wear_case_from_filename(filename, i + 1)
        
        # Skip files with wear_case = -1 (Healthy case)
        if wear_case == -1:
            print(f"⏭️ Skipping Healthy case: {filename}")
            continue
        
        print(f"\n📸 Processing image {i+1}/{len(image_files)}: {filename}")
        
        # Analyze this gear image
        results = analyze_single_gear_image(image_path, wear_case)
        all_results.extend(results)
    
    print(f"\n✅ Completed analysis of {len(image_files)} gear images")
    print(f"   Total measurements: {len(all_results)}")
    
    return all_results

def extract_wear_case_from_filename(filename: str, default_case: int) -> int:
    """
    Extract wear case number from filename
    """
    # Try to extract wear case from filename
    import re
    
    # Skip "Healthy" case - we don't want to include it
    if "healthy" in filename.lower():
        return -1  # Skip this file
    
    # Look for patterns like "Wear1", "Wear10", etc.
    patterns = [
        r'wear(\d+)',  # Matches "Wear1", "Wear10", etc.
        r'w(\d+)',     # Matches "W1", "W10", etc.
        r'(\d+)',      # Matches any number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    # If no pattern found, use default
    return default_case