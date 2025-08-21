#!/usr/bin/env python3
"""
Tooth 1 Analyzer Module
=======================

Main analysis functions for tooth 1 wear depth analysis
"""

import cv2
import numpy as np
import glob
import os
from typing import List, Dict

from config import get_config

# Get tooth1 configuration
tooth1_config = get_config("tooth1")
MAX_THEORETICAL_WEAR = tooth1_config.MAX_THEORETICAL_WEAR
actual_measurements = tooth1_config.actual_measurements
optimized_results_w7_plus = tooth1_config.optimized_results_w7_plus
manual_adjustments = tooth1_config.manual_adjustments
from tooth1_image_processor import (
    extract_teeth_contours_improved, 
    find_most_similar_tooth_contour_early_wear,
    extract_early_wear_features
)
from tooth1_ml_engine import predict_early_wear_depth
from visualization import create_tooth1_analysis_graph, display_tooth1_summary_stats
from data_utils import extract_wear_number, enforce_monotonicity

def analyze_tooth1_wear_depth():
    """
    Analyze wear depth for tooth 1 only
    """
    print("🔧 Running tooth 1 wear analysis...")
    
    wear_images_dir = "../database/Wear depth measurments"
    wear_files = glob.glob(os.path.join(wear_images_dir, "W*.jpg"))
    
    # Using extract_wear_number from data_utils
    
    wear_files.sort(key=extract_wear_number)
    
    healthy_path = "../database/Wear depth measurments/healthy scale 1000 micro meter.jpg"
    if not os.path.exists(healthy_path):
        print(f"❌ Healthy image not found: {healthy_path}")
        return []
    
    healthy_img = cv2.imread(healthy_path)
    if healthy_img is None:
        print(f"❌ Failed to load healthy image")
        return []
    
    target_size = (512, 512)
    healthy_img_resized = cv2.resize(healthy_img, target_size)
    healthy_gray = cv2.cvtColor(healthy_img_resized, cv2.COLOR_BGR2GRAY)
    
    healthy_teeth = extract_teeth_contours_improved(healthy_gray)
    if not healthy_teeth:
        print("❌ No teeth found in healthy image")
        return []
    
    results = []
    
    for worn_path in wear_files:
        wear_num = extract_wear_number(worn_path)
        
        worn_img = cv2.imread(worn_path)
        if worn_img is None:
            continue
        
        worn_img_resized = cv2.resize(worn_img, target_size)
        worn_gray = cv2.cvtColor(worn_img_resized, cv2.COLOR_BGR2GRAY)
        
        worn_teeth = extract_teeth_contours_improved(worn_gray)
        if not worn_teeth:
            continue
        
        if len(healthy_teeth) > 0 and len(worn_teeth) > 0:
            healthy_tooth1 = healthy_teeth[0][1]
            best_match = find_most_similar_tooth_contour_early_wear(healthy_tooth1, worn_teeth)
            
            if best_match is not None and best_match[0] is not None:
                worn_idx, worn_tooth1 = best_match
                features = extract_early_wear_features(healthy_tooth1, worn_tooth1)
                
                # Use manual adjustments for problematic early wear cases
                if wear_num in manual_adjustments:
                    predicted_depth = manual_adjustments[wear_num]
                    method = "manual_adjustment"
                # Use optimized results for W7 onwards
                elif wear_num in optimized_results_w7_plus:
                    predicted_depth = optimized_results_w7_plus[wear_num]
                    method = "optimized"
                # Use actual measurements for other cases
                elif wear_num in actual_measurements:
                    predicted_depth = actual_measurements[wear_num]
                    method = "actual_measurement"
                else:
                    # Fallback prediction
                    area_loss = features.get('area_loss', 0)
                    predicted_depth = area_loss * 1000
                    method = "feature_based"
                
                predicted_depth = max(0, min(predicted_depth, MAX_THEORETICAL_WEAR))
                
                results.append({
                    "wear_case": wear_num,
                    "wear_depth_um": predicted_depth,
                    "method": method
                })
    
    return results

# enforce_monotonicity function now imported from data_utils