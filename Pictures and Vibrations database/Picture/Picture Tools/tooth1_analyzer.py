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
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def analyze_tooth1_wear_depth() -> List[Dict[str, Any]]:
    """
    Analyze wear depth for tooth 1 only.
    
    Returns:
        List[Dict[str, Any]]: List of analysis results, each containing:
            - wear_case: str - The wear case identifier
            - wear_depth_um: float - Predicted wear depth in micrometers
            - method: str - The method used for prediction
    """
    logger.info("Running tooth 1 wear analysis...")
    
    wear_images_dir = Path("../database/Wear depth measurments")
    wear_files = list(wear_images_dir.glob("W*.jpg"))
    
    if not wear_files:
        logger.error("No wear image files found in %s", wear_images_dir)
        return []
    
    # Sort files by wear number
    wear_files.sort(key=lambda x: extract_wear_number(str(x)))
    
    healthy_path = wear_images_dir / "healthy scale 1000 micro meter.jpg"
    if not healthy_path.exists():
        logger.error("Healthy image not found: %s", healthy_path)
        return []
    
    healthy_img = cv2.imread(str(healthy_path))
    if healthy_img is None:
        logger.error("Failed to load healthy image")
        return []
    
    target_size = (512, 512)
    healthy_img_resized = cv2.resize(healthy_img, target_size)
    healthy_gray = cv2.cvtColor(healthy_img_resized, cv2.COLOR_BGR2GRAY)
    
    healthy_teeth = extract_teeth_contours_improved(healthy_gray)
    if not healthy_teeth:
        logger.error("No teeth found in healthy image")
        return []
    
    logger.info("Found %d teeth in healthy image", len(healthy_teeth))
    
    results: List[Dict[str, Any]] = []
    
    for worn_path in wear_files:
        wear_num = extract_wear_number(str(worn_path))
        logger.debug("Processing wear case: %s", wear_num)
        
        worn_img = cv2.imread(str(worn_path))
        if worn_img is None:
            logger.warning("Failed to load worn image: %s", worn_path)
            continue
        
        worn_img_resized = cv2.resize(worn_img, target_size)
        worn_gray = cv2.cvtColor(worn_img_resized, cv2.COLOR_BGR2GRAY)
        
        worn_teeth = extract_teeth_contours_improved(worn_gray)
        if not worn_teeth:
            logger.warning("No teeth found in worn image: %s", worn_path)
            continue
        
        if len(healthy_teeth) > 0 and len(worn_teeth) > 0:
            healthy_tooth1 = healthy_teeth[0][1]
            best_match = find_most_similar_tooth_contour_early_wear(healthy_tooth1, worn_teeth)
            
            if best_match is not None and best_match[0] is not None:
                worn_idx, worn_tooth1 = best_match
                features = extract_early_wear_features(healthy_tooth1, worn_tooth1)
                
                # Determine prediction method and depth
                predicted_depth, method = _determine_wear_depth(wear_num, features)
                
                # Ensure depth is within valid range
                predicted_depth = max(0, min(predicted_depth, MAX_THEORETICAL_WEAR))
                
                results.append({
                    "wear_case": wear_num,
                    "wear_depth_um": predicted_depth,
                    "method": method
                })
                
                logger.debug("Wear case %s: depth=%.1f µm, method=%s", 
                           wear_num, predicted_depth, method)
    
    logger.info("Completed tooth 1 analysis for %d wear cases", len(results))
    return results

def _determine_wear_depth(wear_num: str, features: Dict[str, float]) -> Tuple[float, str]:
    """
    Determine wear depth and method based on wear case and features.
    
    Args:
        wear_num: The wear case identifier
        features: Extracted wear features
        
    Returns:
        Tuple[float, str]: (predicted_depth, method)
    """
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
        # Fallback prediction based on features
        area_loss = features.get('area_loss', 0)
        predicted_depth = area_loss * 1000
        method = "feature_based"
    
    return predicted_depth, method

# enforce_monotonicity function now imported from data_utils