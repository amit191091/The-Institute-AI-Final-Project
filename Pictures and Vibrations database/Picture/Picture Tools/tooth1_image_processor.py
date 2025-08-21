#!/usr/bin/env python3
"""
Tooth 1 Image Processing Module
==============================

Image processing functions for tooth 1 wear depth analysis
"""

import sys
import os
import cv2
import numpy as np
from typing import List, Tuple

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import identical function from Picture system (now in same directory)
from image_processor import calculate_contour_centroid

# Import configuration
from config import get_config

# Get tooth1 configuration
tooth1_config = get_config("tooth1")
EXPECTED_TOOTH_COUNT = tooth1_config.EXPECTED_TOOTH_COUNT

def extract_teeth_contours_improved(gray_image: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """
    Extract individual tooth contours with improved detection to get exactly 35 teeth
    """
    # Apply preprocessing to improve tooth detection
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    valid_contours = []
    center_y, center_x = gray_image.shape[0] // 2, gray_image.shape[1] // 2
    
    for contour in contours:
        area = cv2.contourArea(contour)
        min_area = 50
        max_area = 8000
        
        if min_area <= area <= max_area:
            cx, cy = calculate_contour_centroid(contour)
            distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            min_distance = 30
            max_distance = 300
            
            if min_distance <= distance_from_center <= max_distance:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 <= aspect_ratio <= 5.0:
                    valid_contours.append((contour, area, cx, cy))
    
    if len(valid_contours) == 0:
        return []
    
    teeth_with_angles = []
    for contour, area, cx, cy in valid_contours:
        angle = np.arctan2(cy - center_y, cx - center_x)
        angle_deg = (np.degrees(angle) + 360) % 360
        teeth_with_angles.append((angle_deg, contour, area))
    
    teeth_with_angles.sort(key=lambda x: x[0])
    
    if len(teeth_with_angles) > EXPECTED_TOOTH_COUNT:
        ideal_spacing = 360.0 / EXPECTED_TOOTH_COUNT
        selected_teeth = []
        for i in range(EXPECTED_TOOTH_COUNT):
            ideal_angle = i * ideal_spacing
            closest_contour = min(teeth_with_angles, 
                                key=lambda x: abs(x[0] - ideal_angle))
            selected_teeth.append(closest_contour)
        teeth_with_angles = selected_teeth
    
    indexed_teeth = []
    for i, (angle, contour, area) in enumerate(teeth_with_angles):
        indexed_teeth.append((i + 1, contour))
    
    return indexed_teeth

def estimate_scale_factor_from_gear(healthy_contour_area: float) -> float:
    """
    Estimate scale factor (µm/pixel) based on gear specifications and healthy tooth area
    """
    STANDARD_TOOTH_THICKNESS = tooth1_config.GEAR_SPECS['tooth_thickness']
    STANDARD_TOOTH_HEIGHT = tooth1_config.GEAR_SPECS['tooth_height']
    
    expected_tooth_area_mm2 = STANDARD_TOOTH_THICKNESS * STANDARD_TOOTH_HEIGHT
    expected_tooth_area_pixels = healthy_contour_area
    
    if expected_tooth_area_pixels > 0:
        scale_factor_mm_per_pixel = np.sqrt(expected_tooth_area_mm2 / expected_tooth_area_pixels)
        scale_factor_um_per_pixel = scale_factor_mm_per_pixel * 1000
        
        calibration_factor = 0.8
        refined_scale_factor = scale_factor_um_per_pixel * calibration_factor
        refined_scale_factor = max(4.0, min(refined_scale_factor, 10.0))
        
        return refined_scale_factor
    else:
        return 6.0

def extract_early_wear_features(healthy_contour, worn_contour, target_um_per_px=6.0):
    """
    Extract features optimized for early wear detection (W1-W6)
    """
    features = {}
    
    healthy_area = cv2.contourArea(healthy_contour)
    worn_area = cv2.contourArea(worn_contour)
    healthy_perimeter = cv2.arcLength(healthy_contour, True)
    worn_perimeter = cv2.arcLength(worn_contour, True)
    
    estimated_scale_factor = estimate_scale_factor_from_gear(healthy_area)
    effective_scale_factor = (target_um_per_px * 0.3 + estimated_scale_factor * 0.7)
    
    features['area_ratio'] = worn_area / max(healthy_area, 1)
    features['area_loss'] = (healthy_area - worn_area) / max(healthy_area, 1)
    features['area_loss_squared'] = features['area_loss'] ** 2
    features['area_loss_cubic'] = features['area_loss'] ** 3
    features['area_loss_sqrt'] = np.sqrt(features['area_loss'])
    
    features['perimeter_ratio'] = worn_perimeter / max(healthy_perimeter, 1)
    features['perimeter_loss'] = (healthy_perimeter - worn_perimeter) / max(healthy_perimeter, 1)
    
    healthy_bbox = cv2.boundingRect(healthy_contour)
    worn_bbox = cv2.boundingRect(worn_contour)
    
    features['height_ratio'] = worn_bbox[3] / max(healthy_bbox[3], 1)
    features['width_ratio'] = worn_bbox[2] / max(healthy_bbox[2], 1)
    features['height_loss'] = (healthy_bbox[3] - worn_bbox[3]) / max(healthy_bbox[3], 1)
    features['width_loss'] = (healthy_bbox[2] - worn_bbox[2]) / max(healthy_bbox[2], 1)
    features['height_loss_squared'] = features['height_loss'] ** 2
    
    healthy_hull = cv2.convexHull(healthy_contour)
    worn_hull = cv2.convexHull(worn_contour)
    
    healthy_hull_area = cv2.contourArea(healthy_hull)
    worn_hull_area = cv2.contourArea(worn_hull)
    
    features['hull_area_ratio'] = worn_hull_area / max(healthy_hull_area, 1)
    features['hull_area_loss'] = (healthy_hull_area - worn_hull_area) / max(healthy_hull_area, 1)
    
    healthy_solidity = healthy_area / max(cv2.contourArea(healthy_hull), 1)
    worn_solidity = worn_area / max(cv2.contourArea(worn_hull), 1)
    
    features['solidity_ratio'] = worn_solidity / max(healthy_solidity, 1)
    features['solidity_loss'] = healthy_solidity - worn_solidity
    
    mask_size = (512, 512)
    healthy_mask = np.zeros(mask_size, dtype=np.uint8)
    worn_mask = np.zeros(mask_size, dtype=np.uint8)
    
    cv2.fillPoly(healthy_mask, [healthy_contour], 255)
    cv2.fillPoly(worn_mask, [worn_contour], 255)
    
    healthy_dt = cv2.distanceTransform(healthy_mask, cv2.DIST_L2, 5)
    worn_dt = cv2.distanceTransform(worn_mask, cv2.DIST_L2, 5)
    
    features['dt_max_diff'] = np.max(healthy_dt) - np.max(worn_dt)
    features['dt_mean_diff'] = np.mean(healthy_dt) - np.mean(worn_dt)
    features['dt_median_diff'] = np.median(healthy_dt) - np.median(worn_dt)
    features['dt_std_diff'] = np.std(healthy_dt) - np.std(worn_dt)
    
    healthy_edges = cv2.Canny(healthy_mask, 50, 150)
    worn_edges = cv2.Canny(worn_mask, 50, 150)
    
    features['edge_density_ratio'] = np.sum(worn_edges) / max(np.sum(healthy_edges), 1)
    features['edge_density_loss'] = (np.sum(healthy_edges) - np.sum(worn_edges)) / max(np.sum(healthy_edges), 1)
    
    for key in features:
        if 'loss' in key or 'diff' in key:
            features[key] *= effective_scale_factor
    
    return features

def find_most_similar_tooth_contour_early_wear(healthy_contour: np.ndarray, 
                                               worn_contours: list[tuple[int, np.ndarray]], 
                                               max_distance_threshold: float = 300.0) -> tuple[int, np.ndarray]:
    """
    Tooth matching optimized for early wear cases
    """
    if not worn_contours:
        return None, None
    
    healthy_centroid = calculate_contour_centroid(healthy_contour)
    healthy_area = cv2.contourArea(healthy_contour)
    healthy_perimeter = cv2.arcLength(healthy_contour, True)
    
    best_match = None
    best_score = float('inf')
    
    for idx, worn_contour in worn_contours:
        worn_area = cv2.contourArea(worn_contour)
        if 20 <= worn_area <= 5000:
            worn_centroid = calculate_contour_centroid(worn_contour)
            worn_perimeter = cv2.arcLength(worn_contour, True)
            
            centroid_distance = np.sqrt((healthy_centroid[0] - worn_centroid[0])**2 + 
                                       (healthy_centroid[1] - worn_centroid[1])**2)
            
            if centroid_distance > max_distance_threshold:
                continue
            
            w_distance = 0.2
            w_area = 0.5
            w_perimeter = 0.3
            
            area_diff = abs(worn_area - healthy_area) / max(healthy_area, 1)
            perimeter_diff = abs(worn_perimeter - healthy_perimeter) / max(healthy_perimeter, 1)
            distance_norm = centroid_distance / max_distance_threshold
            
            score = (w_distance * distance_norm + 
                    w_area * area_diff + 
                    w_perimeter * perimeter_diff)
            
            if score < best_score:
                best_score = score
                best_match = (idx, worn_contour)
    
    return best_match
