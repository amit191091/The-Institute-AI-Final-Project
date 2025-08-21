#!/usr/bin/env python3
"""
Image Processing Functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

def load_single_gear_image(image_path: str) -> np.ndarray:
    """
    Load a single gear image containing all teeth
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    print(f"📸 Loading gear image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    return image

def extract_teeth_from_gear_image(gear_image: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """
    Extract individual teeth from a single gear image
    Returns list of (tooth_number, tooth_image) tuples
    """
    print("🔍 Extracting teeth from gear image...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(gear_image, cv2.COLOR_BGR2GRAY)
    
    # Apply more aggressive preprocessing for contact sheet images
    # Use different thresholding approach
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple thresholding methods
    thresh_methods = [
        cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10),
        cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ]
    
    best_contours = []
    best_count = 0
    
    for thresh in thresh_methods:
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        valid_contours = []
        center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # More lenient area constraints for contact sheet images
            min_area = 20
            max_area = 15000
            
            if min_area <= area <= max_area:
                cx, cy = calculate_contour_centroid(contour)
                distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                # More lenient distance constraints
                min_distance = 10
                max_distance = 500
                
                if min_distance <= distance_from_center <= max_distance:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    # More lenient aspect ratio for contact sheet images
                    if 0.1 <= aspect_ratio <= 10.0:
                        valid_contours.append((contour, area, cx, cy))
        
        if len(valid_contours) > best_count:
            best_contours = valid_contours
            best_count = len(valid_contours)
    
    if len(best_contours) == 0:
        print("❌ No valid tooth contours found")
        return []
    
    print(f"   Found {len(best_contours)} potential tooth contours")
    
    # Sort teeth by angle around the center
    teeth_with_angles = []
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    
    for contour, area, cx, cy in best_contours:
        angle = np.arctan2(cy - center_y, cx - center_x)
        angle_deg = (np.degrees(angle) + 360) % 360
        teeth_with_angles.append((angle_deg, contour, area))
    
    teeth_with_angles.sort(key=lambda x: x[0])
    
    # If we have more than 35 teeth, select the best ones (teeth 1-35)
    if len(teeth_with_angles) > EXPECTED_TOOTH_COUNT:
        # Sort by area (larger contours are more likely to be teeth)
        teeth_with_angles.sort(key=lambda x: x[2], reverse=True)
        teeth_with_angles = teeth_with_angles[:EXPECTED_TOOTH_COUNT]
        # Re-sort by angle
        teeth_with_angles.sort(key=lambda x: x[0])
    
    # Extract individual tooth images
    extracted_teeth = []
    for i, (angle, contour, area) in enumerate(teeth_with_angles):
        tooth_number = i + 1  # Start from tooth 1 (teeth 1-35)
        
        # Get bounding rectangle with padding
        x, y, w, h = cv2.boundingRect(contour)
        padding = 15
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gear_image.shape[1], x + w + padding)
        y2 = min(gear_image.shape[0], y + h + padding)
        
        # Extract tooth region
        tooth_image = gear_image[y1:y2, x1:x2]
        
        # Enhance the tooth image
        enhanced_tooth = enhance_tooth_image(tooth_image)
        
        extracted_teeth.append((tooth_number, enhanced_tooth))
        # print(f"   Extracted tooth {tooth_number}: {enhanced_tooth.shape[1]}x{enhanced_tooth.shape[0]} pixels")  # Removed verbose output
    
    print(f"✅ Successfully extracted {len(extracted_teeth)} teeth")
    return extracted_teeth

def enhance_tooth_image(tooth_image: np.ndarray) -> np.ndarray:
    """
    Enhance individual tooth image for better analysis
    """
    # Convert to grayscale if needed
    if len(tooth_image.shape) == 3:
        gray = cv2.cvtColor(tooth_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = tooth_image.copy()
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Resize to standard size for analysis
    target_size = (200, 200)
    resized = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return resized

def calculate_contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    """Calculate centroid of a contour"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)
    else:
        return (0.0, 0.0)