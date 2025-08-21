#!/usr/bin/env python3
"""
Data Loader Module
==========================

Data loading and processing functions for all analysis types:
- all_teeth: Full gear analysis data loading
- tooth1: Single tooth analysis data loading  
- picture: Picture analysis data loading for visualization
"""

import os
import csv
from typing import Dict, Tuple, Union, List
from config import get_config

def load_single_tooth_data(analysis_type="picture") -> Dict[int, float]:
    """
    Load single tooth analysis results from CSV file
    
    Args:
        analysis_type: Type of analysis ("picture", "tooth1")
    
    Returns:
        Dict[int, float]: Dictionary mapping wear case to wear depth
    """
    config = get_config(analysis_type)
    
    if analysis_type == "picture":
        single_tooth_file = config.SINGLE_TOOTH_RESULTS_FILE
    else:
        single_tooth_file = config.RESULTS_FILE
    
    if not os.path.exists(single_tooth_file):
        print(f"❌ Single tooth results file not found: {single_tooth_file}")
        return {}
    
    single_tooth_data = {}
    try:
        with open(single_tooth_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wear_case = int(row['wear_case'])
                wear_depth = float(row['wear_depth_um'])
                single_tooth_data[wear_case] = wear_depth
        
        if analysis_type == "picture":  # Only show message for picture analysis
            print(f"✅ Loaded {len(single_tooth_data)} single tooth results")
        return single_tooth_data
        
    except Exception as e:
        print(f"❌ Error loading single tooth data: {str(e)}")
        return {}

def load_all_teeth_data(analysis_type="picture") -> Dict[int, Dict[int, float]]:
    """
    Load all teeth analysis results from CSV file
    
    Args:
        analysis_type: Type of analysis ("picture", "all_teeth")
    
    Returns:
        Dict[int, Dict[int, float]]: Nested dictionary mapping wear case to tooth number to wear depth
    """
    config = get_config(analysis_type)
    
    if analysis_type == "picture":
        all_teeth_file = config.ALL_TEETH_RESULTS_FILE
    else:
        all_teeth_file = config.RESULTS_FILE
    
    if not os.path.exists(all_teeth_file):
        print(f"❌ All teeth results file not found: {all_teeth_file}")
        return {}
    
    all_teeth_data = {}
    try:
        with open(all_teeth_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wear_case = int(row['wear_case'])
                tooth_number = int(row['tooth_number'])
                wear_depth = float(row['wear_depth_um'])
                
                if wear_case not in all_teeth_data:
                    all_teeth_data[wear_case] = {}
                all_teeth_data[wear_case][tooth_number] = wear_depth
        
        if analysis_type == "picture":  # Only show message for picture analysis
            print(f"✅ Loaded {len(all_teeth_data)} wear cases with all teeth data")
        return all_teeth_data
        
    except Exception as e:
        print(f"❌ Error loading all teeth data: {str(e)}")
        return {}

def load_analysis_data(analysis_type="picture") -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
    """
    Load both single tooth and all teeth analysis data
    
    Args:
        analysis_type: Type of analysis ("picture", "all_teeth", "tooth1")
    
    Returns:
        Tuple[Dict[int, float], Dict[int, Dict[int, float]]]: Single tooth and all teeth data
    """
    single_tooth_data = load_single_tooth_data(analysis_type)
    all_teeth_data = load_all_teeth_data(analysis_type)
    
    return single_tooth_data, all_teeth_data

def load_results_csv(file_path: str, analysis_type: str = "all_teeth") -> Union[Dict, List]:
    """
    Generic CSV loader for results files
    
    Args:
        file_path: Path to CSV file
        analysis_type: Type of analysis to determine data structure
        
    Returns:
        Union[Dict, List]: Loaded data in appropriate structure
    """
    if not os.path.exists(file_path):
        print(f"❌ Results file not found: {file_path}")
        return {} if analysis_type in ["tooth1", "picture"] else []
    
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if analysis_type == "tooth1":
                # Return dictionary mapping wear_case to wear_depth
                data = {}
                for row in reader:
                    wear_case = int(row['wear_case'])
                    wear_depth = float(row['wear_depth_um'])
                    data[wear_case] = wear_depth
                return data
                
            elif analysis_type == "all_teeth":
                # Return nested dictionary mapping wear_case -> tooth_number -> wear_depth
                data = {}
                for row in reader:
                    wear_case = int(row['wear_case'])
                    tooth_number = int(row['tooth_number'])
                    wear_depth = float(row['wear_depth_um'])
                    
                    if wear_case not in data:
                        data[wear_case] = {}
                    data[wear_case][tooth_number] = wear_depth
                return data
                
            else:
                # Return list of dictionaries for other analysis types
                return list(reader)
                
    except Exception as e:
        print(f"❌ Error loading results from {file_path}: {str(e)}")
        return {} if analysis_type in ["tooth1", "picture"] else []

def calculate_statistics(all_teeth_data: Dict[int, Dict[int, float]]) -> Dict[int, Dict[str, float]]:
    """
    Calculate statistics for each wear case
    
    Args:
        all_teeth_data: Dictionary of all teeth data
        
    Returns:
        Dict[int, Dict[str, float]]: Statistics for each wear case
    """
    statistics = {}
    
    for wear_case, teeth_data in all_teeth_data.items():
        teeth_depths = list(teeth_data.values())
        
        if teeth_depths:
            mean_depth = sum(teeth_depths) / len(teeth_depths)
            variance = sum((x - mean_depth) ** 2 for x in teeth_depths) / len(teeth_depths)
            std_dev = variance ** 0.5
            min_depth = min(teeth_depths)
            max_depth = max(teeth_depths)
            range_depth = max_depth - min_depth
            
            statistics[wear_case] = {
                'mean': mean_depth,
                'std_dev': std_dev,
                'min': min_depth,
                'max': max_depth,
                'range': range_depth,
                'count': len(teeth_depths)
            }
    
    return statistics

# Backward compatibility functions (maintain original function names)
def load_single_tooth_data_picture():
    """Backward compatibility for picture analysis"""
    return load_single_tooth_data("picture")

def load_all_teeth_data_picture():
    """Backward compatibility for picture analysis"""
    return load_all_teeth_data("picture")

def load_analysis_data_picture():
    """Backward compatibility for picture analysis"""
    return load_analysis_data("picture")
