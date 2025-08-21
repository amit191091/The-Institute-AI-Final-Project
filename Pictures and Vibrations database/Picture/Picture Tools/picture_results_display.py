#!/usr/bin/env python3
"""
Picture Analysis Results Display Module
=====================================

Results display and table formatting functions for picture analysis
"""

from typing import Dict
from config import TABLE_HEADERS

def format_table_header() -> str:
    """
    Format the table header for results display
    
    Returns:
        str: Formatted header string
    """
    headers = TABLE_HEADERS
    header_line = (f"{'Wear Case':<{headers['wear_case']}} "
                   f"{'Tooth 1':<{headers['tooth1']}} "
                   f"{'Range (1-35)':<{headers['range']}} "
                   f"{'Mean':<{headers['mean']}} "
                   f"{'Std Dev':<{headers['std_dev']}}")
    return header_line

def format_table_row(wear_case: int, tooth1_depth: float, teeth_range: str, 
                    mean_depth: float, std_dev: float) -> str:
    """
    Format a single table row for results display
    
    Args:
        wear_case: Wear case number
        tooth1_depth: Tooth 1 wear depth
        teeth_range: Range of wear depths for all teeth
        mean_depth: Mean wear depth
        std_dev: Standard deviation
        
    Returns:
        str: Formatted row string
    """
    headers = TABLE_HEADERS
    row = (f"{wear_case:<{headers['wear_case']}} "
           f"{tooth1_depth:<{headers['tooth1']}.1f} "
           f"{teeth_range:<{headers['range']}} "
           f"{mean_depth:<{headers['mean']}.1f} "
           f"{std_dev:<{headers['std_dev']}.1f}")
    return row

def calculate_integrated_statistics(single_tooth_data: Dict[int, float], 
                                  all_teeth_data: Dict[int, Dict[int, float]]) -> Dict[int, Dict]:
    """
    Calculate integrated statistics combining single tooth and all teeth data
    
    Args:
        single_tooth_data: Dictionary of single tooth results
        all_teeth_data: Dictionary of all teeth results
        
    Returns:
        Dict[int, Dict]: Statistics for each wear case
    """
    integrated_stats = {}
    
    for wear_case in sorted(single_tooth_data.keys()):
        tooth1_depth = single_tooth_data[wear_case]
        
        if wear_case in all_teeth_data:
            teeth_2_35_depths = list(all_teeth_data[wear_case].values())
            # Include tooth 1 in calculations
            all_teeth_depths = [tooth1_depth] + teeth_2_35_depths
            mean_depth = sum(all_teeth_depths) / len(all_teeth_depths)
            std_dev = (sum((x - mean_depth) ** 2 for x in all_teeth_depths) / len(all_teeth_depths)) ** 0.5
            teeth_range = f"{min(all_teeth_depths):.1f}-{max(all_teeth_depths):.1f}"
        else:
            mean_depth = tooth1_depth
            std_dev = 0
            teeth_range = f"{tooth1_depth:.1f}-{tooth1_depth:.1f}"
        
        integrated_stats[wear_case] = {
            'tooth1_depth': tooth1_depth,
            'teeth_range': teeth_range,
            'mean_depth': mean_depth,
            'std_dev': std_dev
        }
    
    return integrated_stats

def display_integrated_results_table(single_tooth_data: Dict[int, float], 
                                   all_teeth_data: Dict[int, Dict[int, float]]) -> None:
    """
    Display integrated results table combining single tooth and all teeth data
    
    Args:
        single_tooth_data: Dictionary of single tooth results
        all_teeth_data: Dictionary of all teeth results
    """
    try:
        if not single_tooth_data:
            print("❌ No single tooth data to display")
            return
        
        print("\n📊 RESULTS TABLE")
        print("=" * 50)
        
        # Calculate integrated statistics
        integrated_stats = calculate_integrated_statistics(single_tooth_data, all_teeth_data)
        
        # Display table header
        print(format_table_header())
        print("-" * 60)
        
        # Display table rows
        for wear_case, stats in integrated_stats.items():
            row = format_table_row(
                wear_case=wear_case,
                tooth1_depth=stats['tooth1_depth'],
                teeth_range=stats['teeth_range'],
                mean_depth=stats['mean_depth'],
                std_dev=stats['std_dev']
            )
            print(row)
        
        # Summary statistics
        print(f"\n📊 Total wear cases analyzed: {len(single_tooth_data)}")
        print(f"📊 Total teeth analyzed per case: 35")
        
    except Exception as e:
        print(f"❌ Error showing integrated results: {str(e)}")

def display_summary_statistics(single_tooth_data: Dict[int, float], 
                             all_teeth_data: Dict[int, Dict[int, float]]) -> None:
    """
    Display summary statistics for the analysis
    
    Args:
        single_tooth_data: Dictionary of single tooth results
        all_teeth_data: Dictionary of all teeth results
    """
    try:
        print("\n📈 SUMMARY STATISTICS")
        print("=" * 40)
        
        # Single tooth statistics
        if single_tooth_data:
            tooth1_depths = list(single_tooth_data.values())
            print(f"⚙️ Tooth 1 Analysis:")
            print(f"   Cases analyzed: {len(single_tooth_data)}")
            print(f"   Depth range: {min(tooth1_depths):.1f} - {max(tooth1_depths):.1f} µm")
        
        # All teeth statistics
        if all_teeth_data:
            all_depths = []
            for case_data in all_teeth_data.values():
                all_depths.extend(case_data.values())
            
            print(f"\n⚙️ All Teeth Analysis:")
            print(f"   Cases analyzed: {len(all_teeth_data)}")
            print(f"   Total measurements: {len(all_depths)}")
            print(f"   Depth range: {min(all_depths):.1f} - {max(all_depths):.1f} µm")
        
    except Exception as e:
        print(f"❌ Error displaying summary statistics: {str(e)}")
