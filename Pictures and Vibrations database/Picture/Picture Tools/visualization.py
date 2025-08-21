#!/usr/bin/env python3
"""
Visualization Module
===========================

Visualization functions for all analysis types:
- all_teeth: Error bar visualization for 35 teeth analysis
- tooth1: Comparison and accuracy visualization for single tooth analysis
- picture_analysis: Interactive plotting for picture analysis menu

Supports both file output and interactive display modes.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

def create_visualization(results: List[Dict], title: str, filename: str, analysis_type: str = "all_teeth", output_type: str = "file"):
    """
    Create visualization based on analysis type
    
    Args:
        results: List of result dictionaries
        title: Plot title
        filename: Output filename (for file mode)
        analysis_type: "all_teeth", "tooth1", or "picture_analysis"
        output_type: "file" (save to file) or "interactive" (show plots)
    """
    if analysis_type == "all_teeth":
        return create_error_bar_visualization(results, title, filename, output_type)
    elif analysis_type == "tooth1":
        return create_tooth1_analysis_graph(results, filename, output_type)
    elif analysis_type == "picture_analysis":
        return create_picture_analysis_plots(results, output_type)
    else:
        print(f"❌ Unknown analysis type: {analysis_type}")
        return False

def create_error_bar_visualization(results: List[Dict], title: str, filename: str, output_type: str = "file"):
    """
    Create error bar visualization for all-teeth analysis
    
    Args:
        results: List of result dictionaries
        title: Plot title
        filename: Output filename
        output_type: "file" (save) or "interactive" (show)
    """
    if not results:
        print("❌ No results to visualize")
        return False
    
    # Group by wear case
    wear_case_data = {}
    for result in results:
        wear_case = result["wear_case"]
        if wear_case not in wear_case_data:
            wear_case_data[wear_case] = []
        wear_case_data[wear_case].append(result["wear_depth_um"])
    
    # Calculate statistics for each wear case
    wear_cases = sorted(wear_case_data.keys())
    mean_depths = []
    std_depths = []
    
    for case in wear_cases:
        depths = wear_case_data[case]
        mean_depth = sum(depths) / len(depths)
        variance = sum((x - mean_depth) ** 2 for x in depths) / len(depths)
        std_depth = variance ** 0.5
        
        mean_depths.append(mean_depth)
        std_depths.append(std_depth)
    
    # Create error bar plot
    plt.figure(figsize=(12, 8))
    
    # Plot error bars
    plt.errorbar(wear_cases, mean_depths, yerr=std_depths, 
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                color='blue', ecolor='red', elinewidth=1.5, alpha=0.8)
    
    # Add mean line
    plt.plot(wear_cases, mean_depths, 'o-', linewidth=3, markersize=10, 
            color='blue', label='Mean Wear Depth', alpha=0.9)
    
    # Customize plot
    plt.xlabel('Wear Case', fontsize=12, fontweight='bold')
    plt.ylabel('Wear Depth (µm)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to start from 0
    plt.xlim(-1, max(wear_cases) + 1)
    plt.ylim(0, max(mean_depths) * 1.1)
    
    # Add statistics text
    stats_text = f'Total Cases: {len(wear_cases)}\nTotal Teeth: {len(results)}\nMean Range: {min(mean_depths):.1f} - {max(mean_depths):.1f} µm'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_type == "file":
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Error bar visualization saved as '{filename}'")
    else:  # interactive
        plt.show()
        print("✅ Error bar visualization displayed")
    
    print(f"📊 Statistics: {len(wear_cases)} wear cases, {len(results)} measurements")
    print(f"📊 Mean wear depth range: {min(mean_depths):.1f} - {max(mean_depths):.1f} µm")
    
    return True

def create_tooth1_analysis_graph(results, output_filename=None, output_type: str = "file"):
    """
    Create comprehensive visualization for tooth 1 analysis results
    
    Args:
        results: List of result dictionaries
        output_filename: Output filename
        output_type: "file" (save) or "interactive" (show)
    """
    if not results:
        print("❌ No results to visualize")
        return False
    
    # Set default filename if none provided
    if output_filename is None:
        picture_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_filename = os.path.join(picture_dir, "single_tooth_analysis.png")
    
    try:
        # Get tooth1 configuration for actual measurements
        tooth1_config = get_config("tooth1")
        actual_measurements = tooth1_config.actual_measurements
        
        # Prepare data
        wear_cases = [r['wear_case'] for r in results]
        measured_depths = [r['wear_depth_um'] for r in results]
        actual_depths = [actual_measurements.get(case, 0) for case in wear_cases]
        
        # Calculate errors and error percentages
        errors = [abs(measured - actual) for measured, actual in zip(measured_depths, actual_depths)]
        error_percentages = [abs(measured - actual) / actual * 100 if actual > 0 else 0 
                           for measured, actual in zip(measured_depths, actual_depths)]
        
        # Calculate precision (within 20% error threshold)
        correct_predictions = sum(1 for err_pct in error_percentages if err_pct <= 20)
        total_predictions = len(error_percentages)
        precision = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Measured vs Actual with precision info
        ax1.plot(wear_cases, measured_depths, 'bo-', linewidth=2, markersize=6, label='Measured')
        ax1.plot(wear_cases, actual_depths, 'rs-', linewidth=2, markersize=6, label='Actual')
        
        ax1.set_xlabel('Wear Case Number')
        ax1.set_ylabel('Wear Depth (µm)')
        ax1.set_title(f'Final Optimized: Measured vs Actual (Precision: {precision:.1f}%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(wear_cases) + 1)
        ax1.set_ylim(0, 1050)
        
        # Subplot 2: Measurement Error by Case
        bars = ax2.bar(wear_cases, error_percentages, alpha=0.7, color='darkblue')
        
        # Color bars based on threshold
        for i, (bar, err_pct) in enumerate(zip(bars, error_percentages)):
            if err_pct <= 20:
                bar.set_color('lightblue')
            else:
                bar.set_color('blue')
        
        ax2.set_xlabel('Wear Case Number')
        ax2.set_ylabel('Error (%)')
        ax2.set_title('Measurement Error by Case')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(wear_cases) + 1)
        ax2.set_ylim(0, max(error_percentages) * 1.1 if error_percentages else 25)
        
        # Add overall title
        fig.suptitle('Single Tooth Wear Analysis (Tooth 1)', fontsize=16, fontweight='bold')
        
        # Adjust layout and save/display
        plt.tight_layout()
        
        if output_type == "file":
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Tooth 1 visualization saved as '{output_filename}'")
        else:  # interactive
            plt.show()
            print("✅ Tooth 1 visualization displayed")
        
        print(f"📊 Precision: {precision:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"📊 Wear depth range: {min(measured_depths):.1f} - {max(measured_depths):.1f} µm")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating tooth 1 visualization: {e}")
        return False

def display_tooth1_summary_stats(results):
    """
    Display summary statistics for tooth 1 analysis
    """
    if not results:
        print("❌ No results to summarize")
        return
    
    # Get tooth1 configuration for actual measurements
    tooth1_config = get_config("tooth1")
    actual_measurements = tooth1_config.actual_measurements
    
    wear_cases = [r['wear_case'] for r in results]
    measured_depths = [r['wear_depth_um'] for r in results]
    actual_depths = [actual_measurements.get(case, 0) for case in wear_cases]
    
    # Calculate statistics
    min_measured = min(measured_depths)
    max_measured = max(measured_depths)
    
    # Calculate correlation
    correlation = np.corrcoef(measured_depths, actual_depths)[0, 1]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean([(m - a)**2 for m, a in zip(measured_depths, actual_depths)]))
    
    print("\n📊 TOOTH 1 ANALYSIS STATISTICS")
    print("=" * 40)
    print(f"📈 Measured Depths:")
    print(f"   Range: {min_measured:.1f} - {max_measured:.1f} µm")
    print(f"📈 Actual Depths:")
    print(f"📊 Accuracy Metrics:")
    print(f"   Correlation: {correlation:.3f}")
    print(f"   RMSE: {rmse:.1f} µm")
    print(f"   Total Cases: {len(results)}")

# Picture Analysis Visualization Functions
def create_tooth1_comparison_plot(wear_cases: List[int], measured_depths: List[float], 
                                 actual_depths: List[float]) -> None:
    """
    Create subplot for Tooth 1 measured vs actual comparison
    
    Args:
        wear_cases: List of wear case numbers
        measured_depths: List of measured wear depths
        actual_depths: List of actual wear depths
    """
    plt.subplot(2, 2, 1)
    plt.plot(wear_cases, measured_depths, 'bo-', linewidth=2, markersize=6, 
             label='Measured', color='blue')
    plt.plot(wear_cases, actual_depths, 'rs-', linewidth=2, markersize=6, 
             label='Actual', color='red')
    plt.xlabel('Wear Case')
    plt.ylabel('Wear Depth (µm)')
    plt.title('Tooth 1: Measured vs Actual Wear Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_error_analysis_plot(wear_cases: List[int], errors: List[float]) -> None:
    """
    Create subplot for error analysis
    
    Args:
        wear_cases: List of wear case numbers
        errors: List of absolute errors
    """
    plt.subplot(2, 2, 2)
    plt.bar(wear_cases, errors, alpha=0.7, color='orange')
    plt.xlabel('Wear Case')
    plt.ylabel('Absolute Error (µm)')
    plt.title('Tooth 1: Measurement Error vs Actual')
    plt.grid(True, alpha=0.3)

def create_all_teeth_error_plot(wear_cases: List[int], case_means: List[float], 
                               case_stds: List[float]) -> None:
    """
    Create subplot for all teeth mean with error bars
    
    Args:
        wear_cases: List of wear case numbers
        case_means: List of mean wear depths
        case_stds: List of standard deviations
    """
    plt.subplot(2, 2, 3)
    plt.errorbar(wear_cases, case_means, yerr=case_stds, fmt='ro-', 
                 linewidth=2, markersize=6, capsize=5, capthick=2,
                 color='red')
    plt.xlabel('Wear Case')
    plt.ylabel('Mean Wear Depth (µm)')
    plt.title('Mean Wear Depth (Teeth 2-35) with Error Bars')
    plt.grid(True, alpha=0.3)

def create_variation_plot(wear_cases: List[int], case_stds: List[float], 
                         cv_values: List[float]) -> None:
    """
    Create subplot for wear depth variation and deviations
    
    Args:
        wear_cases: List of wear case numbers
        case_stds: List of standard deviations
        cv_values: List of coefficient of variation values
    """
    plt.subplot(2, 2, 4)
    plt.plot(wear_cases, case_stds, 'go-', linewidth=2, markersize=6, 
             label='Standard Deviation', color='green')
    
    # Plot CV on secondary y-axis
    ax2 = plt.twinx()
    ax2.plot(wear_cases, cv_values, 'm^-', linewidth=2, markersize=6, 
             label='Coefficient of Variation (%)', color='magenta')
    ax2.set_ylabel('Coefficient of Variation (%)', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')
    
    plt.xlabel('Wear Case')
    plt.ylabel('Standard Deviation (µm)', color='green')
    plt.title('Wear Depth Variation and Deviations')
    plt.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

def create_all_teeth_distribution_plot(wear_cases: List[int], all_teeth_depths: List[float],
                                      all_teeth_cases: List[int], case_means: List[float]) -> None:
    """
    Create subplot for all teeth wear depth distribution
    
    Args:
        wear_cases: List of wear case numbers
        all_teeth_depths: List of all individual tooth depths
        all_teeth_cases: List of wear cases for each tooth
        case_means: List of mean wear depths
    """
    plt.subplot(2, 2, 1)
    plt.scatter(all_teeth_cases, all_teeth_depths, alpha=0.6, s=20, color='blue')
    plt.plot(wear_cases, case_means, 'ro-', linewidth=3, markersize=8, 
             label='Mean', color='red')
    plt.xlabel('Wear Case')
    plt.ylabel('Wear Depth (µm)')
    plt.title('All Teeth Wear Depth Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_coefficient_variation_plot(wear_cases: List[int], cv_values: List[float]) -> None:
    """
    Create subplot for coefficient of variation
    
    Args:
        wear_cases: List of wear case numbers
        cv_values: List of coefficient of variation values
    """
    plt.subplot(2, 2, 3)
    plt.plot(wear_cases, cv_values, 'go-', linewidth=2, markersize=6, color='green')
    plt.xlabel('Wear Case')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Relative Variability (CV) by Case')
    plt.grid(True, alpha=0.3)

def create_range_analysis_plot(wear_cases: List[int], case_ranges: List[float]) -> None:
    """
    Create subplot for range analysis
    
    Args:
        wear_cases: List of wear case numbers
        case_ranges: List of wear depth ranges
    """
    plt.subplot(2, 2, 4)
    plt.plot(wear_cases, case_ranges, 'mo-', linewidth=2, markersize=6, color='magenta')
    plt.xlabel('Wear Case')
    plt.ylabel('Range (µm)')
    plt.title('Wear Depth Range by Case')
    plt.grid(True, alpha=0.3)

def create_picture_analysis_plots(single_tooth_data: Dict[int, float], 
                                 all_teeth_data: Dict[int, Dict[int, float]] = None,
                                 output_type: str = "interactive") -> bool:
    """
    Create comprehensive visualization plots for picture analysis results
    
    Args:
        single_tooth_data: Dictionary of single tooth results
        all_teeth_data: Dictionary of all teeth results (optional)
        output_type: "file" (save) or "interactive" (show)
    """
    try:
        if not single_tooth_data:
            print("❌ No single tooth data to plot")
            return False
        
        # Get configuration for actual measurements
        config = get_config("picture_analysis")
        actual_measurements = config.actual_measurements
        
        # Prepare data
        wear_cases = sorted(single_tooth_data.keys())
        measured_depths = [single_tooth_data[case] for case in wear_cases]
        actual_depths = [actual_measurements.get(case, 0) for case in wear_cases]
        errors = [abs(measured - actual) for measured, actual in zip(measured_depths, actual_depths)]
        
        # Calculate statistics for all teeth if available
        case_means = []
        case_stds = []
        case_ranges = []
        cv_values = []
        
        if all_teeth_data:
            for case in wear_cases:
                if case in all_teeth_data:
                    teeth_depths = list(all_teeth_data[case].values())
                    mean_depth = sum(teeth_depths) / len(teeth_depths)
                    variance = sum((x - mean_depth) ** 2 for x in teeth_depths) / len(teeth_depths)
                    std_depth = variance ** 0.5
                    case_means.append(mean_depth)
                    case_stds.append(std_depth)
                    case_ranges.append(max(teeth_depths) - min(teeth_depths))
                    cv_values.append((std_depth / mean_depth) * 100 if mean_depth > 0 else 0)
                else:
                    case_means.append(0)
                    case_stds.append(0)
                    case_ranges.append(0)
                    cv_values.append(0)
        else:
            # Use zeros if no all_teeth_data
            case_means = [0] * len(wear_cases)
            case_stds = [0] * len(wear_cases)
            case_ranges = [0] * len(wear_cases)
            cv_values = [0] * len(wear_cases)
        
        # Prepare data for all teeth distribution
        all_teeth_depths = []
        all_teeth_cases = []
        if all_teeth_data:
            for case in wear_cases:
                if case in all_teeth_data:
                    for tooth_num, depth in all_teeth_data[case].items():
                        all_teeth_depths.append(depth)
                        all_teeth_cases.append(case)
        
        # Create plots
        # Figure 1: Tooth 1 Analysis with Actual Comparison
        plt.figure(figsize=(15, 10))
        
        create_tooth1_comparison_plot(wear_cases, measured_depths, actual_depths)
        create_error_analysis_plot(wear_cases, errors)
        create_all_teeth_error_plot(wear_cases, case_means, case_stds)
        create_variation_plot(wear_cases, case_stds, cv_values)
        
        plt.tight_layout()
        
        if output_type == "file":
            plt.savefig("picture_analysis_figure1.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Picture analysis Figure 1 saved as 'picture_analysis_figure1.png'")
        else:
            plt.show()
            print("✅ Picture analysis Figure 1 displayed")
        
        # Figure 2: All Teeth Analysis (only if all_teeth_data available)
        if all_teeth_data and all_teeth_depths:
            plt.figure(figsize=(15, 10))
            
            create_all_teeth_distribution_plot(wear_cases, all_teeth_depths, all_teeth_cases, case_means)
            create_all_teeth_error_plot(wear_cases, case_means, case_stds)
            create_coefficient_variation_plot(wear_cases, cv_values)
            create_range_analysis_plot(wear_cases, case_ranges)
            
            plt.tight_layout()
            
            if output_type == "file":
                plt.savefig("picture_analysis_figure2.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Picture analysis Figure 2 saved as 'picture_analysis_figure2.png'")
            else:
                plt.show()
                print("✅ Picture analysis Figure 2 displayed")
        
        print("✅ Picture analysis plots completed successfully!")
        if all_teeth_data:
            print("📊 Two figures generated:")
            print("   - Figure 1: Tooth 1 comparison with actual measurements")
            print("   - Figure 2: All teeth analysis with error bars and deviations")
        else:
            print("📊 One figure generated:")
            print("   - Figure 1: Tooth 1 comparison with actual measurements")
        
        return True
        
    except ImportError:
        print("❌ matplotlib not available. Please install matplotlib to plot results.")
        return False
    except Exception as e:
        print(f"❌ Error plotting results: {str(e)}")
        return False

# Backward compatibility functions
def create_tooth1_analysis_graph_legacy(results, output_filename=None):
    """Legacy function for backward compatibility"""
    if output_filename is None:
        picture_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_filename = os.path.join(picture_dir, "single_tooth_analysis.png")
    return create_tooth1_analysis_graph(results, output_filename)

def plot_results(single_tooth_data: Dict[int, float], 
                all_teeth_data: Dict[int, Dict[int, float]]) -> None:
    """
    Legacy function for backward compatibility with picture analysis menu
    """
    return create_picture_analysis_plots(single_tooth_data, all_teeth_data, output_type="interactive")

# Alternative function name to avoid conflicts
def plot_analysis_results(single_tooth_data: Dict[int, float], 
                         all_teeth_data: Dict[int, Dict[int, float]]) -> None:
    """
    Plot analysis results - alternative function name
    """
    return create_picture_analysis_plots(single_tooth_data, all_teeth_data, output_type="interactive")