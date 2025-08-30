#!/usr/bin/env python3
"""
Plot Both Analysis Files
========================
Script to plot both single tooth and all teeth analysis results
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add Picture Tools to path
sys.path.append('Picture Tools')
sys.path.append('./Picture Tools')  # Add Picture Tools directory for gear_parameters

def plot_single_tooth_results():
    """Plot single tooth analysis results"""
    try:
        # Load single tooth results
        df = pd.read_csv('single_tooth_results.csv')
        
        # Get actual measurements from config
        sys.path.append('Picture Tools')
        from config import get_config
        tooth1_config = get_config("tooth1")
        actual_measurements = tooth1_config.actual_measurements
        
        # Plot measured vs actual
        wear_cases = df['wear_case'].values
        measured_depths = df['wear_depth_um'].values
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
        plt.savefig('single_tooth_analysis_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Single tooth analysis plot saved as 'single_tooth_analysis_plot.png'")
        print(f"Wear depth range: {min(measured_depths):.1f} - {max(measured_depths):.1f} µm")
        
    except Exception as e:
        print(f"❌ Error plotting single tooth results: {e}")

def plot_all_teeth_results():
    """Plot all teeth analysis results using error bar graph and heatmap"""
    try:
        # Load all teeth results
        df = pd.read_csv('all_teeth_results.csv')
        
        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Group by wear case
        wear_cases = sorted(df['wear_case'].unique())
        
        # Calculate statistics for each wear case
        case_means = []
        case_stds = []
        
        for case in wear_cases:
            case_data = df[df['wear_case'] == case]['wear_depth_um'].values
            case_means.append(np.mean(case_data))
            case_stds.append(np.std(case_data))
        
        # Plot 1: Error bar graph (using existing analysis format)
        ax1.errorbar(wear_cases, case_means, yerr=case_stds, 
                    fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                    color='blue', ecolor='red', elinewidth=1.5, alpha=0.8)
        
        # Add mean line
        ax1.plot(wear_cases, case_means, 'o-', linewidth=3, markersize=10, 
                color='blue', label='Mean Wear Depth', alpha=0.9)
        
        # Customize plot
        ax1.set_xlabel('Wear Case', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Wear Depth (µm)', fontsize=12, fontweight='bold')
        ax1.set_title('All Teeth Wear Analysis from Gear Images', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Set axis limits to start from 0
        ax1.set_xlim(-1, max(wear_cases) + 1)
        ax1.set_ylim(0, max(case_means) * 1.1)
        
        # Add statistics text
        stats_text = f'Total Cases: {len(wear_cases)}\nTotal Teeth: {len(df)}\nMean Range: {min(case_means):.1f} - {max(case_means):.1f} µm'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Heatmap
        # Create pivot table for heatmap
        pivot_data = df.pivot(index='tooth_number', columns='wear_case', values='wear_depth_um')
        im = ax2.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Wear Depth (µm)', fontsize=12, fontweight='bold')
        
        # Set labels and title
        ax2.set_xlabel('Wear Case', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Tooth Number', fontsize=12, fontweight='bold')
        ax2.set_title('All Teeth: Wear Depth Heatmap', fontsize=14, fontweight='bold')
        
        # Set tick labels - show only every 5th wear case on x-axis
        x_ticks = [i for i in range(len(wear_cases)) if wear_cases[i] % 5 == 0 or wear_cases[i] == 1]
        x_tick_labels = [wear_cases[i] for i in x_ticks]
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_tick_labels)
        
        # Set y-axis ticks for tooth numbers
        y_ticks = range(0, len(pivot_data.index), 5)  # Show every 5th tooth
        y_tick_labels = [pivot_data.index[i] for i in y_ticks]
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_tick_labels)
        
        plt.tight_layout()
        plt.savefig('all_teeth_analysis_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("All teeth analysis plot saved as 'all_teeth_analysis_plot.png'")
        
        # Print summary
        print(f"Mean wear depth range: {min(case_means):.1f} - {max(case_means):.1f} µm")
        
    except Exception as e:
        print(f"❌ Error plotting all teeth results: {e}")

def main():
    """Main function to plot both analysis files"""
    print("PLOTTING BOTH ANALYSIS FILES")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists('single_tooth_results.csv'):
        print("❌ single_tooth_results.csv not found")
        return
    
    if not os.path.exists('all_teeth_results.csv'):
        print("❌ all_teeth_results.csv not found")
        return
    
    # Plot single tooth results
    print("\nPlotting single tooth analysis...")
    plot_single_tooth_results()
    
    # Plot all teeth results
    print("\nPlotting all teeth analysis...")
    plot_all_teeth_results()
    
    print("\nBoth plots completed successfully!")

if __name__ == "__main__":
    main()
