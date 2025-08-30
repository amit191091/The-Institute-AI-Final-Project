#!/usr/bin/env python3
"""
Main Orchestrator for All Teeth Analysis
"""

import sys
import os
import traceback
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import *
    from batch_processor import *
    from data_utils import *
    from visualization import *
    
    def save_results_to_table_format(all_teeth_results, filename):
        """
        Save results in table format with tooth numbers as rows and wear cases as columns
        """
        if not all_teeth_results:
            print("❌ No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_teeth_results)
        
        # Pivot the data to create the table format
        # Rows: tooth_number, Columns: wear_case, Values: wear_depth_um
        table_df = df.pivot(index='tooth_number', columns='wear_case', values='wear_depth_um')
        
        # Reset index to make tooth_number a regular column
        table_df = table_df.reset_index()
        
        # Rename the index column to 'Tooth'
        table_df = table_df.rename(columns={'tooth_number': 'Tooth'})
        
        # Rename wear case columns to be more descriptive
        wear_case_columns = {col: f'W{col}' for col in table_df.columns if col != 'Tooth'}
        table_df = table_df.rename(columns=wear_case_columns)
        
        # Round all numeric columns to 1 decimal place
        numeric_columns = [col for col in table_df.columns if col != 'Tooth']
        table_df[numeric_columns] = table_df[numeric_columns].round(1)
        
        # Save the table format
        table_df.to_csv(filename, index=False)
        
        print(f"✅ Results saved in table format to '{filename}'")
        print(f"📊 Table shape: {table_df.shape} (rows: teeth, columns: wear cases)")
        
        return table_df
    
    def main():
        """
        Main function to run the integrated gear wear analysis
        """
        print("🔧 INTEGRATED GEAR WEAR ANALYSIS AGENT - SINGLE IMAGE VERSION")
        print("=" * 70)
        
        # Check if we have gear images to analyze
        image_folder = "../database"
        if not os.path.exists(image_folder):
            print(f"❌ Image folder not found: {image_folder}")
            print("ℹ️ Please place gear images in the database folder")
            return
        
        # Analyze all teeth from gear images
        print("\n📊 ALL TEETH ANALYSIS FROM GEAR IMAGES")
        print("-" * 50)
        all_teeth_results = analyze_all_teeth_wear_from_images(image_folder)
        
        if all_teeth_results:
            # Enforce monotonicity
            all_teeth_results = enforce_monotonicity_for_all_teeth(all_teeth_results)
            
            # Create visualization (commented out to avoid duplication with plot_results.py)
            # create_visualization(all_teeth_results, 
            #                    "All Teeth Wear Analysis from Gear Images", 
            #                    "../all_teeth_analysis_graph.png")
            
            # Save results in table format
            save_results_to_table_format(all_teeth_results, "../all_teeth_results.csv")
            
            # Calculate statistics
            wear_cases = {}
            for result in all_teeth_results:
                wear_case = result["wear_case"]
                if wear_case not in wear_cases:
                    wear_cases[wear_case] = []
                wear_cases[wear_case].append(result)
            
            print(f"📊 Analysis Summary:")
            print(f"   Total wear cases analyzed: {len(wear_cases)}")
            print(f"   Total tooth measurements: {len(all_teeth_results)}")
            print(f"   Wear depth range: {min([r['wear_depth_um'] for r in all_teeth_results]):.1f} - {max([r['wear_depth_um'] for r in all_teeth_results]):.1f} µm")
        
        print("\n🎉 INTEGRATED ANALYSIS COMPLETED!")
        print("=" * 70)
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(f"❌ Error occurred: {type(e).__name__}: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)