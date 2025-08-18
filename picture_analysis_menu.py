"""
Picture Analysis Submenu Module
Handles all picture analysis related menu options and functionality
"""

import os
import sys
import subprocess
import csv
import pandas as pd

def display_picture_analysis_menu(agent):
    """Display submenu for picture analysis"""
    while True:
        print("\n🖼️ PICTURE ANALYSIS SUBMENU")
        print("=" * 35)
        print("1. Database Process")
        print("2. Show Results")
        print("3. Return to Main Menu")
        
        try:
            choice = input(f"\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                display_database_process_menu(agent)
                    
            elif choice == "2":
                display_show_results_menu(agent)
                    
            elif choice == "3":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_database_process_menu(agent):
    """Display menu for database processing """
    while True:
        print("\n🔧 DATABASE PROCESS MENU")
        print("=" * 35)
        print("1. Run Tooth 1 Analysis")
        print("2. Run Integrated Gear Wear Analysis")
        print("3. Run Complete Analysis (Tooth 1 + All Teeth)")
        print("4. Return to Picture Analysis Menu")
        
        try:
            choice = input(f"\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🔧 Running Tooth 1 Analysis...")
                result = run_tooth1_analysis()
                if result:
                    print("✅ Tooth 1 analysis completed successfully!")
                else:
                    print("❌ Tooth 1 analysis failed!")
                    
            elif choice == "2":
                print("\n🔧 Running Integrated Gear Wear Analysis...")
                result = run_integrated_gear_wear_analysis()
                if result:
                    print("✅ Integrated gear wear analysis completed successfully!")
                else:
                    print("❌ Integrated gear wear analysis failed!")
                    
            elif choice == "3":
                print("\n🔧 Running Complete Analysis...")
                print("Step 1: Tooth 1 Analysis...")
                result1 = run_tooth1_analysis()
                if result1:
                    print("✅ Tooth 1 analysis completed!")
                    print("Step 2: Integrated Gear Wear Analysis...")
                    result2 = run_integrated_gear_wear_analysis()
                    if result2:
                        print("✅ Complete analysis finished successfully!")
                    else:
                        print("❌ Integrated analysis failed!")
                else:
                    print("❌ Tooth 1 analysis failed!")
                    
            elif choice == "4":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-4.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_show_results_menu(agent):
    """Display menu for showing results"""
    while True:
        print("\n📊 SHOW RESULTS MENU")
        print("=" * 35)
        print("1. Show Integrated Results Table")
        print("2. Plot Results")
        print("3. Return to Picture Analysis Menu")
        
        try:
            choice = input(f"\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                show_integrated_results_table()
                    
            elif choice == "2":
                plot_results()
                    
            elif choice == "3":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def run_tooth1_analysis():
    """Run tooth 1 wear depth analysis"""
    try:
        # Change to gear_images directory
        gear_images_dir = "gear_images"
        if not os.path.exists(gear_images_dir):
            print(f"❌ Directory not found: {gear_images_dir}")
            return False
        
        # Run the tooth 1 analysis script
        script_path = "analyze_tooth1_wear_depth.py"
        full_script_path = os.path.join(gear_images_dir, script_path)
        if not os.path.exists(full_script_path):
            print(f"❌ Script not found: {full_script_path}")
            return False
        
        print(f"🔧 Executing: {script_path}")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run([sys.executable, script_path], 
                              cwd=gear_images_dir, 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              env=env)
        
        if result.returncode == 0:
            print("✅ Tooth 1 analysis completed successfully!")
            return True
        else:
            print(f"❌ Tooth 1 analysis failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tooth 1 analysis: {str(e)}")
        return False

def run_integrated_gear_wear_analysis():
    """Run integrated gear wear analysis"""
    try:
        # Change to gear_images directory
        gear_images_dir = "gear_images"
        if not os.path.exists(gear_images_dir):
            print(f"❌ Directory not found: {gear_images_dir}")
            return False
        
        # Run the integrated gear wear analysis script
        script_path = "integrated_gear_wear_agent.py"
        full_script_path = os.path.join(gear_images_dir, script_path)
        if not os.path.exists(full_script_path):
            print(f"❌ Script not found: {full_script_path}")
            return False
        
        print(f"🔧 Executing: {script_path}")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run([sys.executable, script_path], 
                              cwd=gear_images_dir, 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              env=env)
        
        if result.returncode == 0:
            print("✅ Integrated gear wear analysis completed successfully!")
            return True
        else:
            print(f"❌ Integrated gear wear analysis failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running integrated gear wear analysis: {str(e)}")
        return False



def plot_results():
    """Plot results from CSV files"""
    try:
        single_tooth_file = os.path.join("gear_images", "single_tooth_results.csv")
        all_teeth_file = os.path.join("gear_images", "all_teeth_results.csv")
        
        if not os.path.exists(single_tooth_file):
            print("❌ Single tooth results file not found")
            return
        
        if not os.path.exists(all_teeth_file):
            print("❌ All teeth results file not found")
            return
        
        # Load data
        single_tooth_data = {}
        with open(single_tooth_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wear_case = int(row['wear_case'])
                wear_depth = float(row['wear_depth_um'])
                single_tooth_data[wear_case] = wear_depth
        
        all_teeth_data = {}
        with open(all_teeth_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wear_case = int(row['wear_case'])
                tooth_number = int(row['tooth_number'])
                wear_depth = float(row['wear_depth_um'])
                
                if wear_case not in all_teeth_data:
                    all_teeth_data[wear_case] = {}
                all_teeth_data[wear_case][tooth_number] = wear_depth
        
        # Load actual measurements for comparison
        actual_measurements = {
            1: 40, 2: 81, 3: 115, 4: 159, 5: 175, 6: 195, 7: 227, 8: 256, 9: 276, 10: 294,
            11: 305, 12: 323, 13: 344, 14: 378, 15: 400, 16: 417, 17: 436, 18: 450, 19: 466, 20: 488,
            21: 510, 22: 524, 23: 557, 24: 579, 25: 608, 26: 637, 27: 684, 28: 720, 29: 744, 30: 769,
            31: 797, 32: 825, 33: 853, 34: 890, 35: 932
        }
        
        # Create plots
        import matplotlib.pyplot as plt
        
        # Figure 1: Tooth 1 Analysis with Actual Comparison
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Measured vs Actual for Tooth 1
        plt.subplot(2, 2, 1)
        wear_cases = sorted(single_tooth_data.keys())
        measured_depths = [single_tooth_data[case] for case in wear_cases]
        actual_depths = [actual_measurements.get(case, 0) for case in wear_cases]
        
        plt.plot(wear_cases, measured_depths, 'bo-', linewidth=2, markersize=6, label='Measured')
        plt.plot(wear_cases, actual_depths, 'rs--', linewidth=2, markersize=6, label='Actual')
        plt.xlabel('Wear Case')
        plt.ylabel('Wear Depth (µm)')
        plt.title('Tooth 1: Measured vs Actual Wear Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Error Analysis for Tooth 1
        plt.subplot(2, 2, 2)
        errors = [abs(measured - actual) for measured, actual in zip(measured_depths, actual_depths)]
        plt.bar(wear_cases, errors, alpha=0.7, color='orange')
        plt.xlabel('Wear Case')
        plt.ylabel('Absolute Error (µm)')
        plt.title('Tooth 1: Measurement Error vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: All Teeth Mean by Case with Error Bars
        plt.subplot(2, 2, 3)
        case_means = []
        case_stds = []
        for case in wear_cases:
            if case in all_teeth_data:
                teeth_depths = list(all_teeth_data[case].values())
                mean_depth = sum(teeth_depths) / len(teeth_depths)
                variance = sum((x - mean_depth) ** 2 for x in teeth_depths) / len(teeth_depths)
                std_depth = variance ** 0.5
                case_means.append(mean_depth)
                case_stds.append(std_depth)
            else:
                case_means.append(0)
                case_stds.append(0)
        
        plt.errorbar(wear_cases, case_means, yerr=case_stds, fmt='ro-', linewidth=2, markersize=6, capsize=5, capthick=2)
        plt.xlabel('Wear Case')
        plt.ylabel('Mean Wear Depth (µm)')
        plt.title('Mean Wear Depth (Teeth 2-35) with Error Bars')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Wear Depth Variation and Deviations
        plt.subplot(2, 2, 4)
        plt.plot(wear_cases, case_stds, 'go-', linewidth=2, markersize=6, label='Standard Deviation')
        
        # Calculate coefficient of variation (CV = std/mean)
        cv_values = []
        for mean, std in zip(case_means, case_stds):
            if mean > 0:
                cv_values.append((std / mean) * 100)  # Convert to percentage
            else:
                cv_values.append(0)
        
        # Plot CV on secondary y-axis
        ax2 = plt.twinx()
        ax2.plot(wear_cases, cv_values, 'm^--', linewidth=2, markersize=6, label='Coefficient of Variation (%)')
        ax2.set_ylabel('Coefficient of Variation (%)', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        
        plt.xlabel('Wear Case')
        plt.ylabel('Standard Deviation (µm)', color='g')
        plt.title('Wear Depth Variation and Deviations')
        plt.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: All Teeth Analysis
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: All Teeth Wear Depth Distribution
        plt.subplot(2, 2, 1)
        all_teeth_depths = []
        all_teeth_cases = []
        for case in wear_cases:
            if case in all_teeth_data:
                for tooth_num, depth in all_teeth_data[case].items():
                    all_teeth_depths.append(depth)
                    all_teeth_cases.append(case)
        
        plt.scatter(all_teeth_cases, all_teeth_depths, alpha=0.6, s=20, color='blue')
        plt.plot(wear_cases, case_means, 'ro-', linewidth=3, markersize=8, label='Mean')
        plt.xlabel('Wear Case')
        plt.ylabel('Wear Depth (µm)')
        plt.title('All Teeth Wear Depth Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Error Bars by Case
        plt.subplot(2, 2, 2)
        plt.errorbar(wear_cases, case_means, yerr=case_stds, fmt='bo-', linewidth=2, markersize=6, capsize=5, capthick=2)
        plt.xlabel('Wear Case')
        plt.ylabel('Mean Wear Depth (µm)')
        plt.title('Mean Wear Depth with Error Bars')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Coefficient of Variation
        plt.subplot(2, 2, 3)
        plt.plot(wear_cases, cv_values, 'go-', linewidth=2, markersize=6)
        plt.xlabel('Wear Case')
        plt.ylabel('Coefficient of Variation (%)')
        plt.title('Relative Variability (CV) by Case')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Range Analysis
        plt.subplot(2, 2, 4)
        case_ranges = []
        for case in wear_cases:
            if case in all_teeth_data:
                teeth_depths = list(all_teeth_data[case].values())
                case_ranges.append(max(teeth_depths) - min(teeth_depths))
            else:
                case_ranges.append(0)
        
        plt.plot(wear_cases, case_ranges, 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Wear Case')
        plt.ylabel('Range (µm)')
        plt.title('Wear Depth Range by Case')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Results plotted successfully!")
        print("📊 Two figures generated:")
        print("   - Figure 1: Tooth 1 comparison with actual measurements")
        print("   - Figure 2: All teeth analysis with error bars and deviations")
        
    except ImportError:
        print("❌ matplotlib not available. Please install matplotlib to plot results.")
    except Exception as e:
        print(f"❌ Error plotting results: {str(e)}")

def show_integrated_results_table():
    """Show integrated results table combining single tooth and all teeth data"""
    try:
        single_tooth_file = os.path.join("gear_images", "single_tooth_results.csv")
        all_teeth_file = os.path.join("gear_images", "all_teeth_results.csv")
        
        if not os.path.exists(single_tooth_file):
            print("❌ Single tooth results file not found")
            return
        
        if not os.path.exists(all_teeth_file):
            print("❌ All teeth results file not found")
            return
        
        print("\n📊 INTEGRATED RESULTS TABLE")
        print("=" * 60)
        print("Combining single tooth (tooth 1) and all teeth (2-35) results")
        print("=" * 60)
        
        # Load single tooth data
        single_tooth_data = {}
        with open(single_tooth_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wear_case = int(row['wear_case'])
                wear_depth = float(row['wear_depth_um'])
                single_tooth_data[wear_case] = wear_depth
        
        # Load all teeth data
        all_teeth_data = {}
        with open(all_teeth_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wear_case = int(row['wear_case'])
                tooth_number = int(row['tooth_number'])
                wear_depth = float(row['wear_depth_um'])
                
                if wear_case not in all_teeth_data:
                    all_teeth_data[wear_case] = {}
                all_teeth_data[wear_case][tooth_number] = wear_depth
        
        # Display integrated table
        print(f"{'Wear Case':<10} {'Tooth 1':<10} {'Range (1-35)':<15} {'Mean':<12} {'Std Dev':<10}")
        print("-" * 60)
        
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
            
            print(f"{wear_case:<10} {tooth1_depth:<10.1f} {teeth_range:<15} {mean_depth:<12.1f} {std_dev:<10.1f}")
        
        # Summary statistics
        print(f"\n📊 Total wear cases analyzed: {len(single_tooth_data)}")
        print(f"📊 Total teeth analyzed per case: 35")
        
    except Exception as e:
        print(f"❌ Error showing integrated results: {str(e)}")

# Legacy functions for backward compatibility
def display_picture_results_menu(agent, results):
    """Legacy function - redirect to new menu"""
    print("⚠️ This function has been replaced. Use 'Show Results' from the main picture analysis menu.")
    display_show_results_menu(agent)

def display_wear_level_summary(agent, results):
    """Legacy function - redirect to new menu"""
    print("⚠️ This function has been replaced. Use 'Show Results' from the main picture analysis menu.")
    display_show_results_menu(agent)

def display_overall_assessment(results):
    """Legacy function - redirect to new menu"""
    print("⚠️ This function has been replaced. Use 'Show Results' from the main picture analysis menu.")
    print("Please use the new integrated gear wear analysis system.")

def display_database_analysis_menu(agent):
    """Legacy function - redirect to new menu"""
    print("⚠️ This function has been replaced. Use 'Database Process' from the main picture analysis menu.")
    display_database_process_menu(agent)
