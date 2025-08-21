"""
Vibration Analysis Submenu Module
Handles all vibration analysis related menu options and functionality
"""

import os
import platform
import subprocess
from typing import List, Dict, Optional

# Configuration for plot files
VIBRATION_PLOTS_CONFIG = {
    "high_speed_45_rps": [
        "from Healthy to W8 45 RPS.jpg",
        "from W9 to W17 45 RPS.jpg", 
        "from W18 to W26 45 RPS.jpg",
        "from W27 to W35 45 RPS.jpg"
    ],
    "low_speed_15_rps": [
        "from Healthy to W8 15 RPS.jpg",
        "from W9 to W17 15 RPS.jpg",
        "from W18 to W26 15 RPS.jpg", 
        "from W27 to W35 15 RPS.jpg"
    ],
    "rms_features": [
        "RMS level.jpg"
    ],
    "fme_features": [
        "FME level.jpg"
    ]
}

def open_image_file(file_path: str) -> bool:
    """
    Open image file with default system viewer
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if file was opened successfully, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        else:  # Linux
            subprocess.run(["xdg-open", file_path])
        
        print(f"✅ Opened: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        print(f"❌ Error opening file: {e}")
        return False

def get_plot_paths(plot_type: str) -> List[str]:
    """
    Get full paths for plot files based on type
    
    Args:
        plot_type: Type of plots ('high_speed_45_rps', 'low_speed_15_rps', 'rms_features', 'fme_features')
        
    Returns:
        List[str]: List of full file paths
    """
    base_dir = os.path.join("Pictures and Vibrations database", "Vibration")
    plot_files = VIBRATION_PLOTS_CONFIG.get(plot_type, [])
    return [os.path.join(base_dir, filename) for filename in plot_files]

def open_multiple_plots(plot_type: str, description: str) -> int:
    """
    Open multiple plot files of a specific type
    
    Args:
        plot_type: Type of plots to open
        description: Description for user feedback
        
    Returns:
        int: Number of plots successfully opened
    """
    print(f"🔄 Opening {description}...")
    
    plot_paths = get_plot_paths(plot_type)
    opened_count = 0
    
    for plot_path in plot_paths:
        if open_image_file(plot_path):
            opened_count += 1
    
    if opened_count > 0:
        print(f"✅ Opened {opened_count} {description.lower()}!")
    else:
        print(f"❌ No {description.lower()} found")
    
    return opened_count

def display_vibration_signal_menu():
    """Display menu for plotting vibration signals"""
    while True:
        print("\n📈 PLOT VIBRATION SIGNAL")
        print("=" * 35)
        print("1. High Speed Graphs (45 RPS)")
        print("2. Low Speed Graphs (15 RPS)")
        print("3. Return to Vibration Analysis Menu")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                open_multiple_plots("high_speed_45_rps", "High Speed Vibration Signal Plots")
                
            elif choice == "2":
                open_multiple_plots("low_speed_15_rps", "Low Speed Vibration Signal Plots")
                
            elif choice == "3":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_vibration_features_menu():
    """Display menu for plotting vibration features"""
    while True:
        print("\n📊 PLOT VIBRATION FEATURES")
        print("=" * 35)
        print("1. RMS Features")
        print("2. FME Features")
        print("3. Return to Vibration Analysis Menu")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                opened_count = open_multiple_plots("rms_features", "RMS Feature Plots")
                if opened_count > 0:
                    print("📊 RMS (Root Mean Square) features show vibration amplitude levels")
                
            elif choice == "2":
                opened_count = open_multiple_plots("fme_features", "FME Feature Plots")
                if opened_count > 0:
                    print("📊 FME (Frequency Modulated Energy) features show frequency domain characteristics")
                
            elif choice == "3":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_vibration_analysis_menu(agent):
    """
    Display main submenu for vibration analysis
    
    Args:
        agent: The main diagnosis agent (for compatibility)
    """
    while True:
        print("\n📊 VIBRATION ANALYSIS SUBMENU")
        print("=" * 35)
        print("1. Plot Vibration Signal")
        print("2. Plot Features")
        print("3. Return to Main Menu")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                display_vibration_signal_menu()
                
            elif choice == "2":
                display_vibration_features_menu()
                
            elif choice == "3":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")