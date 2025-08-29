"""
Vibration Analysis Submenu Module
Handles all vibration analysis related menu options and functionality
"""

import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the service layer
try:
    from vibration_service import VibrationAnalysisService
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from vibration_service import VibrationAnalysisService

def display_vibration_signal_menu(service: VibrationAnalysisService) -> None:
    """
    Display menu for plotting vibration signals.
    
    Args:
        service: The vibration analysis service instance
    """
    while True:
        print("\n📈 PLOT VIBRATION SIGNAL")
        print("=" * 35)
        print("1. High Speed Graphs (45 RPS)")
        print("2. Low Speed Graphs (15 RPS)")
        print("3. Return to Vibration Analysis Menu")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                logger.info("User selected: High Speed Graphs (45 RPS)")
                opened_count = service.open_multiple_plots("high_speed_45_rps", "High Speed Vibration Signal Plots")
                if opened_count > 0:
                    print(f"✅ Opened {opened_count} high speed vibration signal plots!")
                else:
                    print("❌ No high speed vibration signal plots found")
                
            elif choice == "2":
                logger.info("User selected: Low Speed Graphs (15 RPS)")
                opened_count = service.open_multiple_plots("low_speed_15_rps", "Low Speed Vibration Signal Plots")
                if opened_count > 0:
                    print(f"✅ Opened {opened_count} low speed vibration signal plots!")
                else:
                    print("❌ No low speed vibration signal plots found")
                
            elif choice == "3":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted vibration signal menu")
            break
        except Exception as e:
            logger.error("Error in vibration signal menu: %s", str(e))
            print(f"❌ Error: {str(e)}")

def display_vibration_features_menu(service: VibrationAnalysisService) -> None:
    """
    Display menu for plotting vibration features.
    
    Args:
        service: The vibration analysis service instance
    """
    while True:
        print("\n📊 PLOT VIBRATION FEATURES")
        print("=" * 35)
        print("1. RMS Features")
        print("2. FME Features")
        print("3. Return to Vibration Analysis Menu")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                logger.info("User selected: RMS Features")
                opened_count = service.open_multiple_plots("rms_features", "RMS Feature Plots")
                if opened_count > 0:
                    print("✅ RMS Feature plots opened successfully!")
                    print("📊 RMS (Root Mean Square) features show vibration amplitude levels")
                else:
                    print("❌ No RMS feature plots found")
                
            elif choice == "2":
                logger.info("User selected: FME Features")
                opened_count = service.open_multiple_plots("fme_features", "FME Feature Plots")
                if opened_count > 0:
                    print("✅ FME Feature plots opened successfully!")
                    print("📊 FME (Frequency Modulated Energy) features show frequency domain characteristics")
                else:
                    print("❌ No FME feature plots found")
                
            elif choice == "3":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted vibration features menu")
            break
        except Exception as e:
            logger.error("Error in vibration features menu: %s", str(e))
            print(f"❌ Error: {str(e)}")

def display_vibration_data_menu(service: VibrationAnalysisService) -> None:
    """
    Display menu for vibration data analysis.
    
    Args:
        service: The vibration analysis service instance
    """
    while True:
        print("\n📊 VIBRATION DATA ANALYSIS")
        print("=" * 35)
        print("1. Load High Speed Data")
        print("2. Load Low Speed Data")
        print("3. Load RMS Data")
        print("4. Load FME Data")
        print("5. Show Data Info")
        print("6. Return to Vibration Analysis Menu")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                logger.info("User selected: Load High Speed Data")
                data = service.load_vibration_data("high_speed")
                if data is not None:
                    print(f"✅ High speed data loaded: {len(data)} records")
                    print(f"   Columns: {list(data.columns)}")
                else:
                    print("❌ Failed to load high speed data")
                
            elif choice == "2":
                logger.info("User selected: Load Low Speed Data")
                data = service.load_vibration_data("low_speed")
                if data is not None:
                    print(f"✅ Low speed data loaded: {len(data)} records")
                    print(f"   Columns: {list(data.columns)}")
                else:
                    print("❌ Failed to load low speed data")
                
            elif choice == "3":
                logger.info("User selected: Load RMS Data")
                data_45 = service.load_vibration_data("rms_45")
                data_15 = service.load_vibration_data("rms_15")
                if data_45 is not None and data_15 is not None:
                    print(f"✅ RMS data loaded: 45 RPS ({len(data_45)} records), 15 RPS ({len(data_15)} records)")
                else:
                    print("❌ Failed to load RMS data")
                
            elif choice == "4":
                logger.info("User selected: Load FME Data")
                data = service.load_vibration_data("fme")
                if data is not None:
                    print(f"✅ FME data loaded: {len(data)} records")
                    print(f"   Columns: {list(data.columns)}")
                else:
                    print("❌ Failed to load FME data")
                
            elif choice == "5":
                logger.info("User selected: Show Data Info")
                info = service.get_vibration_data_info()
                if "error" not in info:
                    print("📊 Vibration Data Information:")
                    print(f"   Vibration Path: {info['vibration_path']}")
                    print(f"   Database Path: {info['database_path']}")
                    
                    print("\n📁 Data Files:")
                    for file, file_info in info["data_files"].items():
                        status = "✅" if file_info["exists"] else "❌"
                        size_mb = file_info["size"] / (1024 * 1024) if file_info["size"] > 0 else 0
                        print(f"   {status} {file} ({size_mb:.1f} MB)")
                    
                    print("\n🖼️ Plot Files:")
                    for plot_type, files in info["plot_files"].items():
                        existing = sum(1 for f in files.values() if f["exists"])
                        total = len(files)
                        print(f"   {plot_type}: {existing}/{total} files")
                else:
                    print(f"❌ Error getting data info: {info['error']}")
                
            elif choice == "6":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted vibration data menu")
            break
        except Exception as e:
            logger.error("Error in vibration data menu: %s", str(e))
            print(f"❌ Error: {str(e)}")

def display_vibration_analysis_menu(agent: Any) -> None:
    """
    Display main submenu for vibration analysis.
    
    Args:
        agent: The main diagnosis agent (for compatibility)
    """
    service = VibrationAnalysisService()
    
    while True:
        print("\n📊 VIBRATION ANALYSIS SUBMENU")
        print("=" * 35)
        print("1. Plot Vibration Signal")
        print("2. Plot Features")
        print("3. Data Analysis")
        print("4. Return to Main Menu")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                display_vibration_signal_menu(service)
                
            elif choice == "2":
                display_vibration_features_menu(service)
                
            elif choice == "3":
                display_vibration_data_menu(service)
                
            elif choice == "4":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-4.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted vibration analysis menu")
            break
        except Exception as e:
            logger.error("Error in vibration analysis menu: %s", str(e))
            print(f"❌ Error: {str(e)}")