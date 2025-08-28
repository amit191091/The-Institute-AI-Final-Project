"""
Picture Analysis Submenu Module
Handles all picture analysis related menu options and functionality
"""

import os
import sys
import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the service layer
try:
    from Picture_Tools.picture_service import PictureAnalysisService
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Picture Tools'))
    from picture_service import PictureAnalysisService

def display_picture_analysis_menu(agent: Any) -> None:
    """
    Display submenu for picture analysis.
    
    Args:
        agent: The agent object (for compatibility)
    """
    service = PictureAnalysisService()
    
    while True:
        print("\n🖼️ PICTURE ANALYSIS SUBMENU")
        print("=" * 35)
        print("1. Database Process")
        print("2. Show Results")
        print("3. Return to Main Menu")
        
        try:
            choice = input(f"\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                display_database_process_menu(service)
                    
            elif choice == "2":
                display_show_results_menu(service)
                    
            elif choice == "3":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted menu")
            break
        except Exception as e:
            logger.error("Error in picture analysis menu: %s", str(e))
            print(f"❌ Error: {str(e)}")

def display_database_process_menu(service: PictureAnalysisService) -> None:
    """
    Display menu for database processing.
    
    Args:
        service: The picture analysis service instance
    """
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
                logger.info("User selected: Run Tooth 1 Analysis")
                print("\n🔧 Running Tooth 1 Analysis...")
                result = service.run_tooth1_analysis()
                if result:
                    logger.info("Tooth 1 analysis completed successfully")
                    print("✅ Tooth 1 analysis completed successfully!")
                else:
                    logger.error("Tooth 1 analysis failed")
                    print("❌ Tooth 1 analysis failed!")
                    
            elif choice == "2":
                logger.info("User selected: Run Integrated Gear Wear Analysis")
                print("\n🔧 Running Integrated Gear Wear Analysis...")
                result = service.run_integrated_gear_wear_analysis()
                if result:
                    logger.info("Integrated gear wear analysis completed successfully")
                    print("✅ Integrated gear wear analysis completed successfully!")
                else:
                    logger.error("Integrated gear wear analysis failed")
                    print("❌ Integrated gear wear analysis failed!")
                    
            elif choice == "3":
                logger.info("User selected: Run Complete Analysis")
                print("\n🔧 Running Complete Analysis...")
                tooth1_success, integrated_success = service.run_complete_analysis()
                
                if tooth1_success and integrated_success:
                    logger.info("Complete analysis finished successfully")
                    print("✅ Complete analysis finished successfully!")
                elif tooth1_success:
                    logger.warning("Tooth 1 analysis succeeded but integrated analysis failed")
                    print("⚠️ Tooth 1 analysis completed, but integrated analysis failed!")
                else:
                    logger.error("Tooth 1 analysis failed")
                    print("❌ Tooth 1 analysis failed!")
                    
            elif choice == "4":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-4.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted database process menu")
            break
        except Exception as e:
            logger.error("Error in database process menu: %s", str(e))
            print(f"❌ Error: {str(e)}")

def display_show_results_menu(service: PictureAnalysisService) -> None:
    """
    Display menu for showing results.
    
    Args:
        service: The picture analysis service instance
    """
    while True:
        print("\n📊 SHOW RESULTS MENU")
        print("=" * 35)
        print("1. Show Integrated Results Table")
        print("2. Plot Results")
        print("3. Return to Picture Analysis Menu")
        
        try:
            choice = input(f"\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                logger.info("User selected: Show Integrated Results Table")
                success = service.show_integrated_results_table()
                if not success:
                    logger.error("Failed to show integrated results table")
                    print("❌ Failed to show integrated results table")
                    
            elif choice == "2":
                logger.info("User selected: Plot Results")
                success = service.plot_results()
                if success:
                    logger.info("Results plotting completed successfully")
                    print("✅ Results plotting completed successfully!")
                else:
                    logger.error("Results plotting failed")
                    print("❌ Results plotting failed!")
                    
            elif choice == "3":
                break
                
            else:
                logger.warning("Invalid choice entered: %s", choice)
                print("❌ Invalid choice. Please enter a number between 1-3.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted show results menu")
            break
        except Exception as e:
            logger.error("Error in show results menu: %s", str(e))
            print(f"❌ Error: {str(e)}")

# Legacy functions for backward compatibility (simplified)
def display_picture_results_menu(agent: Any, results: Any) -> None:
    """
    Legacy function - redirect to new menu.
    
    Args:
        agent: The agent object (for compatibility)
        results: Results object (for compatibility)
    """
    logger.warning("Legacy function display_picture_results_menu called - redirecting to new menu")
    print("⚠️ This function has been replaced. Use 'Show Results' from the main picture analysis menu.")
    service = PictureAnalysisService()
    display_show_results_menu(service)

def display_wear_level_summary(agent: Any, results: Any) -> None:
    """
    Legacy function - redirect to new menu.
    
    Args:
        agent: The agent object (for compatibility)
        results: Results object (for compatibility)
    """
    logger.warning("Legacy function display_wear_level_summary called - redirecting to new menu")
    print("⚠️ This function has been replaced. Use 'Show Results' from the main picture analysis menu.")
    service = PictureAnalysisService()
    display_show_results_menu(service)

def display_overall_assessment(results: Any) -> None:
    """
    Legacy function - redirect to new menu.
    
    Args:
        results: Results object (for compatibility)
    """
    logger.warning("Legacy function display_overall_assessment called - redirecting to new menu")
    print("⚠️ This function has been replaced. Use 'Show Results' from the main picture analysis menu.")
    print("Please use the new integrated gear wear analysis system.")

def display_database_analysis_menu(agent: Any) -> None:
    """
    Legacy function - redirect to new menu.
    
    Args:
        agent: The agent object (for compatibility)
    """
    logger.warning("Legacy function display_database_analysis_menu called - redirecting to new menu")
    print("⚠️ This function has been replaced. Use 'Database Process' from the main picture analysis menu.")
    service = PictureAnalysisService()
    display_database_process_menu(service)