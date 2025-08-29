#!/usr/bin/env python3
"""
Picture Analysis Service Layer
==============================

Pure business logic functions for picture analysis operations.
This service layer separates business logic from UI/menu code.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PictureAnalysisService:
    """Service class for picture analysis operations."""
    
    def __init__(self, picture_tools_path: Optional[str] = None):
        """
        Initialize the picture analysis service.
        
        Args:
            picture_tools_path: Path to the Picture Tools directory
        """
        if picture_tools_path is None:
            # Default to current directory if running from Picture Tools
            self.picture_tools_path = Path.cwd()
        else:
            self.picture_tools_path = Path(picture_tools_path)
        
        if not self.picture_tools_path.exists():
            raise FileNotFoundError(f"Picture Tools directory not found: {self.picture_tools_path}")
    
    def run_tooth1_analysis(self) -> bool:
        """
        Run tooth 1 wear depth analysis.
        
        Returns:
            bool: True if analysis completed successfully, False otherwise
        """
        try:
            logger.info("Starting tooth 1 analysis...")
            
            script_path = self.picture_tools_path / "Analyze_tooth1.py"
            if not script_path.exists():
                logger.error(f"Script not found: {script_path}")
                return False
            
            logger.info(f"Executing: {script_path.name}")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.picture_tools_path),
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env
            )
            
            # Log output
            if result.stdout:
                logger.info("Analysis output:\n%s", result.stdout)
            if result.stderr:
                logger.warning("Analysis warnings/errors:\n%s", result.stderr)
            
            if result.returncode == 0:
                logger.info("Tooth 1 analysis completed successfully")
                return True
            else:
                logger.error("Tooth 1 analysis failed with return code: %d", result.returncode)
                return False
                
        except Exception as e:
            logger.error("Error running tooth 1 analysis: %s", str(e))
            return False
    
    def run_integrated_gear_wear_analysis(self) -> bool:
        """
        Run integrated gear wear analysis.
        
        Returns:
            bool: True if analysis completed successfully, False otherwise
        """
        try:
            logger.info("Starting integrated gear wear analysis...")
            
            script_path = self.picture_tools_path / "Analyze_all_teeth.py"
            if not script_path.exists():
                logger.error(f"Script not found: {script_path}")
                return False
            
            logger.info(f"Executing: {script_path.name}")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.picture_tools_path),
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env
            )
            
            # Log output
            if result.stdout:
                logger.info("Analysis output:\n%s", result.stdout)
            if result.stderr:
                logger.warning("Analysis warnings/errors:\n%s", result.stderr)
            
            if result.returncode == 0:
                logger.info("Integrated gear wear analysis completed successfully")
                return True
            else:
                logger.error("Integrated gear wear analysis failed with return code: %d", result.returncode)
                return False
                
        except Exception as e:
            logger.error("Error running integrated gear wear analysis: %s", str(e))
            return False
    
    def run_complete_analysis(self) -> Tuple[bool, bool]:
        """
        Run complete analysis (tooth 1 + integrated gear wear).
        
        Returns:
            Tuple[bool, bool]: (tooth1_success, integrated_success)
        """
        logger.info("Starting complete analysis...")
        
        # Step 1: Tooth 1 Analysis
        logger.info("Step 1: Tooth 1 Analysis...")
        tooth1_success = self.run_tooth1_analysis()
        
        if not tooth1_success:
            logger.error("Tooth 1 analysis failed, stopping complete analysis")
            return False, False
        
        # Step 2: Integrated Gear Wear Analysis
        logger.info("Step 2: Integrated Gear Wear Analysis...")
        integrated_success = self.run_integrated_gear_wear_analysis()
        
        if integrated_success:
            logger.info("Complete analysis finished successfully")
        else:
            logger.error("Integrated analysis failed")
        
        return tooth1_success, integrated_success
    
    def load_analysis_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load analysis data from CSV files.
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
                (single_tooth_data, all_teeth_data)
        """
        try:
            # Add Picture Tools directory to path for imports
            sys.path.insert(0, str(self.picture_tools_path))
            
            from data_loader import load_analysis_data as load_data
            
            single_tooth_data, all_teeth_data = load_data()
            
            logger.info("Analysis data loaded successfully")
            return single_tooth_data, all_teeth_data
            
        except Exception as e:
            logger.error("Error loading analysis data: %s", str(e))
            return None, None
    
    def plot_results(self) -> bool:
        """
        Plot results from CSV files.
        
        Returns:
            bool: True if plotting completed successfully, False otherwise
        """
        try:
            logger.info("Starting results plotting...")
            
            # Load data
            single_tooth_data, all_teeth_data = self.load_analysis_data()
            
            if not single_tooth_data or not all_teeth_data:
                logger.error("No data available for plotting")
                return False
            
            # Change to Picture directory and run the plot_results.py script
            original_dir = Path.cwd()
            picture_dir = original_dir / "Pictures and Vibrations database" / "Picture"
            
            if not picture_dir.exists():
                logger.error(f"Picture directory not found: {picture_dir}")
                return False
            
            os.chdir(picture_dir)
            
            # Run the plot_results.py script
            result = subprocess.run(
                [sys.executable, "plot_results.py"],
                capture_output=True,
                text=True
            )
            
            # Return to original directory
            os.chdir(original_dir)
            
            # Log output
            if result.stdout:
                logger.info("Plotting output:\n%s", result.stdout)
            if result.stderr:
                logger.warning("Plotting warnings/errors:\n%s", result.stderr)
            
            if result.returncode == 0:
                logger.info("Results plotting completed successfully")
                return True
            else:
                logger.error("Results plotting failed")
                return False
                
        except Exception as e:
            logger.error("Error plotting results: %s", str(e))
            return False
    
    def show_integrated_results_table(self) -> bool:
        """
        Show integrated results table combining single tooth and all teeth data.
        
        Returns:
            bool: True if display completed successfully, False otherwise
        """
        try:
            logger.info("Loading data for integrated results table...")
            
            # Load data
            single_tooth_data, all_teeth_data = self.load_analysis_data()
            
            if not single_tooth_data:
                logger.error("No single tooth data to display")
                return False
            
            # Add Picture Tools directory to path for imports
            sys.path.insert(0, str(self.picture_tools_path))
            
            from picture_results_display import (
                display_integrated_results_table, 
                display_summary_statistics
            )
            
            # Display results
            display_integrated_results_table(single_tooth_data, all_teeth_data)
            display_summary_statistics(single_tooth_data, all_teeth_data)
            
            logger.info("Integrated results table displayed successfully")
            return True
            
        except Exception as e:
            logger.error("Error showing integrated results: %s", str(e))
            return False
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get the current status of analysis files and results.
        
        Returns:
            Dict[str, Any]: Status information about analysis files
        """
        try:
            status = {
                "picture_tools_path": str(self.picture_tools_path),
                "scripts_exist": {},
                "results_exist": {},
                "data_files_exist": {}
            }
            
            # Check if analysis scripts exist
            scripts = ["Analyze_tooth1.py", "Analyze_all_teeth.py"]
            for script in scripts:
                status["scripts_exist"][script] = (self.picture_tools_path / script).exists()
            
            # Check if result files exist (they're saved in the parent Picture directory)
            result_files = ["single_tooth_results.csv", "all_teeth_results.csv"]
            for result_file in result_files:
                result_path = self.picture_tools_path.parent / result_file
                status["results_exist"][result_file] = result_path.exists()
            
            # Check if data files exist
            data_files = ["ground_truth_measurements.csv"]
            for data_file in data_files:
                status["data_files_exist"][data_file] = (self.picture_tools_path / data_file).exists()
            
            return status
            
        except Exception as e:
            logger.error("Error getting analysis status: %s", str(e))
            return {"error": str(e)}


# Convenience functions for backward compatibility
def run_tooth1_analysis() -> bool:
    """Convenience function to run tooth 1 analysis."""
    service = PictureAnalysisService()
    return service.run_tooth1_analysis()

def run_integrated_gear_wear_analysis() -> bool:
    """Convenience function to run integrated gear wear analysis."""
    service = PictureAnalysisService()
    return service.run_integrated_gear_wear_analysis()

def run_complete_analysis() -> Tuple[bool, bool]:
    """Convenience function to run complete analysis."""
    service = PictureAnalysisService()
    return service.run_complete_analysis()
