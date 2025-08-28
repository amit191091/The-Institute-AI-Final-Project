#!/usr/bin/env python3
"""
Vibration Analysis Service Layer
===============================

Pure business logic functions for vibration analysis operations.
This service layer separates business logic from UI/menu code.
"""

import os
import platform
import subprocess
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibrationAnalysisService:
    """Service class for vibration analysis operations."""
    
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
    
    def __init__(self, vibration_path: Optional[str] = None):
        """
        Initialize the vibration analysis service.
        
        Args:
            vibration_path: Path to the Vibration directory
        """
        if vibration_path is None:
            # Default to current directory if running from Vibration
            self.vibration_path = Path.cwd()
        else:
            self.vibration_path = Path(vibration_path)
        
        if not self.vibration_path.exists():
            raise FileNotFoundError(f"Vibration directory not found: {self.vibration_path}")
        
        self.database_path = self.vibration_path / "database"
        if not self.database_path.exists():
            logger.warning("Database directory not found: %s", self.database_path)
    
    def open_image_file(self, file_path: str) -> bool:
        """
        Open image file with default system viewer.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if file was opened successfully, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return False
        
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            else:  # Linux
                subprocess.run(["xdg-open", file_path])
            
            logger.info("Opened: %s", os.path.basename(file_path))
            return True
        except Exception as e:
            logger.error("Error opening file: %s", e)
            return False
    
    def get_plot_paths(self, plot_type: str) -> List[str]:
        """
        Get full paths for plot files based on type.
        
        Args:
            plot_type: Type of plots ('high_speed_45_rps', 'low_speed_15_rps', 'rms_features', 'fme_features')
            
        Returns:
            List[str]: List of full file paths
        """
        plot_files = self.VIBRATION_PLOTS_CONFIG.get(plot_type, [])
        return [str(self.vibration_path / filename) for filename in plot_files]
    
    def open_multiple_plots(self, plot_type: str, description: str) -> int:
        """
        Open multiple plot files of a specific type.
        
        Args:
            plot_type: Type of plots to open
            description: Description for user feedback
            
        Returns:
            int: Number of plots successfully opened
        """
        logger.info("Opening %s...", description)
        
        plot_paths = self.get_plot_paths(plot_type)
        opened_count = 0
        
        for plot_path in plot_paths:
            if self.open_image_file(plot_path):
                opened_count += 1
        
        if opened_count > 0:
            logger.info("Opened %d %s", opened_count, description.lower())
        else:
            logger.warning("No %s found", description.lower())
        
        return opened_count
    
    def load_vibration_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """
        Load vibration data from CSV files.
        
        Args:
            data_type: Type of data to load ('high_speed', 'low_speed', 'rms_45', 'rms_15', 'fme')
            
        Returns:
            Optional[pd.DataFrame]: Loaded data or None if failed
        """
        try:
            if not self.database_path.exists():
                logger.error("Database directory not found")
                return None
            
            data_files = {
                'high_speed': 'Vibration signals high speed.csv',
                'low_speed': 'Vibration signals low speed.csv',
                'high_speed_spectrum': 'Vibration signals high speed spectrum.csv',
                'low_speed_spectrum': 'Vibration signals low speed spectrum.csv',
                'rms_45': 'RMS45.csv',
                'rms_15': 'RMS15.csv',
                'fme': 'FME Values.csv',
                'records': 'Records.csv',
                'record_examples': 'Record examples.csv'
            }
            
            if data_type not in data_files:
                logger.error("Unknown data type: %s", data_type)
                return None
            
            file_path = self.database_path / data_files[data_type]
            if not file_path.exists():
                logger.error("Data file not found: %s", file_path)
                return None
            
            logger.info("Loading %s data from %s", data_type, file_path.name)
            data = pd.read_csv(file_path)
            logger.info("Loaded %d records from %s", len(data), file_path.name)
            
            return data
            
        except Exception as e:
            logger.error("Error loading %s data: %s", data_type, str(e))
            return None
    
    def get_vibration_data_info(self) -> Dict[str, Any]:
        """
        Get information about available vibration data files.
        
        Returns:
            Dict[str, Any]: Information about data files
        """
        try:
            info = {
                "vibration_path": str(self.vibration_path),
                "database_path": str(self.database_path),
                "data_files": {},
                "plot_files": {}
            }
            
            # Check data files
            if self.database_path.exists():
                data_files = [
                    'Vibration signals high speed.csv',
                    'Vibration signals low speed.csv',
                    'Vibration signals high speed spectrum.csv',
                    'Vibration signals low speed spectrum.csv',
                    'RMS45.csv',
                    'RMS15.csv',
                    'FME Values.csv',
                    'Records.csv',
                    'Record examples.csv'
                ]
                
                for file in data_files:
                    file_path = self.database_path / file
                    info["data_files"][file] = {
                        "exists": file_path.exists(),
                        "size": file_path.stat().st_size if file_path.exists() else 0
                    }
            
            # Check plot files
            for plot_type, files in self.VIBRATION_PLOTS_CONFIG.items():
                info["plot_files"][plot_type] = {}
                for file in files:
                    file_path = self.vibration_path / file
                    info["plot_files"][plot_type][file] = {
                        "exists": file_path.exists(),
                        "size": file_path.stat().st_size if file_path.exists() else 0
                    }
            
            return info
            
        except Exception as e:
            logger.error("Error getting vibration data info: %s", str(e))
            return {"error": str(e)}
    
    def analyze_vibration_data(self, data_type: str) -> Dict[str, Any]:
        """
        Perform basic analysis on vibration data.
        
        Args:
            data_type: Type of data to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            data = self.load_vibration_data(data_type)
            if data is None:
                return {"error": f"Failed to load {data_type} data"}
            
            analysis = {
                "data_type": data_type,
                "records": len(data),
                "columns": list(data.columns),
                "memory_usage": data.memory_usage(deep=True).sum(),
                "dtypes": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict()
            }
            
            # Add basic statistics for numeric columns
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                analysis["numeric_stats"] = data[numeric_cols].describe().to_dict()
            
            logger.info("Analysis completed for %s: %d records", data_type, len(data))
            return analysis
            
        except Exception as e:
            logger.error("Error analyzing %s data: %s", data_type, str(e))
            return {"error": str(e)}
    
    def get_plot_status(self) -> Dict[str, Any]:
        """
        Get status of plot files.
        
        Returns:
            Dict[str, Any]: Status of plot files
        """
        try:
            status = {
                "vibration_path": str(self.vibration_path),
                "plot_types": {}
            }
            
            for plot_type, files in self.VIBRATION_PLOTS_CONFIG.items():
                status["plot_types"][plot_type] = {
                    "total_files": len(files),
                    "existing_files": 0,
                    "missing_files": 0,
                    "files": {}
                }
                
                for file in files:
                    file_path = self.vibration_path / file
                    exists = file_path.exists()
                    status["plot_types"][plot_type]["files"][file] = exists
                    
                    if exists:
                        status["plot_types"][plot_type]["existing_files"] += 1
                    else:
                        status["plot_types"][plot_type]["missing_files"] += 1
            
            return status
            
        except Exception as e:
            logger.error("Error getting plot status: %s", str(e))
            return {"error": str(e)}


# Convenience functions for backward compatibility
def open_image_file(file_path: str) -> bool:
    """Convenience function to open image file."""
    service = VibrationAnalysisService()
    return service.open_image_file(file_path)

def get_plot_paths(plot_type: str) -> List[str]:
    """Convenience function to get plot paths."""
    service = VibrationAnalysisService()
    return service.get_plot_paths(plot_type)

def open_multiple_plots(plot_type: str, description: str) -> int:
    """Convenience function to open multiple plots."""
    service = VibrationAnalysisService()
    return service.open_multiple_plots(plot_type, description)
