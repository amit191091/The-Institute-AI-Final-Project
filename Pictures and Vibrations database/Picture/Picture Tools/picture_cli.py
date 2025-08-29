#!/usr/bin/env python3
"""
Picture Analysis CLI
===================

Command-line interface for picture analysis operations using Typer.
"""

import typer
import logging
from pathlib import Path
from typing import Optional
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from picture_service import PictureAnalysisService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Picture Analysis Tools - Gear wear analysis and visualization",
    no_args_is_help=True
)

def version_callback(value: bool):
    """Version callback for CLI."""
    if value:
        typer.echo("Picture Analysis Tools v1.0.0")
        raise typer.Exit()

@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)
):
    """Picture Analysis Tools - Advanced gear wear analysis and visualization."""
    pass

@app.command()
def analyze(
    tooth1: bool = typer.Option(False, "--tooth1", "-t", help="Run tooth 1 analysis only"),
    integrated: bool = typer.Option(False, "--integrated", "-i", help="Run integrated gear wear analysis only"),
    complete: bool = typer.Option(False, "--complete", "-c", help="Run complete analysis (tooth1 + integrated)"),
    tools_path: Optional[str] = typer.Option(None, "--tools-path", "-p", help="Path to Picture Tools directory")
):
    """
    Run picture analysis operations.
    
    Examples:
        picture-cli analyze --tooth1
        picture-cli analyze --integrated
        picture-cli analyze --complete
        picture-cli analyze --tooth1 --integrated
    """
    try:
        service = PictureAnalysisService(tools_path)
        
        if tooth1:
            typer.echo("🔧 Running Tooth 1 Analysis...")
            success = service.run_tooth1_analysis()
            if success:
                typer.echo("✅ Tooth 1 analysis completed successfully!")
            else:
                typer.echo("❌ Tooth 1 analysis failed!")
                raise typer.Exit(1)
        
        if integrated:
            typer.echo("🔧 Running Integrated Gear Wear Analysis...")
            success = service.run_integrated_gear_wear_analysis()
            if success:
                typer.echo("✅ Integrated gear wear analysis completed successfully!")
            else:
                typer.echo("❌ Integrated gear wear analysis failed!")
                raise typer.Exit(1)
        
        if complete:
            typer.echo("🔧 Running Complete Analysis...")
            tooth1_success, integrated_success = service.run_complete_analysis()
            
            if tooth1_success and integrated_success:
                typer.echo("✅ Complete analysis finished successfully!")
            elif tooth1_success:
                typer.echo("⚠️ Tooth 1 analysis completed, but integrated analysis failed!")
                raise typer.Exit(1)
            else:
                typer.echo("❌ Tooth 1 analysis failed!")
                raise typer.Exit(1)
        
        # If no specific analysis was requested, run complete analysis
        if not any([tooth1, integrated, complete]):
            typer.echo("🔧 Running Complete Analysis (default)...")
            tooth1_success, integrated_success = service.run_complete_analysis()
            
            if tooth1_success and integrated_success:
                typer.echo("✅ Complete analysis finished successfully!")
            else:
                typer.echo("❌ Analysis failed!")
                raise typer.Exit(1)
                
    except Exception as e:
        logger.error("Error in analysis: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def results(
    table: bool = typer.Option(False, "--table", "-t", help="Show integrated results table"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Plot results"),
    tools_path: Optional[str] = typer.Option(None, "--tools-path", help="Path to Picture Tools directory")
):
    """
    Display analysis results.
    
    Examples:
        picture-cli results --table
        picture-cli results --plot
        picture-cli results --table --plot
    """
    try:
        service = PictureAnalysisService(tools_path)
        
        if table:
            typer.echo("📊 Showing Integrated Results Table...")
            success = service.show_integrated_results_table()
            if not success:
                typer.echo("❌ Failed to show integrated results table")
                raise typer.Exit(1)
        
        if plot:
            typer.echo("📊 Plotting Results...")
            success = service.plot_results()
            if success:
                typer.echo("✅ Results plotting completed successfully!")
            else:
                typer.echo("❌ Results plotting failed!")
                raise typer.Exit(1)
        
        # If no specific display was requested, show both
        if not any([table, plot]):
            typer.echo("📊 Showing Results (default)...")
            
            # Show table
            typer.echo("📊 Showing Integrated Results Table...")
            table_success = service.show_integrated_results_table()
            
            # Show plot
            typer.echo("📊 Plotting Results...")
            plot_success = service.plot_results()
            
            if not table_success or not plot_success:
                typer.echo("❌ Some display operations failed!")
                raise typer.Exit(1)
                
    except Exception as e:
        logger.error("Error displaying results: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def status(
    tools_path: Optional[str] = typer.Option(None, "--tools-path", help="Path to Picture Tools directory")
):
    """
    Show the current status of analysis files and results.
    """
    try:
        service = PictureAnalysisService(tools_path)
        status_info = service.get_analysis_status()
        
        typer.echo("📋 Picture Analysis Status:")
        typer.echo(f"  📁 Tools Path: {status_info['picture_tools_path']}")
        
        # Scripts status
        typer.echo("\n🔧 Analysis Scripts:")
        for script, exists in status_info["scripts_exist"].items():
            status_icon = "✅" if exists else "❌"
            typer.echo(f"  {status_icon} {script}")
        
        # Results status
        typer.echo("\n📊 Results Files:")
        for result_file, exists in status_info["results_exist"].items():
            status_icon = "✅" if exists else "❌"
            typer.echo(f"  {status_icon} {result_file}")
        
        # Data files status
        typer.echo("\n📁 Data Files:")
        for data_file, exists in status_info["data_files_exist"].items():
            status_icon = "✅" if exists else "❌"
            typer.echo(f"  {status_icon} {data_file}")
            
    except Exception as e:
        logger.error("Error getting status: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def load_data(
    tools_path: Optional[str] = typer.Option(None, "--tools-path", help="Path to Picture Tools directory")
):
    """
    Load and display analysis data.
    """
    try:
        service = PictureAnalysisService(tools_path)
        
        typer.echo("📊 Loading Analysis Data...")
        single_tooth_data, all_teeth_data = service.load_analysis_data()
        
        if single_tooth_data is not None:
            typer.echo(f"✅ Single tooth data loaded: {len(single_tooth_data)} records")
        else:
            typer.echo("❌ Failed to load single tooth data")
        
        if all_teeth_data is not None:
            typer.echo(f"✅ All teeth data loaded: {len(all_teeth_data)} records")
        else:
            typer.echo("❌ Failed to load all teeth data")
            
        if single_tooth_data is None and all_teeth_data is None:
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error("Error loading data: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

def main() -> None:
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
