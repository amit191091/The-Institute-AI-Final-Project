#!/usr/bin/env python3
"""
Vibration Analysis CLI
=====================

Command-line interface for vibration analysis operations using Typer.
"""

import typer
import logging
from pathlib import Path
from typing import Optional
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vibration_service import VibrationAnalysisService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Vibration Analysis Tools - Signal analysis and feature extraction",
    no_args_is_help=True
)

def version_callback(value: bool):
    """Version callback for CLI."""
    if value:
        typer.echo("Vibration Analysis Tools v1.0.0")
        raise typer.Exit()

@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)
):
    """Vibration Analysis Tools - Advanced signal analysis and feature extraction."""
    pass

@app.command()
def plots(
    high_speed: bool = typer.Option(False, "--high-speed", "-h", help="Open high speed plots (45 RPS)"),
    low_speed: bool = typer.Option(False, "--low-speed", "-l", help="Open low speed plots (15 RPS)"),
    rms: bool = typer.Option(False, "--rms", "-r", help="Open RMS feature plots"),
    fme: bool = typer.Option(False, "--fme", "-f", help="Open FME feature plots"),
    all_plots: bool = typer.Option(False, "--all", "-a", help="Open all plot types"),
    vibration_path: Optional[str] = typer.Option(None, "--vibration-path", "-p", help="Path to Vibration directory")
):
    """
    Open vibration plots and visualizations.
    
    Examples:
        vibration-cli plots --high-speed
        vibration-cli plots --low-speed --rms
        vibration-cli plots --all
    """
    try:
        service = VibrationAnalysisService(vibration_path)
        
        if high_speed:
            typer.echo("📈 Opening High Speed Plots (45 RPS)...")
            opened_count = service.open_multiple_plots("high_speed_45_rps", "High Speed Vibration Signal Plots")
            if opened_count > 0:
                typer.echo(f"✅ Opened {opened_count} high speed plots!")
            else:
                typer.echo("❌ No high speed plots found")
        
        if low_speed:
            typer.echo("📈 Opening Low Speed Plots (15 RPS)...")
            opened_count = service.open_multiple_plots("low_speed_15_rps", "Low Speed Vibration Signal Plots")
            if opened_count > 0:
                typer.echo(f"✅ Opened {opened_count} low speed plots!")
            else:
                typer.echo("❌ No low speed plots found")
        
        if rms:
            typer.echo("📊 Opening RMS Feature Plots...")
            opened_count = service.open_multiple_plots("rms_features", "RMS Feature Plots")
            if opened_count > 0:
                typer.echo(f"✅ Opened {opened_count} RMS feature plots!")
                typer.echo("📊 RMS (Root Mean Square) features show vibration amplitude levels")
            else:
                typer.echo("❌ No RMS feature plots found")
        
        if fme:
            typer.echo("📊 Opening FME Feature Plots...")
            opened_count = service.open_multiple_plots("fme_features", "FME Feature Plots")
            if opened_count > 0:
                typer.echo(f"✅ Opened {opened_count} FME feature plots!")
                typer.echo("📊 FME (Frequency Modulated Energy) features show frequency domain characteristics")
            else:
                typer.echo("❌ No FME feature plots found")
        
        if all_plots:
            typer.echo("📈 Opening All Plot Types...")
            plot_types = ["high_speed_45_rps", "low_speed_15_rps", "rms_features", "fme_features"]
            descriptions = ["High Speed Plots", "Low Speed Plots", "RMS Features", "FME Features"]
            
            total_opened = 0
            for plot_type, description in zip(plot_types, descriptions):
                opened_count = service.open_multiple_plots(plot_type, description)
                total_opened += opened_count
            
            typer.echo(f"✅ Opened {total_opened} plots total!")
        
        # If no specific plot type was requested, show status
        if not any([high_speed, low_speed, rms, fme, all_plots]):
            typer.echo("📊 Plot Status:")
            status = service.get_plot_status()
            if "error" not in status:
                for plot_type, info in status["plot_types"].items():
                    typer.echo(f"  {plot_type}: {info['existing_files']}/{info['total_files']} files")
            else:
                typer.echo(f"❌ Error getting plot status: {status['error']}")
                
    except Exception as e:
        logger.error("Error opening plots: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def data(
    load: Optional[str] = typer.Option(None, "--load", "-l", help="Load specific data type (high_speed, low_speed, rms_45, rms_15, fme)"),
    analyze: Optional[str] = typer.Option(None, "--analyze", "-a", help="Analyze specific data type"),
    info: bool = typer.Option(False, "--info", "-i", help="Show data information"),
    vibration_path: Optional[str] = typer.Option(None, "--vibration-path", "-p", help="Path to Vibration directory")
):
    """
    Work with vibration data files.
    
    Examples:
        vibration-cli data --load high_speed
        vibration-cli data --analyze rms_45
        vibration-cli data --info
    """
    try:
        service = VibrationAnalysisService(vibration_path)
        
        if load:
            typer.echo(f"📊 Loading {load} data...")
            data = service.load_vibration_data(load)
            if data is not None:
                typer.echo(f"✅ {load} data loaded: {len(data)} records")
                typer.echo(f"   Columns: {list(data.columns)}")
                typer.echo(f"   Memory usage: {data.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
            else:
                typer.echo(f"❌ Failed to load {load} data")
                raise typer.Exit(1)
        
        if analyze:
            typer.echo(f"📊 Analyzing {analyze} data...")
            analysis = service.analyze_vibration_data(analyze)
            if "error" not in analysis:
                typer.echo(f"✅ Analysis completed for {analyze}:")
                typer.echo(f"   Records: {analysis['records']}")
                typer.echo(f"   Columns: {len(analysis['columns'])}")
                typer.echo(f"   Memory usage: {analysis['memory_usage'] / (1024*1024):.1f} MB")
                typer.echo(f"   Missing values: {sum(analysis['missing_values'].values())}")
            else:
                typer.echo(f"❌ Analysis failed: {analysis['error']}")
                raise typer.Exit(1)
        
        if info:
            typer.echo("📊 Vibration Data Information:")
            info_data = service.get_vibration_data_info()
            if "error" not in info_data:
                typer.echo(f"  Vibration Path: {info_data['vibration_path']}")
                typer.echo(f"  Database Path: {info_data['database_path']}")
                
                typer.echo("\n📁 Data Files:")
                for file, file_info in info_data["data_files"].items():
                    status = "✅" if file_info["exists"] else "❌"
                    size_mb = file_info["size"] / (1024 * 1024) if file_info["size"] > 0 else 0
                    typer.echo(f"  {status} {file} ({size_mb:.1f} MB)")
                
                typer.echo("\n🖼️ Plot Files:")
                for plot_type, files in info_data["plot_files"].items():
                    existing = sum(1 for f in files.values() if f["exists"])
                    total = len(files)
                    typer.echo(f"  {plot_type}: {existing}/{total} files")
            else:
                typer.echo(f"❌ Error getting data info: {info_data['error']}")
                raise typer.Exit(1)
        
        # If no specific action was requested, show available data types
        if not any([load, analyze, info]):
            typer.echo("📊 Available Data Types:")
            data_types = [
                "high_speed", "low_speed", "high_speed_spectrum", "low_speed_spectrum",
                "rms_45", "rms_15", "fme", "records", "record_examples"
            ]
            for data_type in data_types:
                typer.echo(f"  - {data_type}")
            typer.echo("\nUse --load <type> to load data or --analyze <type> to analyze data")
            
    except Exception as e:
        logger.error("Error working with data: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command()
def status(
    vibration_path: Optional[str] = typer.Option(None, "--vibration-path", help="Path to Vibration directory")
):
    """
    Show the current status of vibration files and data.
    """
    try:
        service = VibrationAnalysisService(vibration_path)
        
        # Get plot status
        plot_status = service.get_plot_status()
        if "error" in plot_status:
            typer.echo(f"❌ Error getting plot status: {plot_status['error']}")
            raise typer.Exit(1)
        
        typer.echo("📋 Vibration Analysis Status:")
        typer.echo(f"  📁 Vibration Path: {plot_status['vibration_path']}")
        
        # Plot files status
        typer.echo("\n🖼️ Plot Files:")
        for plot_type, info in plot_status["plot_types"].items():
            status_icon = "✅" if info["existing_files"] == info["total_files"] else "⚠️"
            typer.echo(f"  {status_icon} {plot_type}: {info['existing_files']}/{info['total_files']} files")
        
        # Data files status
        data_info = service.get_vibration_data_info()
        if "error" not in data_info:
            typer.echo("\n📁 Data Files:")
            for file, file_info in data_info["data_files"].items():
                status_icon = "✅" if file_info["exists"] else "❌"
                size_mb = file_info["size"] / (1024 * 1024) if file_info["size"] > 0 else 0
                typer.echo(f"  {status_icon} {file} ({size_mb:.1f} MB)")
        else:
            typer.echo(f"\n❌ Error getting data status: {data_info['error']}")
            
    except Exception as e:
        logger.error("Error getting status: %s", str(e))
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

def main() -> None:
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
