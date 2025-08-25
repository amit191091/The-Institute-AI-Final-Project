#!/usr/bin/env python3
"""
Run RAG System with OpenAI Only
==============================

This script sets up the environment and runs the RAG system using only OpenAI.
"""

import os
import sys
from pathlib import Path

def setup_openai_environment():
    """Set up environment for OpenAI only."""
    print("🔧 Setting up OpenAI-only environment...")
    
    # Read OpenAI API key
    parent_dir = Path("C:/Users/amitl/Documents/AI Developers/PracticeBase")
    openai_key_file = parent_dir / "OpenAI API key.txt"
    
    try:
        with open(openai_key_file, 'r') as f:
            openai_key = f.read().strip()
        
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["FORCE_OPENAI_ONLY"] = "true"
        
        # Clear any existing Google API key
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        print("✅ Environment set up successfully")
        print("✅ OpenAI API key loaded")
        print("✅ Force OpenAI only mode enabled")
        print("✅ Google API key cleared")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        return False

def run_main():
    """Run the main RAG system."""
    print("\n🚀 Starting RAG system...")
    
    # Add RAG folder to Python path
    rag_path = Path(__file__).parent / "RAG"
    sys.path.insert(0, str(rag_path))
    
    # Import and run main
    try:
        from Main import main
        main()
    except Exception as e:
        print(f"❌ Error running Main.py: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if setup_openai_environment():
        run_main()
    else:
        print("❌ Failed to set up environment")
        sys.exit(1)
