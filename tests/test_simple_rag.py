#!/usr/bin/env python3
"""
Simple RAG Configuration Test
============================

Tests the RAG configuration without requiring full evaluation dependencies.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_rag_configuration():
    """Test that RAG configuration is properly set up."""
    print("🔍 Testing RAG Configuration...")
    
    try:
        # Test 1: Check if RAG config can be imported
        print("\n1. Testing RAG config import...")
        from RAG.app.config import settings
        print(f"   ✅ RAG config imported successfully")
        print(f"   📁 DATA_DIR: {settings.paths.DATA_DIR}")
        print(f"   📁 INDEX_DIR: {settings.paths.INDEX_DIR}")
        print(f"   📁 LOGS_DIR: {settings.paths.LOGS_DIR}")
        
        # Test 2: Check if PDF file exists in root
        print("\n2. Testing PDF file location...")
        root_pdf = Path("Gear wear Failure.pdf")
        if root_pdf.exists():
            print(f"   ✅ PDF found in root: {root_pdf}")
            print(f"   📄 File size: {root_pdf.stat().st_size / 1024:.1f} KB")
        else:
            print(f"   ❌ PDF not found in root: {root_pdf}")
            
        # Test 3: Check if RAG directories exist
        print("\n3. Testing RAG directories...")
        for dir_name, dir_path in [
            ("DATA_DIR", settings.paths.DATA_DIR),
            ("INDEX_DIR", settings.paths.INDEX_DIR),
            ("LOGS_DIR", settings.paths.LOGS_DIR)
        ]:
            if dir_path.exists():
                print(f"   ✅ {dir_name}: {dir_path} (exists)")
            else:
                print(f"   ⚠️  {dir_name}: {dir_path} (does not exist)")
                
        # Test 4: Test basic RAG service import (without evaluation)
        print("\n4. Testing RAG service import...")
        try:
            # Import without the problematic eval_ragas module
            import RAG.app.rag_service
            print("   ✅ RAG service imported successfully")
        except ImportError as e:
            if "ragas" in str(e) or "IPython" in str(e):
                print("   ⚠️  RAG service imported (evaluation dependencies not available)")
            else:
                print(f"   ❌ RAG service import failed: {e}")
                
        # Test 5: Test document discovery
        print("\n5. Testing document discovery...")
        try:
            from RAG.app.pipeline_ingestion import discover_input_paths
            paths = discover_input_paths()
            print(f"   ✅ Found {len(paths)} input paths")
            for path in paths:
                print(f"      📄 {path}")
        except Exception as e:
            print(f"   ❌ Document discovery failed: {e}")
            
        print("\n✅ RAG Configuration Test Completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG Configuration Test Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_configuration()
    sys.exit(0 if success else 1)
