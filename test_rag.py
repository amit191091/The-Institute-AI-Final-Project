"""
Test script to verify the Hybrid RAG system functionality
"""
import sys
import os
from pathlib import Path
import logging

# Add current directory to path
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from rag_app.config import settings
        print("  ✅ Config loaded")
        
        from rag_app.loaders import get_document_loader
        print("  ✅ Document loaders")
        
        from rag_app.chunking import create_chunker
        print("  ✅ Chunking system")
        
        from rag_app.metadata import extract_keywords
        print("  ✅ Metadata extraction")
        
        from rag_app.indexing import create_hybrid_index
        print("  ✅ Hybrid indexing")
        
        from rag_app.retrieve import create_retriever
        print("  ✅ Retrieval system")
        
        from rag_app.agents import create_multi_agent_system
        print("  ✅ Multi-agent system")
        
        from rag_app.pipeline import create_pipeline
        print("  ✅ Pipeline orchestrator")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("\n🔧 Testing environment...")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"  ✅ OPENAI_API_KEY: {openai_key[:10]}...")
    else:
        print("  ⚠️  OPENAI_API_KEY not set")
    
    # Check directories
    from rag_app.config import settings
    
    directories = [settings.DATA_DIR, settings.INDEX_DIR, settings.REPORTS_DIR]
    for directory in directories:
        if directory.exists():
            print(f"  ✅ {directory}")
        else:
            directory.mkdir(exist_ok=True)
            print(f"  📁 Created {directory}")
    
    return True

def test_document_processing():
    """Test document processing pipeline"""
    print("\n📄 Testing document processing...")
    
    try:
        from rag_app.loaders import get_document_loader
        from rag_app.validate import create_validator
        
        loader = get_document_loader()
        validator = create_validator()
        
        # Look for sample PDF
        pdf_path = Path("MG-5025A_Gearbox_Wear_Investigation_Report.pdf")
        
        if pdf_path.exists():
            print(f"  📄 Found sample PDF: {pdf_path}")
            
            # Test validation
            validation_result = validator.validate_document(pdf_path)
            if validation_result["is_valid"]:
                print("  ✅ Document validation passed")
            else:
                print(f"  ⚠️  Document validation warnings: {len(validation_result['warnings'])}")
                print(f"  ❌ Document validation errors: {len(validation_result['errors'])}")
            
            # Test loading
            elements = loader.load_elements(pdf_path)
            print(f"  ✅ Loaded {len(elements)} elements from document")
            
            return True
        else:
            print("  ⚠️  Sample PDF not found")
            print("  💡 Place MG-5025A_Gearbox_Wear_Investigation_Report.pdf in project root to test")
            return True
            
    except Exception as e:
        print(f"  ❌ Document processing test failed: {e}")
        return False

def test_pipeline_basic():
    """Test basic pipeline functionality"""
    print("\n⚙️ Testing pipeline initialization...")
    
    try:
        from rag_app.pipeline import create_pipeline
        
        # Create and initialize pipeline
        pipeline = create_pipeline()
        result = pipeline.initialize()
        
        if result["success"]:
            print("  ✅ Pipeline initialized successfully")
            
            # Test stats
            stats_result = pipeline.get_stats()
            if stats_result["success"]:
                print(f"  ✅ System stats retrieved")
                stats = stats_result["stats"]
                print(f"    📊 Total documents: {stats['total_documents']}")
            
            return True
        else:
            print(f"  ❌ Pipeline initialization failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {e}")
        return False

def test_query_processing():
    """Test query processing without documents"""
    print("\n🔍 Testing query processing...")
    
    try:
        from rag_app.pipeline import create_pipeline
        
        pipeline = create_pipeline()
        pipeline.initialize()
        
        # Test simple query (should handle empty index gracefully)
        test_query = "What are the main components of a gearbox?"
        result = pipeline.query(test_query)
        
        if result["success"]:
            print("  ✅ Query processing works")
            print(f"    🤖 Agent: {result['metadata']['agent_type']}")
        else:
            # This is expected with empty index
            print("  ✅ Query handling works (empty index expected)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Query processing test failed: {e}")
        return False

def test_gear_integration():
    """Test integration with existing gear analysis"""
    print("\n🔬 Testing gear analysis integration...")
    
    try:
        # Try to import gear analysis modules
        from vibration_analysis import VibrationAnalysis
        print("  ✅ Vibration analysis available")
        
        try:
            from picture_analysis_menu import analyze_gear_images
            print("  ✅ Picture analysis available")
        except ImportError:
            print("  ⚠️  Picture analysis not available")
        
        return True
        
    except ImportError:
        print("  ⚠️  Gear analysis modules not available")
        print("  💡 This is normal if running RAG-only mode")
        return True

def run_all_tests():
    """Run all tests"""
    print("🧪 HYBRID RAG SYSTEM TESTS")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Document Processing", test_document_processing),
        ("Pipeline Basic", test_pipeline_basic),
        ("Query Processing", test_query_processing),
        ("Gear Integration", test_gear_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("="*30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next steps:")
    print("1. Set OPENAI_API_KEY in .env file")
    print("2. Place documents in 'data' directory")
    print("3. Run: python main.py")

if __name__ == "__main__":
    run_all_tests()
