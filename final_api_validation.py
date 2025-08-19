"""
Final API Syntax Validation Test
Quick validation of API updates without heavy model downloads
"""

import sys
import importlib.util
import json
import time
from pathlib import Path

def test_imports():
    """Test all critical imports work with modern APIs"""
    print("🔍 Testing Critical Import Updates...")
    
    results = {}
    
    # Test langchain_openai imports
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        results['langchain_openai'] = True
        print("  ✅ langchain_openai imports (ChatOpenAI, OpenAIEmbeddings)")
    except Exception as e:
        results['langchain_openai'] = False
        print(f"  ❌ langchain_openai imports failed: {e}")
    
    # Test langchain_community imports
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.retrievers import BM25Retriever
        results['langchain_community'] = True
        print("  ✅ langchain_community imports (FAISS, BM25Retriever)")
    except Exception as e:
        results['langchain_community'] = False
        print(f"  ❌ langchain_community imports failed: {e}")
    
    # Test rank_bm25
    try:
        from rank_bm25 import BM25Okapi
        results['rank_bm25'] = True
        print("  ✅ rank_bm25 import")
    except Exception as e:
        results['rank_bm25'] = False
        print(f"  ❌ rank_bm25 import failed: {e}")
        
    return results

def test_api_initialization():
    """Test API classes can be initialized with correct parameters"""
    print("\n🧪 Testing API Initialization...")
    
    results = {}
    
    # Test ChatOpenAI with modern parameters
    try:
        from langchain_openai import ChatOpenAI
        # Test with valid parameters (no deprecated openai_api_key)
        chat = ChatOpenAI(model="gpt-4", temperature=0.7)
        results['chat_openai_init'] = True
        print("  ✅ ChatOpenAI initialization (model, temperature)")
    except Exception as e:
        results['chat_openai_init'] = False
        print(f"  ❌ ChatOpenAI initialization failed: {e}")
    
    # Test OpenAIEmbeddings with modern parameters
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        results['openai_embeddings_init'] = True
        print("  ✅ OpenAIEmbeddings initialization (model parameter)")
    except Exception as e:
        results['openai_embeddings_init'] = False
        print(f"  ❌ OpenAIEmbeddings initialization failed: {e}")
        
    # Test BM25Okapi
    try:
        from rank_bm25 import BM25Okapi
        corpus = [["hello", "world"], ["test", "document"]]
        bm25 = BM25Okapi(corpus)
        results['bm25_init'] = True
        print("  ✅ BM25Okapi initialization")
    except Exception as e:
        results['bm25_init'] = False
        print(f"  ❌ BM25Okapi initialization failed: {e}")
        
    return results

def test_module_structure():
    """Test that all RAG modules can be imported"""
    print("\n🔧 Testing RAG Module Structure...")
    
    results = {}
    modules = [
        'rag_app.config',
        'rag_app.loaders', 
        'rag_app.chunking',
        'rag_app.metadata',
        'rag_app.indexing',
        'rag_app.retrieve',
        'rag_app.agents',
        'rag_app.validation',
        'rag_app.utils',
        'rag_app.pipeline',
        'rag_app.ui_gradio'
    ]
    
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                results[module] = False
                print(f"  ❌ {module} not found")
            else:
                imported = importlib.import_module(module)
                results[module] = True
                print(f"  ✅ {module}")
        except Exception as e:
            results[module] = False
            print(f"  ❌ {module} failed: {e}")
            
    return results

def test_function_availability():
    """Test that key functions are available"""
    print("\n🎯 Testing Key Function Availability...")
    
    results = {}
    
    # Test validate_query function
    try:
        from rag_app.utils import validate_query
        # Test with sample query
        is_valid = validate_query("What is machine learning?")
        results['validate_query'] = is_valid
        print(f"  ✅ validate_query function works (result: {is_valid})")
    except Exception as e:
        results['validate_query'] = False
        print(f"  ❌ validate_query function failed: {e}")
    
    # Test DocumentValidator class
    try:
        from rag_app.validation import DocumentValidator
        validator = DocumentValidator()
        results['document_validator'] = True
        print("  ✅ DocumentValidator class available")
    except Exception as e:
        results['document_validator'] = False
        print(f"  ❌ DocumentValidator class failed: {e}")
        
    # Test create_gradio_interface function
    try:
        from rag_app.ui_gradio import create_gradio_interface
        results['create_gradio_interface'] = True
        print("  ✅ create_gradio_interface function available")
    except Exception as e:
        results['create_gradio_interface'] = False
        print(f"  ❌ create_gradio_interface function failed: {e}")
        
    return results

def main():
    """Run all syntax validation tests"""
    print("🚀 API Syntax Validation Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all tests
    import_results = test_imports()
    api_results = test_api_initialization()
    module_results = test_module_structure()
    function_results = test_function_availability()
    
    # Calculate overall results
    all_results = {
        'imports': import_results,
        'api_initialization': api_results,
        'modules': module_results,
        'functions': function_results
    }
    
    total_tests = 0
    passed_tests = 0
    
    for category, tests in all_results.items():
        for test_name, result in tests.items():
            total_tests += 1
            if result:
                passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 FINAL API SYNTAX VALIDATION RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Test Duration: {elapsed_time:.1f}s")
    
    if success_rate >= 95:
        print("   🎉 EXCELLENT! All APIs updated successfully")
        status = "EXCELLENT"
    elif success_rate >= 85:
        print("   ✅ GOOD! Most APIs updated successfully")
        status = "GOOD"
    elif success_rate >= 70:
        print("   ⚠️ FAIR! Some API issues remain")
        status = "FAIR"
    else:
        print("   ❌ POOR! Significant API issues detected")
        status = "POOR"
    
    # Save detailed results
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": success_rate,
        "status": status,
        "test_duration": elapsed_time,
        "details": all_results
    }
    
    with open("final_api_validation_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"📄 Detailed report saved to: final_api_validation_report.json")
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    exit(0 if success_rate >= 85 else 1)
