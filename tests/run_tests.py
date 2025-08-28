#!/usr/bin/env python3
"""
Test Runner for RAG System
==========================

Comprehensive test runner for all RAG system components.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_test_suite(test_file: str, verbose: bool = True) -> bool:
    """
    Run a specific test suite.
    
    Args:
        test_file: Path to test file
        verbose: Whether to run with verbose output
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    cmd = ["python", "-m", "pytest", test_file]
    if verbose:
        cmd.append("-v")
    
    print(f"\n🧪 Running tests: {test_file}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_all_tests() -> dict:
    """
    Run all test suites and return results.
    
    Returns:
        dict: Test results summary
    """
    test_files = [
        "RAG/tests/test_basic.py",
        "RAG/tests/test_rag_service.py",
        "RAG/tests/test_rag_cli.py", 
        "RAG/tests/test_core_components.py",
        "RAG/tests/test_normalize_snapshot.py"
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    print("🚀 Starting Comprehensive RAG System Tests")
    print("=" * 60)
    
    for test_file in test_files:
        if Path(test_file).exists():
            passed = run_test_suite(test_file)
            results[test_file] = passed
            if passed:
                total_passed += 1
                print(f"✅ {test_file}: PASSED")
            else:
                total_failed += 1
                print(f"❌ {test_file}: FAILED")
        else:
            print(f"⚠️  {test_file}: NOT FOUND")
            results[test_file] = False
            total_failed += 1
    
    return {
        "results": results,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_tests": len(test_files)
    }


def run_specific_test(test_name: str) -> bool:
    """
    Run a specific test by name.
    
    Args:
        test_name: Name of the test to run
        
    Returns:
        bool: True if test passed, False otherwise
    """
    test_mapping = {
        "service": "RAG/tests/test_rag_service.py",
        "cli": "RAG/tests/test_rag_cli.py",
        "components": "RAG/tests/test_core_components.py",
        "normalize": "RAG/tests/test_normalize_snapshot.py",
        "all": None
    }
    
    if test_name == "all":
        return run_all_tests()["total_failed"] == 0
    
    if test_name in test_mapping:
        test_file = test_mapping[test_name]
        return run_test_suite(test_file)
    else:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_mapping.keys())}")
        return False


def print_summary(results: dict):
    """Print test results summary."""
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_file, passed in results["results"].items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_file}: {status}")
    
    print(f"\nTotal Tests: {results['total_tests']}")
    print(f"Passed: {results['total_passed']}")
    print(f"Failed: {results['total_failed']}")
    
    if results['total_failed'] == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n❌ {results['total_failed']} TEST(S) FAILED")
        return False


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument(
        "test", 
        nargs="?", 
        default="all",
        choices=["basic", "service", "cli", "components", "normalize", "all"],
        help="Test suite to run"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Run tests quietly (less verbose output)"
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        results = run_all_tests()
        success = print_summary(results)
        sys.exit(0 if success else 1)
    else:
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
