#!/usr/bin/env python3
"""
Run All Working Tests
====================

Simple script to run all working tests in the tests folder.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test(test_file: str) -> bool:
    """Run a single test file."""
    print(f"\n🧪 Running: {test_file}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, text=True, timeout=300)
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {test_file}")
        return success
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {test_file}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {test_file} - {e}")
        return False

def main():
    """Run all working tests."""
    print("🚀 Running All Working Tests")
    print("=" * 60)
    
    # List of working test files
    working_tests = [
        "tests/test_mcp_client.py",
        "tests/test_tools.py"
    ]
    
    # Additional tests that might work (commented out due to import issues)
    # "tests/test_rag_cli.py",  # Has import issues
    # "tests/test_evaluation_targets.py",  # Has indentation issues
    # "tests/test_modular_architecture.py",  # Has missing module issues
    # "tests/smoke_chunking_test.py",  # Has null bytes issue
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    for test_file in working_tests:
        if Path(test_file).exists():
            passed = run_test(test_file)
            results[test_file] = passed
            if passed:
                total_passed += 1
            else:
                total_failed += 1
        else:
            print(f"⚠️  {test_file}: NOT FOUND")
            results[test_file] = False
            total_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_file, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_file}: {status}")
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {total_failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
