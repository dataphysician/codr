#!/usr/bin/env python3
"""
Test Runner for CODR System

Runs all test suites and provides summary results.
Tests converted from examples to ensure all functionality works correctly.

Usage:
    python tests/run_tests.py           # Run all tests
    python tests/run_tests.py -v        # Verbose output
    python tests/run_tests.py -k tree   # Run only tree tests
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_pytest(test_path: str, args: list = None) -> tuple[bool, str]:
    """Run pytest on specified path and return success status and output."""
    cmd = ["python", "-m", "pytest", test_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, f"Error running tests: {e}"

def main():
    parser = argparse.ArgumentParser(description="Run CODR system tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-k", "--keyword", help="Run only tests matching keyword")
    parser.add_argument("--test", help="Run specific test file")
    args = parser.parse_args()
    
    # Test files to run
    test_files = [
        ("Tree Interface", "tests/test_tree_interface.py"),
        ("Traversal Engine", "tests/test_traversal_engine.py"), 
        ("Agents", "tests/test_agents.py"),
        ("Orchestration", "tests/test_orchestration.py"),
        ("End-to-End", "tests/test_end_to_end.py"),
        ("Rewind & Feedback", "tests/test_rewind_feedback.py")
    ]
    
    print("CODR System Test Suite")
    print("=" * 50)
    print("Running converted tests from examples folder\n")
    
    # Build pytest arguments
    pytest_args = []
    if args.verbose:
        pytest_args.append("-v")
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    
    # Run specific test or all tests
    if args.test:
        test_files = [(args.test, args.test)]
    
    results = []
    
    for name, test_file in test_files:
        print(f"Running {name} Tests...")
        print("-" * 30)
        
        success, output = run_pytest(test_file, pytest_args)
        results.append((name, success, output))
        
        if args.verbose or not success:
            print(output)
        else:
            # Show summary for successful tests
            lines = output.split('\n')
            summary_lines = [line for line in lines if 'passed' in line or 'failed' in line or 'error' in line]
            if summary_lines:
                print(summary_lines[-1])
        
        print()
    
    # Overall summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, output in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All test suites passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test suite(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())