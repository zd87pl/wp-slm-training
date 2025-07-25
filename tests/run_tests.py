#!/usr/bin/env python3
"""
Test runner for WordPress SLM.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all test suites."""
    # Discover all test files
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_module):
    """Run a specific test module."""
    loader = unittest.TestLoader()
    
    try:
        # Import the test module
        module = __import__(f'tests.{test_module}', fromlist=[''])
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except ImportError as e:
        print(f"Error: Could not import test module '{test_module}': {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1].replace('.py', '')
        exit_code = run_specific_test(test_module)
    else:
        # Run all tests
        exit_code = run_all_tests()
        
    sys.exit(exit_code)