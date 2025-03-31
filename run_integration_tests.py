#!/usr/bin/env python
"""
BCBS Values Integration Tests Runner

This script runs all integration tests for the BCBS Values application.
It provides a convenient way to execute the full test suite or specific
test modules with appropriate configuration.

Usage:
    python run_integration_tests.py [options]

Options:
    --etl-only        Run only ETL pipeline integration tests
    --api-only        Run only API integration tests
    --verbose, -v     Increase verbosity
    --help, -h        Show this help message

Examples:
    # Run all integration tests
    python run_integration_tests.py

    # Run only ETL integration tests
    python run_integration_tests.py --etl-only

    # Run only API integration tests
    python run_integration_tests.py --api-only

    # Run with increased verbosity
    python run_integration_tests.py -v

Author: BCBS Test Engineering Team
Last Updated: 2025-03-31
"""
import os
import sys
import argparse
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('integration_tests')

# Define test modules
INTEGRATION_TEST_MODULES = [
    './tests/test_integration_advanced.py',
    './tests/test_etl_integration.py'
]

ETL_TEST_MODULES = [
    './tests/test_etl_integration.py'
]

API_TEST_MODULES = [
    './tests/test_integration_advanced.py'
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run BCBS Values integration tests')
    parser.add_argument('--etl-only', action='store_true', help='Run only ETL pipeline integration tests')
    parser.add_argument('--api-only', action='store_true', help='Run only API integration tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase verbosity')
    return parser.parse_args()


def run_tests(test_modules, verbose=False):
    """
    Run the specified test modules using pytest.
    
    Args:
        test_modules: List of test modules to run
        verbose: Whether to increase verbosity
    
    Returns:
        int: Return code from pytest (0 for success, non-zero for failure)
    """
    # Build pytest command
    cmd = ['pytest']
    if verbose:
        cmd.append('-vvs')
    else:
        cmd.append('-xvs')
    
    # Add test modules
    cmd.extend(test_modules)
    
    logger.info(f"Running tests: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Determine which tests to run
    if args.etl_only:
        logger.info("Running ETL pipeline integration tests")
        test_modules = ETL_TEST_MODULES
    elif args.api_only:
        logger.info("Running API integration tests")
        test_modules = API_TEST_MODULES
    else:
        logger.info("Running all integration tests")
        test_modules = INTEGRATION_TEST_MODULES
    
    # Run the tests
    return_code = run_tests(test_modules, args.verbose)
    
    # Report results
    if return_code == 0:
        logger.info("All tests passed successfully!")
    else:
        logger.error(f"Tests failed with return code {return_code}")
    
    return return_code


if __name__ == '__main__':
    sys.exit(main())