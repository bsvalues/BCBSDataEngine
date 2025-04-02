
import unittest
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'tests/test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

def run_test_suite():
    """Run all test suites in order"""
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suites = [
        loader.discover(start_dir, pattern='test_basic.py'),
        loader.discover(start_dir, pattern='test_core.py'),
        loader.discover(start_dir, pattern='test_environment.py')
    ]
    
    combined_suite = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
