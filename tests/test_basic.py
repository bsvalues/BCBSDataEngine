
import unittest
import os
import sys
import json
from flask import Flask

class TestBasicFunctionality(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        
    def test_environment(self):
        """Test Python environment basics"""
        self.assertIsNotNone(sys.executable)
        self.assertEqual(sys.version_info.major, 3)
        self.assertGreaterEqual(sys.version_info.minor, 11)
        
    def test_flask_setup(self):
        """Test Flask application setup"""
        self.assertIsNotNone(self.app)
        
    def test_critical_files(self):
        """Test presence of critical application files"""
        required_files = ['app_minimal.py', 'run_tests.py']
        for file in required_files:
            self.assertTrue(os.path.exists(file))

if __name__ == '__main__':
    unittest.main()
