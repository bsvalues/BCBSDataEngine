
import unittest
import os
import sys

class TestCoreEnvironment(unittest.TestCase):
    def test_python_environment(self):
        """Verify Python environment is correctly configured"""
        self.assertEqual(sys.version_info.major, 3)
        self.assertGreaterEqual(sys.version_info.minor, 11)
        
    def test_file_structure(self):
        """Verify critical application files exist"""
        required_files = ['app_minimal.py', 'run_tests.py']
        for file in required_files:
            self.assertTrue(os.path.exists(file), f"Missing required file: {file}")

    def test_port_availability(self):
        """Verify port 5000 is available"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('0.0.0.0', 5000))
            self.assertNotEqual(result, 0, "Port 5000 should be available")

if __name__ == '__main__':
    unittest.main()
