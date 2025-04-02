
import sys
import unittest
import importlib.util

class TestEnvironment(unittest.TestCase):
    def test_python_version(self):
        """Verify Python version is 3.11.x"""
        major, minor = sys.version_info[:2]
        self.assertEqual(major, 3)
        self.assertGreaterEqual(minor, 11)

    def test_required_modules(self):
        """Verify critical modules are available"""
        required_modules = ['flask', 'requests', 'pandas', 'numpy']
        for module in required_modules:
            self.assertIsNotNone(
                importlib.util.find_spec(module),
                f"Required module {module} is not installed"
            )

    def test_flask_app_creation(self):
        """Verify Flask app can be created"""
        from flask import Flask
        app = Flask(__name__)
        self.assertIsNotNone(app)

if __name__ == '__main__':
    unittest.main()
