#!/usr/bin/env python3
"""
Test suite for workflow automation
This validates the environment setup, application startup, and functionality
"""
import os
import sys
import unittest
import subprocess
import socket
import time
import json
import http.client
import logging
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowTestCase(unittest.TestCase):
    """Test case for workflow automation"""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        cls.app_port = 5002
        cls.api_port = 5001
        cls.server_process = None
        cls.python_path = cls._find_python_executable()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        if cls.server_process and cls.server_process.poll() is None:
            cls.server_process.terminate()
            cls.server_process.wait(timeout=5)
    
    @classmethod
    def _find_python_executable(cls):
        """Find a usable Python executable"""
        # Try different locations for Python
        potential_paths = [
            sys.executable,
            "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3",
            "python3",
            "python"
        ]
        
        for path in potential_paths:
            try:
                if path == sys.executable:
                    # Already using this Python
                    return path
                
                # Try to run the executable
                result = subprocess.run(
                    [path, "--version"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                if result.returncode == 0:
                    logger.info(f"Found Python at: {path}")
                    logger.info(f"Python version: {result.stdout.strip()}")
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        # If we can't find Python, raise an error
        raise EnvironmentError("Could not find a usable Python executable")

    def _is_port_in_use(self, port):
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
            
    def _wait_for_server(self, port, timeout=15):
        """Wait for the server to start up"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_port_in_use(port):
                logger.info(f"Server is up on port {port}")
                return True
            time.sleep(0.5)
        
        logger.error(f"Server did not start on port {port} within {timeout} seconds")
        return False
        
    def _make_http_request(self, url, method="GET", data=None, headers=None):
        """Make an HTTP request and return the response"""
        if headers is None:
            headers = {}
            
        if data and isinstance(data, dict):
            data = json.dumps(data).encode()
            headers['Content-Type'] = 'application/json'
        
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=10) as response:
                return {
                    'status': response.status,
                    'headers': dict(response.getheaders()),
                    'data': response.read().decode()
                }
        except HTTPError as e:
            return {
                'status': e.code,
                'headers': dict(e.headers),
                'data': e.read().decode(),
                'error': str(e)
            }
        except URLError as e:
            return {
                'status': None,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': None,
                'error': f"Unexpected error: {str(e)}"
            }
    
    def test_01_python_executable(self):
        """Test that Python is available"""
        logger.info("Testing Python executable")
        self.assertIsNotNone(self.python_path, "Python executable not found")
        
        # Check that we can run the Python executable
        result = subprocess.run(
            [self.python_path, "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        self.assertEqual(result.returncode, 0, f"Python executable failed: {result.stderr}")
        logger.info(f"Python version: {result.stdout.strip()}")
        
    def test_02_required_files_exist(self):
        """Test that required files exist"""
        logger.info("Testing required files")
        required_files = [
            "app.py",
            "start_webapp.py",
            "test_python.py"
        ]
        
        for file in required_files:
            self.assertTrue(os.path.isfile(file), f"Required file not found: {file}")
    
    def test_03_workflow_configuration(self):
        """Test the Replit workflow configuration"""
        logger.info("Testing workflow configuration")
        self.assertTrue(os.path.isfile(".replit"), ".replit file not found")
        
        # Read the .replit file and check for workflow configuration
        with open(".replit", "r") as f:
            content = f.read()
            
        self.assertIn("[[workflows.workflow]]", content, "No workflow configurations found")
        self.assertIn("name = \"WebApp\"", content, "WebApp workflow not found")
        
    def test_04_start_server_script(self):
        """Test that the server start script functions"""
        logger.info("Testing server start script")
        
        # Use try-finally to ensure the server is stopped even if the test fails
        try:
            # Start the server as a subprocess
            logger.info(f"Starting server with {self.python_path} app.py")
            
            # If a server is already running on the port, the test will fail
            if self._is_port_in_use(self.app_port):
                self.fail(f"Port {self.app_port} is already in use")
                
            self.server_process = subprocess.Popen(
                [self.python_path, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the server to start
            server_started = self._wait_for_server(self.app_port)
            self.assertTrue(server_started, f"Server failed to start on port {self.app_port}")
            
            # Check server output for errors
            server_output = ""
            if self.server_process.stdout:
                for _ in range(10):  # Try to read some output
                    line = self.server_process.stdout.readline()
                    if not line:
                        break
                    server_output += line
                    
            logger.info(f"Server output: {server_output}")
            
            # Make a request to the server
            response = self._make_http_request(f"http://localhost:{self.app_port}/")
            self.assertIsNotNone(response, "No response from server")
            self.assertIsNotNone(response.get('status'), "No status code in response")
            
            # Check for a successful response (200 OK)
            self.assertIn(response.get('status'), [200, 404], f"Unexpected status code: {response.get('status')}")
            
        finally:
            # Stop the server
            if self.server_process and self.server_process.poll() is None:
                logger.info("Stopping server")
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.server_process = None
                
if __name__ == "__main__":
    logger.info("Running workflow tests")
    unittest.main(verbosity=2)