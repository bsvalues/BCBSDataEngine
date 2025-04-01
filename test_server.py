#!/usr/bin/env python3
"""
Test suite for BCBS Values Platform server functionality.
Run with: python3 test_server.py
"""
import unittest
import subprocess
import time
import socket
import os
import sys
import json
import urllib.request
import urllib.error

class ServerTests(unittest.TestCase):
    """Tests for the BCBS Values Platform server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.python_path = "/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
        cls.port = 5002
        cls.host = "localhost"
        cls.base_url = f"http://{cls.host}:{cls.port}"
        # Process will be set in test_1_server_starts
        cls.process = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if cls.process:
            cls.process.terminate()
            cls.process.wait()
            print("Server process terminated.")
    
    def test_1_python_path_exists(self):
        """Test if the Python path exists and is executable."""
        self.assertTrue(os.path.exists(self.python_path), 
                      f"Python path {self.python_path} does not exist")
        self.assertTrue(os.access(self.python_path, os.X_OK),
                      f"Python path {self.python_path} is not executable")
        
        # Test if it can run Python
        result = subprocess.run([self.python_path, "-c", "print('Python works')"], 
                               capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "Python executable returned non-zero exit code")
        self.assertIn("Python works", result.stdout, "Python failed to execute simple command")
    
    def test_2_server_starts(self):
        """Test if server starts without errors."""
        # Kill any existing server processes
        try:
            subprocess.run(["pkill", "-f", "simple_server.py"], check=False)
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Failed to kill existing processes: {e}")
        
        # Start the server
        try:
            self.__class__.process = subprocess.Popen(
                [self.python_path, "simple_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Wait for server to start
            time.sleep(2)
            
            # Check if process is still running
            self.assertIsNone(self.__class__.process.poll(), 
                            "Server process terminated unexpectedly")
            
            # Check server output for errors
            output = ""
            while True:
                line = self.__class__.process.stdout.readline()
                if not line and self.__class__.process.poll() is not None:
                    break
                output += line
                if "Starting server at" in line:
                    break
            
            self.assertIn("Starting server at", output, 
                        "Server did not output expected startup message")
            
        except Exception as e:
            self.fail(f"Failed to start server: {e}")
    
    def test_3_port_is_open(self):
        """Test if the server port is open and accepting connections."""
        # Simple check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((self.host, self.port))
        sock.close()
        
        self.assertEqual(result, 0, f"Port {self.port} is not open")
    
    def test_4_index_html_served(self):
        """Test if index.html is served correctly."""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/")
            content = response.read().decode('utf-8')
            
            self.assertEqual(response.status, 200, "Server did not return HTTP 200")
            self.assertIn("html", content.lower(), "Response does not contain HTML")
            
            # Check content type
            self.assertIn('text/html', response.headers.get('Content-Type', ''),
                        "Response Content-Type is not text/html")
            
        except urllib.error.URLError as e:
            self.fail(f"Failed to connect to server: {e}")
    
    def test_5_dashboard_html_served(self):
        """Test if dashboard.html is served correctly."""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/dashboard.html")
            content = response.read().decode('utf-8')
            
            self.assertEqual(response.status, 200, "Server did not return HTTP 200 for dashboard.html")
            self.assertIn("dashboard", content.lower(), "Response does not appear to be the dashboard")
            
        except urllib.error.URLError as e:
            self.fail(f"Failed to connect to server: {e}")
    
    def test_6_404_handling(self):
        """Test if missing files return 404."""
        try:
            # This should fail with 404
            urllib.request.urlopen(f"{self.base_url}/nonexistent-file.html")
            self.fail("Server did not return 404 for nonexistent file")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404, f"Expected 404, got {e.code}")
    
    def test_7_workflow_json_syntax(self):
        """Test if workflow JSON is valid and properly formatted."""
        try:
            with open('.replit.workflow.json', 'r') as f:
                workflow_json = json.load(f)
            
            # Basic validation
            self.assertIn('version', workflow_json, "Workflow JSON missing 'version' field")
            self.assertIn('workflows', workflow_json, "Workflow JSON missing 'workflows' field")
            
            # Check for WebApp workflow
            webapp_found = False
            for workflow in workflow_json['workflows']:
                if workflow.get('name') == 'WebApp':
                    webapp_found = True
                    # Check for required tasks
                    self.assertIn('tasks', workflow, "WebApp workflow missing 'tasks'")
                    self.assertTrue(len(workflow['tasks']) > 0, "WebApp workflow has no tasks")
                    
                    # Check for port forwarding
                    port_forward_found = False
                    for task in workflow['tasks']:
                        if task.get('task') == 'port.forward':
                            port_forward_found = True
                            self.assertIn('args', task, "port.forward task missing 'args'")
                            self.assertIn('port', task['args'], "port.forward args missing 'port'")
                            self.assertEqual(task['args']['port'], 5002, "port.forward port is not 5002")
                    
                    self.assertTrue(port_forward_found, "WebApp workflow missing port.forward task")
                    
                    # Check for shell.exec
                    shell_exec_found = False
                    for task in workflow['tasks']:
                        if task.get('task') == 'shell.exec':
                            shell_exec_found = True
                            self.assertIn('args', task, "shell.exec task missing 'args'")
                            self.assertIn('command', task['args'], "shell.exec args missing 'command'")
                            self.assertIn('python', task['args']['command'].lower(), 
                                       "shell.exec command doesn't contain 'python'")
                    
                    self.assertTrue(shell_exec_found, "WebApp workflow missing shell.exec task")
            
            self.assertTrue(webapp_found, "No WebApp workflow found in workflow JSON")
            
        except FileNotFoundError:
            self.fail(".replit.workflow.json file not found")
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON in .replit.workflow.json: {e}")

if __name__ == '__main__':
    unittest.main()