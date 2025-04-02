#!/usr/bin/env python3
"""
Multi-Implementation Server

This module provides a HTTP server that can use multiple
implementations (Python, Node.js, or Bash) depending on
what's available in the environment.
"""

import os
import sys
import subprocess
import json
import logging
import time
import socket
import threading
import tempfile
from pathlib import Path

# Import environment detection
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from test_environment import detect_environment
except ImportError:
    # Define a basic version if the import fails
    def detect_environment():
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "python_available": True,
            "node_available": False,
            "bash_available": False,
            "bash_utilities": {},
            "allowed_ports": [5001, 5002, 5003, 8000, 8080]
        }

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiImplementationServer:
    """
    A server that can use multiple implementations based on what's available.
    """
    
    def __init__(self, host="0.0.0.0", port=5002, content_dir="."):
        """
        Initialize the server
        
        Args:
            host (str): Host to bind to
            port (int): Port to bind to
            content_dir (str): Directory containing content to serve
        """
        self.host = host
        self.port = port
        self.content_dir = os.path.abspath(content_dir)
        self.process = None
        self.implementation = None
        self.environment = detect_environment()
        self.temp_files = []
    
    def _create_html_content(self):
        """
        Create basic HTML content for the server
        
        Returns:
            str: Path to the HTML file
        """
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .status { display: flex; align-items: center; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
        .status-good { background-color: #2ecc71; }
        .status-warning { background-color: #f39c12; }
        .status-bad { background-color: #e74c3c; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    
    <div class="card">
        <h2>System Status</h2>
        <div class="status">
            <div class="status-indicator status-good"></div>
            <p>Server Running - Implementation: {implementation}</p>
        </div>
        <p>Server started at {server_time}</p>
        <p>Client time: <script>document.write(new Date().toLocaleString());</script></p>
    </div>
    
    <div class="card">
        <h2>Environment Information</h2>
        <pre id="env-info">Loading...</pre>
        <script>
            fetch('/api/environment')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('env-info').textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    document.getElementById('env-info').textContent = 'Error loading environment data';
                });
        </script>
    </div>
</body>
</html>
"""
        # Replace placeholders
        html_content = html_content.replace("{implementation}", self.implementation or "unknown")
        html_content = html_content.replace("{server_time}", time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".html")
        os.write(fd, html_content.encode("utf-8"))
        os.close(fd)
        self.temp_files.append(path)
        return path
    
    def _create_python_server(self):
        """
        Create a Python-based HTTP server
        
        Returns:
            str: Path to the server script
        """
        server_script = """#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import sys
import platform

# Server configuration
PORT = {port}
HOST = "{host}"
CONTENT_PATH = "{content_path}"
INDEX_HTML = "{index_html}"

class BCBSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve environment data for API request
        if self.path == "/api/environment":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            env_data = {{
                "platform": sys.platform,
                "python_version": sys.version,
                "os_name": os.name,
                "platform_details": platform.platform(),
                "implementation": "Python SimpleHTTPServer"
            }}
            
            self.wfile.write(json.dumps(env_data).encode())
            return
        
        # Redirect root to index.html
        if self.path == "/":
            self.path = INDEX_HTML
        
        # Default handler for other paths
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

handler = BCBSRequestHandler
httpd = socketserver.TCPServer((HOST, PORT), handler)
print(f"Server running at http://{{HOST}}:{{PORT}}/")
httpd.serve_forever()
"""
        # Replace placeholders
        index_html = self._create_html_content()
        server_script = server_script.replace("{port}", str(self.port))
        server_script = server_script.replace("{host}", self.host)
        server_script = server_script.replace("{content_path}", self.content_dir)
        server_script = server_script.replace("{index_html}", index_html)
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, server_script.encode("utf-8"))
        os.close(fd)
        os.chmod(path, 0o755)  # Make executable
        self.temp_files.append(path)
        return path
    
    def _create_node_server(self):
        """
        Create a Node.js-based HTTP server
        
        Returns:
            str: Path to the server script
        """
        server_script = """#!/usr/bin/env node
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Server configuration
const PORT = {port};
const HOST = "{host}";
const CONTENT_PATH = "{content_path}";
const INDEX_HTML = "{index_html}";

// Create server
const server = http.createServer((req, res) => {
    // Serve environment data for API request
    if (req.url === "/api/environment") {
        res.writeHead(200, {{ "Content-Type": "application/json" }});
        
        const envData = {{
            platform: process.platform,
            node_version: process.version,
            os_name: os.type(),
            os_version: os.release(),
            implementation: "Node.js HTTP Server"
        }};
        
        res.end(JSON.stringify(envData, null, 2));
        return;
    }
    
    // Serve index.html for root path
    let filePath = req.url === "/" ? INDEX_HTML : path.join(CONTENT_PATH, req.url);
    
    // Check if file exists
    fs.access(filePath, fs.constants.F_OK, (err) => {
        if (err) {
            res.writeHead(404);
            res.end("File not found");
            return;
        }
        
        // Determine content type
        let contentType = "text/plain";
        const ext = path.extname(filePath).toLowerCase();
        
        switch (ext) {{
            case ".html":
                contentType = "text/html";
                break;
            case ".js":
                contentType = "text/javascript";
                break;
            case ".css":
                contentType = "text/css";
                break;
            case ".json":
                contentType = "application/json";
                break;
            case ".png":
                contentType = "image/png";
                break;
            case ".jpg":
            case ".jpeg":
                contentType = "image/jpeg";
                break;
        }}
        
        // Read and serve the file
        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(500);
                res.end(`Server Error: ${{err.code}}`);
                return;
            }
            
            res.writeHead(200, {{ "Content-Type": contentType }});
            res.end(content);
        }});
    }});
}});

// Start server
server.listen(PORT, HOST, () => {{
    console.log(`Server running at http://${{HOST}}:${{PORT}}/`);
}});
"""
        # Replace placeholders
        index_html = self._create_html_content()
        server_script = server_script.replace("{port}", str(self.port))
        server_script = server_script.replace("{host}", self.host)
        server_script = server_script.replace("{content_path}", self.content_dir)
        server_script = server_script.replace("{index_html}", index_html)
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".js")
        os.write(fd, server_script.encode("utf-8"))
        os.close(fd)
        os.chmod(path, 0o755)  # Make executable
        self.temp_files.append(path)
        return path
    
    def _create_bash_server(self):
        """
        Create a Bash-based HTTP server
        
        Returns:
            str: Path to the server script
        """
        index_html = self._create_html_content()
        
        server_script = """#!/bin/bash
# Simple HTTP server implemented in Bash

# Server configuration
PORT={port}
HOST="{host}"
INDEX_HTML="{index_html}"

echo "Server running at http://$HOST:$PORT/"

# Function to parse HTTP request
parse_request() {
    local request="$1"
    local method=$(echo "$request" | head -n 1 | cut -d ' ' -f 1)
    local path=$(echo "$request" | head -n 1 | cut -d ' ' -f 2)
    
    echo "Method: $method, Path: $path"
    
    if [ "$path" = "/" ]; then
        path="$INDEX_HTML"
    elif [ "$path" = "/api/environment" ]; then
        # Serve environment data
        echo "HTTP/1.1 200 OK"
        echo "Content-Type: application/json"
        echo ""
        echo "{"
        echo "  \\"platform\\": \\"$(uname -s)\\","
        echo "  \\"kernel\\": \\"$(uname -r)\\","
        echo "  \\"implementation\\": \\"Bash HTTP Server\\""
        echo "}"
        return
    fi
    
    # Check if file exists
    if [ -f "$path" ]; then
        # Determine content type
        local ext="${path##*.}"
        local content_type="text/plain"
        
        case "$ext" in
            html) content_type="text/html" ;;
            js) content_type="text/javascript" ;;
            css) content_type="text/css" ;;
            json) content_type="application/json" ;;
            png) content_type="image/png" ;;
            jpg|jpeg) content_type="image/jpeg" ;;
        esac
        
        # Serve the file
        echo "HTTP/1.1 200 OK"
        echo "Content-Type: $content_type"
        echo ""
        cat "$path"
    else
        # File not found
        echo "HTTP/1.1 404 Not Found"
        echo "Content-Type: text/plain"
        echo ""
        echo "404 - File not found"
    fi
}

# Try to use different methods to create a server
if command -v nc &> /dev/null; then
    # Use netcat if available
    while true; do
        nc -l -p $PORT -c 'request=$(cat); parse_request "$request"'
    done
elif command -v socat &> /dev/null; then
    # Use socat if available
    socat TCP-LISTEN:$PORT,fork EXEC:"bash -c 'request=\$(cat); parse_request \"\$request\"'"
else
    # Fallback to pure Bash (only works on Linux)
    while true; do
        coproc bash -c "exec 3<>/dev/tcp/$HOST/$PORT; cat <&3 | parse_request | cat >&3; exec 3<&-"
        wait $COPROC_PID
    done
fi
"""
        # Replace placeholders
        server_script = server_script.replace("{port}", str(self.port))
        server_script = server_script.replace("{host}", self.host)
        server_script = server_script.replace("{index_html}", index_html)
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".sh")
        os.write(fd, server_script.encode("utf-8"))
        os.close(fd)
        os.chmod(path, 0o755)  # Make executable
        self.temp_files.append(path)
        return path
    
    def _check_server_availability(self, timeout=10):
        """
        Check if the server is available by attempting to connect
        
        Args:
            timeout (int): Timeout in seconds
            
        Returns:
            bool: True if server is available, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((self.host, self.port))
                sock.close()
                return True
            except (socket.error, OSError):
                time.sleep(0.5)
        
        return False
    
    def start(self):
        """
        Start the server using the best available implementation
        
        Returns:
            dict: Result of the start attempt
        """
        # Try to start with Python first
        if self.environment["python_available"]:
            try:
                script_path = self._create_python_server()
                logger.info(f"Starting Python server with: {sys.executable} {script_path}")
                
                self.process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if self._check_server_availability():
                    self.implementation = "python"
                    logger.info("Python server started successfully")
                    return {
                        "success": True,
                        "implementation": "python",
                        "port": self.port
                    }
                else:
                    logger.warning("Python server failed to start")
                    self.process.terminate()
                    self.process = None
            except Exception as e:
                logger.error(f"Error starting Python server: {e}")
        
        # Try Node.js next
        if self.environment["node_available"]:
            try:
                script_path = self._create_node_server()
                node_path = "node"  # Use the PATH to find node
                logger.info(f"Starting Node.js server with: {node_path} {script_path}")
                
                self.process = subprocess.Popen(
                    [node_path, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if self._check_server_availability():
                    self.implementation = "node"
                    logger.info("Node.js server started successfully")
                    return {
                        "success": True,
                        "implementation": "node",
                        "port": self.port
                    }
                else:
                    logger.warning("Node.js server failed to start")
                    self.process.terminate()
                    self.process = None
            except Exception as e:
                logger.error(f"Error starting Node.js server: {e}")
        
        # Try Bash as a last resort
        if self.environment["bash_available"]:
            try:
                script_path = self._create_bash_server()
                bash_path = "bash"  # Use the PATH to find bash
                logger.info(f"Starting Bash server with: {bash_path} {script_path}")
                
                self.process = subprocess.Popen(
                    [bash_path, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if self._check_server_availability():
                    self.implementation = "bash"
                    logger.info("Bash server started successfully")
                    return {
                        "success": True,
                        "implementation": "bash",
                        "port": self.port
                    }
                else:
                    logger.warning("Bash server failed to start")
                    self.process.terminate()
                    self.process = None
            except Exception as e:
                logger.error(f"Error starting Bash server: {e}")
        
        # All implementations failed
        logger.error("All server implementations failed to start")
        return {
            "success": False,
            "implementation": None,
            "port": self.port
        }
    
    def stop(self):
        """
        Stop the server
        
        Returns:
            bool: True if server was stopped successfully, False otherwise
        """
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.process = None
                self.implementation = None
                
                # Clean up temporary files
                for path in self.temp_files:
                    try:
                        os.unlink(path)
                    except (OSError, IOError):
                        pass
                self.temp_files = []
                
                return True
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
                return False
        
        return True  # Already stopped

def test_server_startup():
    """Test if at least one server implementation can start."""
    server = MultiImplementationServer()
    result = server.start()
    assert result["success"] is True
    assert result["implementation"] in ["python", "node", "bash"]
    assert result["port"] > 0
    server.stop()
    return True

def main():
    """Main function for server testing"""
    logger.info("Starting multi-implementation server...")
    
    server = MultiImplementationServer()
    result = server.start()
    
    if result["success"]:
        logger.info(f"Server started successfully with {result['implementation']} implementation")
        logger.info(f"Listening on http://{server.host}:{server.port}/")
        
        try:
            # Keep the server running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping server...")
            server.stop()
    else:
        logger.error("Failed to start server")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())