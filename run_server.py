#!/usr/bin/env python3
"""
Universal server startup script for BCBS Values Platform
This script can run on any Python version without external dependencies
"""
import os
import sys
import subprocess
import time
import datetime
import socket
import http.server
import socketserver

# Configuration
PORT = 5002
HOST = "0.0.0.0"

def log(message):
    """Print a log message with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def find_python_script():
    """Find the best available Python script to run"""
    # Scripts in order of preference
    scripts = [
        "simple_http_server.py",
        "app.py",
        "start_webapp.py"
    ]
    
    for script in scripts:
        if os.path.isfile(script):
            log(f"Found server script: {script}")
            return script
    
    log("No suitable server script found, using built-in HTTP server")
    return None

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0

def run_server():
    """Run the HTTP server"""
    log("Starting BCBS Values Platform server")
    
    # Check if port is already in use
    if is_port_in_use(PORT):
        log(f"ERROR: Port {PORT} is already in use")
        sys.exit(1)
    
    # Find the best script to run
    script = find_python_script()
    
    if script:
        try:
            # Try to run the script directly
            log(f"Attempting to run {script}")
            
            # Make the script executable if it isn't already
            if not os.access(script, os.X_OK):
                os.chmod(script, 0o755)
                log(f"Made {script} executable")
            
            # Run the script
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Log the PID
            log(f"Server started with PID {process.pid}")
            
            # Wait for the process
            return_code = process.wait()
            
            # Check if process exited
            if return_code != 0:
                log(f"Server process exited with code {return_code}")
                stdout, stderr = process.communicate()
                log(f"stdout: {stdout}")
                log(f"stderr: {stderr}")
                
                # Fall back to built-in server
                log("Falling back to built-in HTTP server")
                run_builtin_server()
            
        except Exception as e:
            log(f"Error running {script}: {e}")
            log("Falling back to built-in HTTP server")
            run_builtin_server()
    else:
        # No script found, use built-in server
        run_builtin_server()

def run_builtin_server():
    """Run a built-in HTTP server as a last resort"""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            log(f"{self.address_string()} - {format % args}")
    
    try:
        # Create the server
        log(f"Starting built-in HTTP server on {HOST}:{PORT}")
        httpd = socketserver.TCPServer((HOST, PORT), Handler)
        
        # Serve until interrupted
        log("Server running, press Ctrl+C to stop")
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("Server stopped by user")
    except Exception as e:
        log(f"Error running built-in server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()