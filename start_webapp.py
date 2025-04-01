#!/usr/bin/env python3
"""
Launcher script for BCBS Values Platform web server
This script auto-detects the environment and starts the appropriate server
"""

import os
import sys
import logging
import socket
import time
import subprocess
import signal
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default port to use
PORT = 5000

class BCBSHandler(SimpleHTTPRequestHandler):
    """Custom handler for BCBS Values Platform"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info("%s - %s", self.address_string(), format % args)

def is_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_python_server(port):
    """Start the Python HTTP server"""
    server_address = ('0.0.0.0', port)
    
    # Create the server
    httpd = HTTPServer(server_address, BCBSHandler)
    
    # Print server information
    logger.info("Starting BCBS Values Platform Python server...")
    logger.info(f"Server running at http://0.0.0.0:{port}/")
    logger.info("Available pages:")
    logger.info(f"- Home: http://0.0.0.0:{port}/")
    logger.info(f"- Static Fallback: http://0.0.0.0:{port}/static_fallback.html")
    logger.info(f"- Dashboard: http://0.0.0.0:{port}/dashboard.html")
    logger.info(f"- What-If Analysis: http://0.0.0.0:{port}/what-if-analysis.html")
    logger.info(f"- Agent Dashboard: http://0.0.0.0:{port}/agent-dashboard.html")
    
    # Check if required files exist
    required_files = ['index.html', 'dashboard.html', 'static_fallback.html']
    logger.info("Checking for required files:")
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"- {file}: Found")
        else:
            logger.warning(f"- {file}: Not found")
    
    # Start the server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.server_close()

if __name__ == "__main__":
    try:
        logger.info("Starting BCBS Values Platform...")
        
        # Check if port is available
        if not is_port_available(PORT):
            logger.warning(f"Port {PORT} is already in use")
            logger.info("Trying to find an available port...")
            for test_port in range(PORT + 1, PORT + 10):
                if is_port_available(test_port):
                    PORT = test_port
                    logger.info(f"Using port {PORT}")
                    break
            else:
                logger.error("Unable to find an available port, exiting")
                sys.exit(1)
        
        # Start the server
        start_python_server(PORT)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)