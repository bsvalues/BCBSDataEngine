#!/usr/bin/env python
"""
Simple HTTP server to serve static files for BCBS Values
This is a fallback server when Node.js is unavailable
"""

import sys
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default port
PORT = 5000

def run_server(port):
    """Run a simple HTTP server on the specified port"""
    server_address = ('0.0.0.0', port)
    
    # Create the server
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    # Print server information
    logging.info("Starting BCBS Values Python fallback server...")
    logging.info(f"Server running at http://0.0.0.0:{port}/")
    logging.info("Available pages:")
    logging.info(f"- Home: http://0.0.0.0:{port}/")
    logging.info(f"- Static Fallback: http://0.0.0.0:{port}/static_fallback.html")
    logging.info(f"- Dashboard: http://0.0.0.0:{port}/dashboard.html")
    logging.info(f"- What-If Analysis: http://0.0.0.0:{port}/what-if-analysis.html")
    logging.info(f"- Agent Dashboard: http://0.0.0.0:{port}/agent-dashboard.html")
    
    # Check if required files exist
    required_files = ['index.html', 'dashboard.html', 'what-if-analysis.html', 'agent-dashboard.html', 'static_fallback.html']
    logging.info("Checking for required files:")
    for file in required_files:
        if os.path.exists(file):
            logging.info(f"- {file}: Found")
        else:
            logging.warning(f"- {file}: Not found")
    
    # Start the server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
        httpd.server_close()

if __name__ == "__main__":
    # Try to get port from command line argument
    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            logging.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    # Run the server
    run_server(PORT)