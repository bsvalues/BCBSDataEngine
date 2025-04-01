#!/usr/bin/env python3
"""
Web Application Server for BCBS Values Platform
Provides a simple HTTP server for serving the dashboard and static files
"""
import os
import sys
import http.server
import socketserver
import socket
import logging
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PORT = 5002
HOST = "0.0.0.0"

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the BCBS Values Platform."""
    
    def log_message(self, format, *args):
        """Log message with client address."""
        logger.info(f"{self.client_address[0]} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests."""
        # Special case for root
        if self.path == "/":
            self.path = "/index.html"
        
        # Handle API requests
        if self.path.startswith("/api/"):
            self.handle_api_request()
            return
        
        # Let the parent class handle file serving
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def handle_api_request(self):
        """Handle API requests."""
        parts = urlparse(self.path)
        path = parts.path[5:]  # Remove /api/ prefix
        
        # Basic API for sample data
        if path == "data":
            self.send_json_response({
                "status": "success",
                "data": {
                    "propertyCount": 12548,
                    "averageValue": 452000,
                    "recentProperties": [
                        {"address": "123 Main St", "value": 350000},
                        {"address": "456 Oak Ave", "value": 475000},
                        {"address": "789 Pine Blvd", "value": 560000}
                    ]
                }
            })
        else:
            self.send_json_response({
                "status": "error",
                "message": f"Unknown API endpoint: {path}"
            }, 404)
    
    def send_json_response(self, data, status=200):
        """Send a JSON response."""
        import json
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

def is_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def main():
    """Run the web application server."""
    # Print diagnostics
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # List HTML files
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    logger.info(f"HTML files found: {', '.join(html_files)}")
    
    # Check if port is available
    if not is_port_available(PORT):
        logger.error(f"Port {PORT} is already in use!")
        sys.exit(1)
    
    # Create server
    try:
        with socketserver.TCPServer((HOST, PORT), RequestHandler) as httpd:
            logger.info(f"Server running at http://{HOST}:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by keyboard interrupt")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()