#!/usr/bin/env python
"""
Simple starter script for the BCBS Values Platform Dashboard
with Micro-Animations
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

# Default port for the server
PORT = 5000

# Set the directory containing the HTML files
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRECTORY)

print("BCBS Values Platform Dashboard Server")
print("====================================")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print("HTML files found:")
html_files = [f for f in os.listdir(".") if f.endswith(".html")]
for html_file in html_files:
    print(f"  - {html_file}")

class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for dashboard HTML files"""
    
    def do_GET(self):
        """Handle GET requests to the server"""
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/' or parsed_path.path == '':
            self.path = '/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def end_headers(self):
        """Add CORS headers to allow local testing"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

if __name__ == "__main__":
    handler = DashboardHTTPRequestHandler
    
    # Create the server on the specified host and port
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Serving BCBS Values Platform Dashboard at http://0.0.0.0:{PORT}/")
        print("Access the enhanced dashboard at http://0.0.0.0:5000/dashboard.html")
        print("Press Ctrl+C to stop the server")
        
        # Serve requests until interrupted
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")