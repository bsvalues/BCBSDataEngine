#!/usr/bin/env python3
"""
Simple HTTP Server for BCBS Dashboard
This server works with Python 3.10+ to serve static files
"""

import http.server
import socketserver
import os
import sys
import time

# Constants
PORT = 5002
DIRECTORY = os.getcwd()  # Current working directory
DEFAULT_PAGE = 'dashboard_static.html'

# Print startup info
print(f"Server starting on http://0.0.0.0:{PORT}/")
print(f"Serving files from: {DIRECTORY}")
print(f"Default page: {DEFAULT_PAGE}")
print(f"Python version: {sys.version}")
print("Press Ctrl+C to stop the server")
sys.stdout.flush()  # Ensure output is displayed immediately

# Verify that the default page exists
if not os.path.exists(DEFAULT_PAGE):
    print(f"WARNING: Default page '{DEFAULT_PAGE}' not found in current directory!")
    print(f"Current directory contains: {os.listdir('.')}")
    sys.stdout.flush()

# Custom request handler to serve the default page for root requests
class BCBSDashboardHandler(http.server.SimpleHTTPRequestHandler):
    # Override to set common headers and handle default page
    def do_GET(self):
        # Serve the default page for root requests
        if self.path == '/':
            self.path = f'/{DEFAULT_PAGE}'
            print(f"Serving default page: {DEFAULT_PAGE}")
        
        # Log the request
        print(f"[Request] {self.command} {self.path}")
        sys.stdout.flush()
        
        try:
            # Handle the request using the parent class method
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        except Exception as e:
            print(f"Error serving {self.path}: {str(e)}")
            sys.stdout.flush()
            self.send_error(500, f"Server error: {str(e)}")
    
    # Set response headers for better caching and security
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')  # Allow cross-origin requests
        http.server.SimpleHTTPRequestHandler.end_headers(self)

# Attempt to create the server with retries
max_retries = 3
retry_count = 0
while retry_count < max_retries:
    try:
        # Allow port reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        # Set up and start the server
        handler = BCBSDashboardHandler
        httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
        print(f"Server successfully bound to port {PORT}")
        sys.stdout.flush()
        break
    except OSError as e:
        retry_count += 1
        print(f"Failed to bind to port {PORT}: {str(e)}")
        if retry_count < max_retries:
            print(f"Retrying in 5 seconds... (Attempt {retry_count}/{max_retries})")
            sys.stdout.flush()
            time.sleep(5)
        else:
            print("Maximum retries reached. Unable to start server.")
            sys.exit(1)

try:
    # Start the HTTP server
    print("Server is now ready to accept connections")
    sys.stdout.flush()
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped by user")
    httpd.server_close()
    sys.exit(0)
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    sys.stdout.flush()
    httpd.server_close()
    sys.exit(1)