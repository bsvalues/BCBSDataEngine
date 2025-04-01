#!/usr/bin/env python3
"""
Simple Python HTTP Server for BCBS Values Diagnostic
This script creates a minimal HTTP server to serve the index.html file
"""

import http.server
import socketserver
import os
import sys
import socket
import datetime

# Configuration
PORT = int(os.environ.get('PORT', 5000))
HOST = '0.0.0.0'

# Print startup message
print("=" * 50)
print("BCBS VALUES PYTHON DIAGNOSTIC SERVER")
print("=" * 50)
print(f"Starting at: {datetime.datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Listening on: http://{HOST}:{PORT}")
print("=" * 50)

# Create a simple request handler
class BCBSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for BCBS diagnostic server"""
    
    def log_message(self, format, *args):
        """Override to provide more detailed logs"""
        sys.stderr.write(f"[{datetime.datetime.now()}] {self.address_string()} - {format % args}\n")
    
    def do_GET(self):
        """Handle GET requests with fallback to index.html"""
        # If the request is for root, always serve index.html
        if self.path == '/':
            self.path = '/index.html'
        
        # Try to serve the requested file
        try:
            super().do_GET()
        except FileNotFoundError:
            # If file not found, try to serve index.html instead
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            error_message = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>404 - File Not Found</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
                    h1 {{ color: #d9534f; }}
                    pre {{ background: #f5f5f5; padding: 15px; }}
                </style>
            </head>
            <body>
                <h1>404 - File Not Found</h1>
                <p>The requested file "{self.path}" was not found on this server.</p>
                <p><a href="/">Go to Home Page</a></p>
            </body>
            </html>
            """
            
            self.wfile.write(error_message.encode('utf-8'))

# Create the HTTP server
try:
    with socketserver.TCPServer((HOST, PORT), BCBSRequestHandler) as httpd:
        print(f"Server started successfully at http://{HOST}:{PORT}/")
        print("Press Ctrl+C to stop the server")
        
        # Serve until interrupted
        httpd.serve_forever()
except socket.error as e:
    if e.errno == 98:  # Address already in use
        print(f"ERROR: Port {PORT} is already in use.")
        print("Try setting a different port using the PORT environment variable.")
        sys.exit(1)
    else:
        print(f"Socket error: {e}")
        sys.exit(1)
except KeyboardInterrupt:
    print("\nServer stopped by user.")
    sys.exit(0)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)