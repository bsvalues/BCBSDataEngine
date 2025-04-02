#!/usr/bin/env python3
"""
Simple HTTP Server for BCBS Dashboard using Python's built-in http.server module
"""

import http.server
import socketserver
import os
import sys
import time

# Constants
PORT = 8080  # Use standard port 8080
DIRECTORY = os.getcwd()  # Current working directory
DEFAULT_PAGE = 'dashboard_static.html'

class BCBSDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = f'/{DEFAULT_PAGE}'
            print(f"Serving default page: {DEFAULT_PAGE}")
            
        print(f"[Request] {self.command} {self.path}")
        sys.stdout.flush()
        
        try:
            return super().do_GET()
        except Exception as e:
            print(f"Error serving {self.path}: {str(e)}")
            sys.stdout.flush()
            self.send_error(500, f"Server error: {str(e)}")

    def log_message(self, format, *args):
        """Override to log to stdout instead of stderr"""
        sys.stdout.write("%s - - [%s] %s\n" %
                     (self.address_string(),
                      self.log_date_time_string(),
                      format % args))
        sys.stdout.flush()

def main():
    # Print startup info
    print(f"Server starting on http://0.0.0.0:{PORT}/")
    print(f"Serving files from: {DIRECTORY}")
    print(f"Default page: {DEFAULT_PAGE}")
    print(f"Python version: {sys.version}")
    sys.stdout.flush()

    # Verify default page exists
    if not os.path.exists(DEFAULT_PAGE):
        print(f"WARNING: Default page '{DEFAULT_PAGE}' not found in current directory.")
        print(f"Current directory contains: {', '.join(os.listdir('.'))}")
        sys.stdout.flush()

    # Create and start the server
    try:
        # Allow address reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("0.0.0.0", PORT), BCBSDashboardHandler) as httpd:
            print(f"Server successfully bound to port {PORT}")
            print("Press Ctrl+C to stop the server")
            sys.stdout.flush()
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()