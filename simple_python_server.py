#!/usr/bin/env python3
"""
Simple HTTP server for BCBS Values Platform
Serves static files from the current directory.
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse, parse_qs

# Configuration
PORT = 5000
BIND_ADDRESS = "0.0.0.0"

class BCBSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for BCBS Values Platform."""
    
    def do_GET(self):
        """Handle GET requests."""
        print(f"Request: {self.path}")
        
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Default to index.html
        if path == '/':
            path = '/index.html'
            
        # Try to serve the file
        try:
            # Remove leading slash
            file_path = path[1:] if path.startswith('/') else path
            
            # Check if file exists
            if os.path.exists(file_path) and os.path.isfile(file_path):
                # Determine content type
                content_type = self.guess_type(file_path)
                
                # Serve the file
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', len(content))
                self.end_headers()
                self.wfile.write(content)
            else:
                # Try to serve static_fallback.html if available
                if os.path.exists('static_fallback.html'):
                    with open('static_fallback.html', 'rb') as f:
                        content = f.read()
                    
                    self.send_response(404)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    # Default 404 response
                    self.send_error(404, f"File not found: {path}")
        except Exception as e:
            print(f"Error serving {path}: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
            
    def log_message(self, format, *args):
        """Override log_message to include timestamp."""
        sys.stderr.write(f"{self.log_date_time_string()} - {self.address_string()} - {format % args}\n")


def main():
    """Start the server."""
    print(f"BCBS Values Platform server starting on http://{BIND_ADDRESS}:{PORT}/")
    print("Available pages:")
    print(f"- Home: http://{BIND_ADDRESS}:{PORT}/")
    print(f"- Dashboard: http://{BIND_ADDRESS}:{PORT}/dashboard.html")
    
    # List HTML files in current directory
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    if html_files:
        print("\nDetected HTML files:")
        for html_file in html_files:
            print(f"- {html_file}: http://{BIND_ADDRESS}:{PORT}/{html_file}")
    
    try:
        # Create the server
        httpd = socketserver.TCPServer((BIND_ADDRESS, PORT), BCBSHTTPRequestHandler)
        
        # Serve forever
        print("\nServer is running. Press Ctrl+C to stop.")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nServer error: {e}")
    finally:
        try:
            httpd.server_close()
        except:
            pass
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()