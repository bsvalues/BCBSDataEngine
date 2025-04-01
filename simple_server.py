"""
Simple HTTP Server for BCBS Values Platform
"""

import os
import sys
import http.server
import socketserver

PORT = 5002

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP request handler with GET support."""
    
    def log_message(self, format, *args):
        """Log message with client address."""
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))
    
    def do_GET(self):
        """Handle GET requests."""
        print(f"GET request for {self.path}")
        if self.path == '/':
            self.path = '/index.html'
        
        if self.path.startswith('/api/'):
            self.handle_api_request()
            return
        
        try:
            # Try to serve file
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        except Exception as e:
            # If file not found, serve 404 page
            self.send_error(404, f"File not found: {self.path}")
            print(f"Error serving {self.path}: {e}")
    
    def handle_api_request(self):
        """Handle API requests."""
        import json
        
        # Parse API endpoint
        parts = self.path.split('/')
        endpoint = parts[2] if len(parts) > 2 else ''
        
        response_data = {}
        
        # Basic API endpoints
        if endpoint == 'data':
            response_data = {
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
            }
        elif endpoint == 'status':
            import time
            response_data = {
                "status": "success",
                "serverStatus": "running",
                "uptime": time.time(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        else:
            response_data = {
                "status": "error",
                "message": f"Unknown endpoint: {endpoint}"
            }
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())

def run_server():
    """Run the server."""
    try:
        handler = SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
        
        print(f"Server running at http://0.0.0.0:{PORT}/")
        print(f"Current working directory: {os.getcwd()}")
        
        # List available HTML files
        html_files = [f for f in os.listdir('.') if f.endswith('.html')]
        print(f"HTML files available: {', '.join(html_files)}")
        
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()