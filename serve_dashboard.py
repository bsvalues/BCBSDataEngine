#!/usr/bin/env python3
import http.server
import socketserver
import os

# Set port for the HTTP server
PORT = 5000

# Set the directory to serve files from (current directory)
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Change directory to where the script is located
        os.chdir(DIRECTORY)
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type, X-API-KEY")
        super().end_headers()

# Create the HTTP server with the custom handler
with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"Serving dashboard at http://0.0.0.0:{PORT}/dashboard_demo.html")
    # Start serving requests
    httpd.serve_forever()