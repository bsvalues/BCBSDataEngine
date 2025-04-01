#!/usr/bin/env python3
"""
Very simple HTTP server for BCBS Values Platform.
Designed to be as compatible as possible with the Replit environment.
"""
import http.server
import socketserver

# Configuration
PORT = 5002
HOSTNAME = "0.0.0.0"

# Create handler
handler = http.server.SimpleHTTPRequestHandler

# Create server
with socketserver.TCPServer((HOSTNAME, PORT), handler) as httpd:
    print(f"Server started at http://{HOSTNAME}:{PORT}")
    httpd.serve_forever()