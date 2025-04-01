#!/bin/bash

# Simple diagnostic script for the BCBS Values Platform
# This script tries to run a nix-shell Python command as a last resort

echo "Starting BCBS Values Platform Diagnostic Server"
echo "================================================"
echo "Date: $(date)"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"

# Create a full Python script directly in this shell script
cat > temp_server.py << 'EOF'
#!/usr/bin/env python
import os
import sys
import http.server
import socketserver

PORT = 5002

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"GET request for {self.path}")
        if self.path == '/':
            self.path = '/index.html'
        
        if self.path.startswith('/api/'):
            # Handle API requests
            import json
            parts = self.path.split('/')
            endpoint = parts[2] if len(parts) > 2 else ''
            
            response_data = {
                "status": "success",
                "message": "BCBS Values Platform API",
                "endpoint": endpoint
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            return
        
        try:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        except Exception as e:
            self.send_error(404, f"File not found: {self.path}")
            print(f"Error serving {self.path}: {e}")

# Find available HTML files
html_files = [f for f in os.listdir('.') if f.endswith('.html')]
print(f"HTML files available: {', '.join(html_files)}")

handler = SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
print(f"Server running at http://0.0.0.0:{PORT}/")
httpd.serve_forever()
EOF

# Make the Python script executable
chmod +x temp_server.py

# List directory contents
echo "Files in directory:"
ls -la

# Try to find Python using different methods
echo "Attempting to find Python executable:"
which python3 || which python || echo "Python not found by which"

# Attempt to run with direct nix-shell invocation
echo "Attempting to run with nix-shell..."
nix-shell -p python3 --run "python3 temp_server.py"