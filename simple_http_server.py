#!/usr/bin/env python3
"""
Simple HTTP Server for BCBS Values Platform
Uses only Python standard library, no external dependencies
"""

import http.server
import socketserver
import os
import json
import datetime
import sys
import glob

# Configuration
PORT = 5002
SERVER_HOST = "0.0.0.0"


class BCBSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for BCBS Values Platform"""
    
    # Counter for requests
    request_count = 0
    # Server start time
    start_time = datetime.datetime.now()
    
    def log_message(self, format, *args):
        """Override to add timestamp to log messages"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        sys.stderr.write(f"[{timestamp}] {format % args}\n")
    
    def log_request(self, code='-', size='-'):
        """Override to count requests"""
        BCBSRequestHandler.request_count += 1
        super().log_request(code, size)
    
    def do_GET(self):
        """Handle GET requests"""
        # Handle API requests
        if self.path.startswith('/api/'):
            self.handle_api_request()
            return
        
        # Handle normal file requests
        if self.path == '/':
            self.path = '/index.html'
        
        # Try to serve the file
        try:
            super().do_GET()
        except Exception as e:
            self.log_error(f"Error serving {self.path}: {str(e)}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_api_request(self):
        """Handle API requests"""
        # Parse the path to get the API endpoint
        path_parts = self.path.split('/')
        endpoint = path_parts[2] if len(path_parts) > 2 else ''
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        # Current time
        now = datetime.datetime.now()
        
        # Generate response based on endpoint
        if endpoint == 'status':
            # Server status endpoint
            uptime = now - BCBSRequestHandler.start_time
            response = {
                'status': 'online',
                'uptime_seconds': int(uptime.total_seconds()),
                'requests_handled': BCBSRequestHandler.request_count,
                'python_version': sys.version,
                'timestamp': now.isoformat()
            }
        elif endpoint == 'agent-status':
            # Agent status endpoint - mock response
            response = {
                'status': 'success',
                'message': 'Agent status retrieved successfully',
                'data': {
                    'agents': [
                        {
                            'id': 'agent-001',
                            'name': 'ETL-Controller',
                            'status': 'active',
                            'last_heartbeat': now.isoformat(),
                            'queue_size': 12,
                            'success_rate': 0.97
                        },
                        {
                            'id': 'agent-002',
                            'name': 'Model-Executor',
                            'status': 'active',
                            'last_heartbeat': now.isoformat(),
                            'queue_size': 5,
                            'success_rate': 0.99
                        },
                        {
                            'id': 'agent-003',
                            'name': 'API-Gateway',
                            'status': 'active',
                            'last_heartbeat': now.isoformat(),
                            'queue_size': 0,
                            'success_rate': 1.0
                        }
                    ],
                    'timestamp': now.isoformat()
                }
            }
        else:
            # Default response for other endpoints
            response = {
                'status': 'success',
                'message': f'API endpoint: {endpoint}',
                'path': self.path,
                'timestamp': now.isoformat()
            }
        
        # Send the response
        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))


def ensure_index_html_exists():
    """Create a default index.html if it doesn't exist"""
    if not os.path.exists('index.html'):
        print(f"[{datetime.datetime.now().isoformat()}] Creating default index.html")
        with open('index.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
        .success { background: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    
    <div class="card">
        <h2>System Status</h2>
        <p><span class="status success">Server is running</span></p>
        <p>Current time: <span id="server-time">loading...</span></p>
    </div>
    
    <div class="card">
        <h2>Available Pages</h2>
        <ul id="page-list">
            <li><a href="/">Home</a></li>
        </ul>
    </div>
    
    <script>
        // Update server time
        function updateTime() {
            document.getElementById('server-time').textContent = new Date().toLocaleString();
        }
        
        // Initial update
        updateTime();
        
        // Update every second
        setInterval(updateTime, 1000);
        
        // List HTML files
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                console.log('Server status:', data);
            })
            .catch(error => {
                console.error('Error fetching server status:', error);
            });
    </script>
</body>
</html>""")


def main():
    """Main entry point for the HTTP server"""
    # Print banner
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{timestamp}] Starting BCBS Values Platform Server")
    print(f"[{timestamp}] Python version: {sys.version}")
    print(f"[{timestamp}] Current directory: {os.getcwd()}")
    
    # Make sure we have an index.html
    ensure_index_html_exists()
    
    # List available HTML files
    html_files = glob.glob('*.html')
    if html_files:
        print(f"[{timestamp}] Available HTML files: {', '.join(html_files)}")
    else:
        print(f"[{timestamp}] No HTML files found in current directory")
    
    # Start the server
    handler = BCBSRequestHandler
    httpd = socketserver.TCPServer((SERVER_HOST, PORT), handler)
    
    print(f"[{timestamp}] Server running at http://{SERVER_HOST}:{PORT}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Server stopped")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())