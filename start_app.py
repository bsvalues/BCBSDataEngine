#!/usr/bin/env python
"""
Simple diagnostic server for BCBS Values application
"""
import datetime
import html
import json
import os
import platform
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class DiagnosticHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for diagnostic information"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Send a basic diagnostic page
            self.wfile.write(self.generate_diagnostic_html().encode())
        elif self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                "status": "diagnostic_mode",
                "service": "BCBS Values API",
                "version": "1.0.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "environment": {
                    "python_version": platform.python_version(),
                    "system": platform.system(),
                    "node": platform.node()
                }
            }
            self.wfile.write(json.dumps(health_data, indent=2).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>404 Not Found</h1><p>The requested path was not found.</p>')
    
    def generate_diagnostic_html(self):
        """Generate the diagnostic HTML page"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BCBS Values Diagnostic</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #0056b3; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; background-color: #f9f9f9; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .api-test {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BCBS Values Diagnostic</h1>
        <p>This is a diagnostic server for the BCBS Values application.</p>
        
        <div class="card">
            <h2>System Information</h2>
            <table>
                <tr><th>Item</th><th>Value</th></tr>
                <tr><td>Date & Time</td><td>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td>Python Version</td><td>{platform.python_version()}</td></tr>
                <tr><td>Platform</td><td>{platform.platform()}</td></tr>
                <tr><td>System</td><td>{platform.system()} {platform.release()}</td></tr>
                <tr><td>Machine</td><td>{platform.machine()}</td></tr>
                <tr><td>Node</td><td>{platform.node()}</td></tr>
                <tr><td>Current Directory</td><td>{os.getcwd()}</td></tr>
                <tr><td>Python Path</td><td>{sys.executable}</td></tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Environment Variables</h2>
            <table>
                <tr><th>Variable</th><th>Value</th></tr>
                <tr><td>DATABASE_URL</td><td>{html.escape(os.environ.get('DATABASE_URL', 'Not set'))}</td></tr>
                <tr><td>SESSION_SECRET</td><td>{html.escape('*****' if os.environ.get('SESSION_SECRET') else 'Not set')}</td></tr>
                <tr><td>PORT</td><td>{html.escape(os.environ.get('PORT', '5000'))}</td></tr>
                <tr><td>PYTHONPATH</td><td>{html.escape(os.environ.get('PYTHONPATH', 'Not set'))}</td></tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Available Endpoints</h2>
            <table>
                <tr><th>Endpoint</th><th>Description</th><th>Status</th></tr>
                <tr>
                    <td>/</td>
                    <td>This diagnostic page</td>
                    <td class="success">Available</td>
                </tr>
                <tr>
                    <td>/api/health</td>
                    <td>API health check endpoint</td>
                    <td class="success">Available</td>
                </tr>
                <tr>
                    <td>/api/properties</td>
                    <td>Property data API</td>
                    <td class="warning">Not Implemented in Diagnostic Mode</td>
                </tr>
                <tr>
                    <td>/api/valuations</td>
                    <td>Valuation data API</td>
                    <td class="warning">Not Implemented in Diagnostic Mode</td>
                </tr>
                <tr>
                    <td>/api/agent-status</td>
                    <td>Agent status API</td>
                    <td class="warning">Not Implemented in Diagnostic Mode</td>
                </tr>
                <tr>
                    <td>/api/etl-status</td>
                    <td>ETL status API</td>
                    <td class="warning">Not Implemented in Diagnostic Mode</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>API Test</h2>
            <div class="api-test">
                <button onclick="testAPI('/api/health')">Test Health Endpoint</button>
                <pre id="api-result">Click the button to test the API</pre>
            </div>
        </div>
    </div>

    <script>
        function testAPI(endpoint) {
            const resultElement = document.getElementById('api-result');
            resultElement.textContent = 'Loading...';
            resultElement.style.color = 'blue';
            
            fetch(endpoint)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    resultElement.textContent = JSON.stringify(data, null, 2);
                    resultElement.style.color = 'green';
                })
                .catch(error => {
                    resultElement.textContent = `Error: ${error.message}`;
                    resultElement.style.color = 'red';
                });
        }
    </script>
</body>
</html>
"""

def run_server(port=5000):
    """Run the HTTP server"""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, DiagnosticHandler)
    print(f"Starting server on port {port}...")
    print(f"Access it at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user.")

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    run_server(port)