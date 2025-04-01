#!/usr/bin/env python3
"""
Direct Diagnostic Server for BCBS Values
This is a simplified Python HTTP server to display diagnostic information
"""

import os
import sys
import platform
import datetime
import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

# Default port
PORT = int(os.environ.get("PORT", 5000))

class DiagnosticHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the diagnostic server"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_html().encode('utf-8'))
        elif self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            health_data = {
                'status': 'direct_diagnostic_mode',
                'message': 'Running in simplified direct diagnostic mode',
                'timestamp': datetime.datetime.now().isoformat(),
                'python_version': platform.python_version(),
                'system': platform.platform()
            }
            self.wfile.write(json.dumps(health_data, indent=2).encode('utf-8'))
        else:
            # Handle 404
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            self.wfile.write(b'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>404 - Not Found</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 650px; margin: 40px auto; padding: 0 10px; }
                    h1 { color: #e74c3c; }
                    a { color: #3498db; }
                </style>
            </head>
            <body>
                <h1>404 - Page Not Found</h1>
                <p>The requested resource was not found on this server.</p>
                <p><a href="/">Return to home page</a></p>
            </body>
            </html>
            ''')
    
    def generate_html(self):
        """Generate HTML for diagnostic page"""
        hostname = socket.gethostname()
        
        # Check if database connection is available
        db_status = "Unknown"
        db_status_class = "warning"
        db_url = os.environ.get("DATABASE_URL", "Not set")
        # Mask password in DATABASE_URL if present
        if "://" in db_url and "@" in db_url:
            parts = db_url.split("@")
            prefix = parts[0].split("://")
            if len(prefix) > 1:
                masked_url = f"{prefix[0]}://{prefix[1].split(':')[0]}:****@{parts[1]}"
                db_url = masked_url
        
        # Environment variables (filtered for safety)
        safe_vars = {}
        for key, value in os.environ.items():
            if key.startswith("PG") or key.startswith("DATABASE") or key == "PORT":
                if "PASSWORD" in key or key == "DATABASE_URL":
                    # Mask sensitive values
                    safe_vars[key] = "[REDACTED]"
                else:
                    safe_vars[key] = value
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>BCBS Values Direct Diagnostic</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    margin: 0; 
                    padding: 20px; 
                    color: #333;
                    background-color: #f9f9f9;
                }}
                h1 {{ 
                    color: #2c3e50; 
                    border-bottom: 2px solid #3498db; 
                    padding-bottom: 10px; 
                }}
                h2 {{ 
                    color: #3498db; 
                    margin-top: 30px; 
                }}
                pre {{ 
                    background: #f5f5f5; 
                    padding: 15px; 
                    overflow-x: auto; 
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .container {{ 
                    max-width: 800px; 
                    margin: 0 auto; 
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .card {{ 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 20px; 
                    margin-bottom: 20px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }}
                .status {{ 
                    padding: 5px 10px; 
                    border-radius: 4px; 
                    display: inline-block; 
                    font-weight: bold; 
                }}
                .success {{ 
                    background-color: #d4edda; 
                    color: #155724; 
                }}
                .warning {{ 
                    background-color: #fff3cd; 
                    color: #856404; 
                }}
                .error {{ 
                    background-color: #f8d7da; 
                    color: #721c24; 
                }}
                .info {{ 
                    background-color: #d1ecf1; 
                    color: #0c5460; 
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                }}
                table, th, td {{ 
                    border: 1px solid #ddd; 
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #f2f2f2; 
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BCBS Values Direct Diagnostic</h1>
                
                <div class="card">
                    <h2>System Information</h2>
                    <table>
                        <tr><td>Current Time:</td><td>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                        <tr><td>Python Version:</td><td>{platform.python_version()}</td></tr>
                        <tr><td>System:</td><td>{platform.platform()}</td></tr>
                        <tr><td>Hostname:</td><td>{hostname}</td></tr>
                        <tr><td>Server Port:</td><td>{PORT}</td></tr>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Application Status</h2>
                    <p><span class="status warning">DIAGNOSTIC MODE</span> Running in direct diagnostic mode.</p>
                    <p>The full application is not available at this time. This is a simplified diagnostic server.</p>
                </div>
                
                <div class="card">
                    <h2>Database Connection</h2>
                    <p><span class="status {db_status_class}">{db_status}</span></p>
                    <p>Database URL: {db_url}</p>
                </div>
                
                <div class="card">
                    <h2>Environment Variables</h2>
                    <pre>{json.dumps(safe_vars, indent=2)}</pre>
                </div>
                
                <div class="card">
                    <h2>BCBS Values Features</h2>
                    <ul>
                        <li><strong>Property Search and Filtering</strong> - Search and filter properties by various criteria</li>
                        <li><strong>Valuation Analytics</strong> - View detailed valuation data and analytics</li>
                        <li><strong>Agent Status Monitoring</strong> - Monitor the status of the BS Army of Agents</li>
                        <li><strong>ETL Pipeline Status</strong> - Monitor the status of the data extraction, transformation, and loading pipeline</li>
                        <li><strong>What-If Analysis</strong> - Perform what-if analysis to see how changes in property features affect valuations</li>
                    </ul>
                    <p><em>Note: Features are not available in diagnostic mode.</em></p>
                </div>
                
                <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
                    <p>BCBS Values Direct Diagnostic Server</p>
                    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </footer>
            </div>
        </body>
        </html>
        """

def run_server():
    """Run the diagnostic HTTP server"""
    try:
        server_address = ('0.0.0.0', PORT)
        httpd = HTTPServer(server_address, DiagnosticHandler)
        print(f"Starting direct diagnostic server on http://0.0.0.0:{PORT}")
        print(f"Server started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        httpd.server_close()
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()