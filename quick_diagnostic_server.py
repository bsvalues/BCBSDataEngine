#!/usr/bin/env python3
"""
Simple HTTP Server to display BCBS Values Diagnostic Information
"""
import os
import sys
import datetime
import json
import html
from http.server import HTTPServer, BaseHTTPRequestHandler

class DiagnosticHandler(BaseHTTPRequestHandler):
    """HTTP Handler for serving diagnostic information"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Send the diagnostic HTML
            self.wfile.write(self.generate_diagnostic_html().encode())
        elif self.path == '/api.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Send API information in JSON format
            api_info = {
                "status": "diagnostic",
                "api_version": "1.0.0",
                "endpoints": [
                    {"path": "/api/health", "method": "GET", "auth": False, "status": "not_checked"},
                    {"path": "/api/properties", "method": "GET", "auth": True, "status": "not_checked"},
                    {"path": "/api/valuations", "method": "GET", "auth": True, "status": "not_checked"},
                    {"path": "/api/agent-status", "method": "GET", "auth": True, "status": "not_checked"},
                    {"path": "/api/etl-status", "method": "GET", "auth": True, "status": "not_checked"},
                    {"path": "/api/market-trends", "method": "GET", "auth": True, "status": "not_checked"}
                ],
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(api_info, indent=2).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>404 Not Found</h1>')
    
    def generate_diagnostic_html(self):
        """Generate the diagnostic HTML page"""
        # Get environment information
        env_info = self.get_environment_info()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>BCBS Values - Diagnostic Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding-top: 2rem; padding-bottom: 2rem; background-color: #f8f9fa; }}
                .header {{ background-color: #3366cc; color: white; padding: 2rem 0; margin-bottom: 2rem; }}
                .status-card {{ margin-bottom: 1.5rem; transition: all 0.3s ease; }}
                .status-card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
                .api-section {{ background-color: #f0f4f8; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; }}
                .endpoint-item {{ padding: 0.75rem; border-left: 4px solid #3366cc; margin-bottom: 0.5rem; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header text-center">
                    <h1>BCBS Values Diagnostic Report</h1>
                    <p class="lead">System and Configuration Status</p>
                </div>

                <div class="row">
                    <div class="col-md-6">
                        <div class="card status-card">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0">Environment Status</h5>
                            </div>
                            <div class="card-body">
                                <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                                <p><strong>Python:</strong> {sys.version.split()[0]}</p>
                                <p><strong>OS:</strong> {html.escape(sys.platform)}</p>
                                <p><strong>Database:</strong> <span class="warning">Connection not verified</span></p>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-6">
                        <div class="card status-card">
                            <div class="card-header bg-info text-white">
                                <h5 class="mb-0">Environment Variables</h5>
                            </div>
                            <div class="card-body">
                                <p><strong>DATABASE_URL:</strong> <span class="{env_info['db_url_class']}">{env_info['db_url_status']}</span></p>
                                <p><strong>PGDATABASE:</strong> <span class="{env_info['pgdatabase_class']}">{env_info['pgdatabase_status']}</span></p>
                                <p><strong>SESSION_SECRET:</strong> <span class="{env_info['session_secret_class']}">{env_info['session_secret_status']}</span></p>
                                <p><strong>API_KEY:</strong> <span class="{env_info['api_key_class']}">{env_info['api_key_status']}</span></p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="api-section mt-4">
                    <h3>Available API Endpoints</h3>
                    <p>These endpoints are defined in the BCBS Values API:</p>

                    <div class="endpoint-item">
                        <h5>/api/health</h5>
                        <p>Health check endpoint to verify API is operational</p>
                        <p><small>Method: GET, Authentication: None</small></p>
                    </div>

                    <div class="endpoint-item">
                        <h5>/api/properties</h5>
                        <p>Retrieve property listings with optional filtering</p>
                        <p><small>Method: GET, Authentication: API Key</small></p>
                    </div>

                    <div class="endpoint-item">
                        <h5>/api/valuations</h5>
                        <p>Access property valuations with filtering options</p>
                        <p><small>Method: GET, Authentication: API Key</small></p>
                    </div>

                    <div class="endpoint-item">
                        <h5>/api/agent-status</h5>
                        <p>Get status information on valuation agents</p>
                        <p><small>Method: GET, Authentication: API Key</small></p>
                    </div>

                    <div class="endpoint-item">
                        <h5>/api/etl-status</h5>
                        <p>Monitor ETL pipeline execution status</p>
                        <p><small>Method: GET, Authentication: API Key</small></p>
                    </div>
                    
                    <div class="endpoint-item">
                        <h5>/api/market-trends</h5>
                        <p>Get real estate market trend data</p>
                        <p><small>Method: GET, Authentication: API Key</small></p>
                    </div>
                </div>

                <div class="text-center mt-4 mb-5">
                    <h4>Diagnostic Results</h4>
                    <p>This is a read-only diagnostic page. The Flask application is not fully operational.</p>
                    <p>
                        <strong>Summary:</strong> Python environment found, but Flask application dependencies 
                        may not be properly installed. PostgreSQL database connection could not be verified.
                    </p>
                </div>
                
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0">API Information</h5>
                            </div>
                            <div class="card-body">
                                <p>For structured API information, visit the <a href="/api.json">API JSON endpoint</a>.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="bg-light text-center text-muted py-4 mt-5">
                <div class="container">
                    <p>BCBS Values Diagnostic Server - {datetime.datetime.now().year}</p>
                </div>
            </footer>
        </body>
        </html>
        """
        
        return html_content
    
    def get_environment_info(self):
        """Get information about the environment variables"""
        info = {}
        
        # Check DATABASE_URL
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
            info['db_url_status'] = "Set (value hidden)"
            info['db_url_class'] = "success"
        else:
            info['db_url_status'] = "Not set"
            info['db_url_class'] = "error"
        
        # Check PGDATABASE
        pgdatabase = os.environ.get('PGDATABASE')
        if pgdatabase:
            info['pgdatabase_status'] = pgdatabase
            info['pgdatabase_class'] = "success"
        else:
            info['pgdatabase_status'] = "Not set"
            info['pgdatabase_class'] = "error"
        
        # Check SESSION_SECRET
        session_secret = os.environ.get('SESSION_SECRET')
        if session_secret:
            if session_secret == "bcbs_values_session_secret_key_2025":
                info['session_secret_status'] = "Default value (should be changed)"
                info['session_secret_class'] = "warning"
            else:
                info['session_secret_status'] = "Set (value hidden)"
                info['session_secret_class'] = "success"
        else:
            info['session_secret_status'] = "Not set"
            info['session_secret_class'] = "error"
        
        # Check API_KEY
        api_key = os.environ.get('API_KEY') or os.environ.get('BCBS_VALUES_API_KEY')
        if api_key:
            if api_key == "bcbs_values_api_key_2025":
                info['api_key_status'] = "Default value (should be changed)"
                info['api_key_class'] = "warning"
            else:
                info['api_key_status'] = "Set (value hidden)"
                info['api_key_class'] = "success"
        else:
            info['api_key_status'] = "Not set"
            info['api_key_class'] = "error"
        
        return info

def run_server(port=5000):
    """Run the diagnostic server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, DiagnosticHandler)
    print(f"Starting diagnostic server on port {port}...")
    print(f"Python version: {sys.version}")
    print(f"Server address: http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        run_server(port)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)