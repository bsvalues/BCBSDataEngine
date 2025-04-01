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
                                <p><strong>Python Version:</strong> {sys.version.split()[0]}</p>
                                <p><strong>Python Path:</strong> {html.escape(env_info.get('python_path', 'Unknown'))}</p>
                                <p><strong>OS Platform:</strong> {html.escape(sys.platform)}</p>
                                <p><strong>Database:</strong> <span class="{env_info.get('db_conn_class', 'warning')}">{env_info.get('db_conn_summary', 'Connection not verified')}</span></p>
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
                                <p><strong>PGHOST:</strong> <span class="{env_info.get('pghost_class', 'warning')}">{env_info.get('pghost_status', 'Not checked')}</span></p>
                                <p><strong>PGPORT:</strong> <span class="{env_info.get('pgport_class', 'warning')}">{env_info.get('pgport_status', 'Not checked')}</span></p>
                                <p><strong>PGUSER:</strong> <span class="{env_info.get('pguser_class', 'warning')}">{env_info.get('pguser_status', 'Not checked')}</span></p>
                                <p><strong>PGPASSWORD:</strong> <span class="{env_info.get('pgpassword_class', 'warning')}">{env_info.get('pgpassword_status', 'Not checked')}</span></p>
                                <p><strong>PGDATABASE:</strong> <span class="{env_info.get('pgdatabase_class', 'warning')}">{env_info.get('pgdatabase_status', 'Not checked')}</span></p>
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
                    <p>This is a read-only diagnostic page showing the environment configuration.</p>
                    <p>
                        <strong>Summary:</strong> Python {sys.version.split()[0]} environment detected at {html.escape(env_info.get('python_path', 'Unknown'))}. 
                        PostgreSQL database connection parameters are {env_info.get('db_conn_summary', 'not fully configured')}.
                    </p>
                    <p>
                        Common issues:
                        <ul class="list-inline">
                            <li class="list-inline-item badge bg-info text-white">Missing environment variables</li>
                            <li class="list-inline-item badge bg-info text-white">Missing Python dependencies</li>
                            <li class="list-inline-item badge bg-info text-white">Invalid database connection</li>
                        </ul>
                    </p>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0">Required Python Modules</h5>
                            </div>
                            <div class="card-body">
                                <p><strong>Flask:</strong> <span class="{env_info.get('flask_class', 'warning')}">{env_info.get('flask_status', 'Not checked')}</span></p>
                                <p><strong>Pandas:</strong> <span class="{env_info.get('pandas_class', 'warning')}">{env_info.get('pandas_status', 'Not checked')}</span></p>
                                <p><strong>SQLAlchemy:</strong> <span class="{env_info.get('sqlalchemy_class', 'warning')}">{env_info.get('sqlalchemy_status', 'Not checked')}</span></p>
                                <p><strong>Requests:</strong> <span class="{env_info.get('requests_class', 'warning')}">{env_info.get('requests_status', 'Not checked')}</span></p>
                                <p><strong>Psycopg2:</strong> <span class="{env_info.get('psycopg2_class', 'warning')}">{env_info.get('psycopg2_status', 'Not checked')}</span></p>
                                <p><strong>FastAPI:</strong> <span class="{env_info.get('fastapi_class', 'warning')}">{env_info.get('fastapi_status', 'Not checked')}</span></p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
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
            # Check if it's a valid PostgreSQL URL
            if db_url.startswith('postgresql://'):
                info['db_url_status'] = "Set (valid PostgreSQL URL)"
                info['db_url_class'] = "success"
            else:
                info['db_url_status'] = "Set, but may be invalid (doesn't start with postgresql://)"
                info['db_url_class'] = "warning"
        else:
            info['db_url_status'] = "Not set (required for database access)"
            info['db_url_class'] = "error"
        
        # Check database connection params
        db_params = ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE']
        all_pg_params_present = True
        
        for param in db_params:
            value = os.environ.get(param)
            param_key = param.lower()
            
            if value:
                if param != 'PGPASSWORD':
                    info[f'{param_key}_status'] = value
                else:
                    info[f'{param_key}_status'] = "Set (value hidden)"
                info[f'{param_key}_class'] = "success"
            else:
                info[f'{param_key}_status'] = "Not set"
                info[f'{param_key}_class'] = "error"
                all_pg_params_present = False
        
        # Add a summary for database connection
        if db_url or all_pg_params_present:
            info['db_conn_summary'] = "Database connection parameters available"
            info['db_conn_class'] = "success"
        else:
            info['db_conn_summary'] = "Missing required database connection parameters"
            info['db_conn_class'] = "error"
        
        # Check SESSION_SECRET
        session_secret = os.environ.get('SESSION_SECRET')
        if session_secret:
            if session_secret == "bcbs_values_session_secret_key_2025":
                info['session_secret_status'] = "Default value (should be changed in production)"
                info['session_secret_class'] = "warning"
            else:
                info['session_secret_status'] = "Set (value hidden)"
                info['session_secret_class'] = "success"
        else:
            info['session_secret_status'] = "Not set (required for Flask sessions)"
            info['session_secret_class'] = "error"
        
        # Check API_KEY
        api_key = os.environ.get('API_KEY') or os.environ.get('BCBS_VALUES_API_KEY')
        if api_key:
            if api_key == "bcbs_values_api_key_2025":
                info['api_key_status'] = "Default value (should be changed in production)"
                info['api_key_class'] = "warning"
            else:
                info['api_key_status'] = "Set (value hidden)"
                info['api_key_class'] = "success"
        else:
            info['api_key_status'] = "Not set (required for API authentication)"
            info['api_key_class'] = "error"
        
        # Check Python modules
        module_info = [
            {"name": "pandas", "required_for": "ETL and data processing", "attr": "pandas"},
            {"name": "flask", "required_for": "web application", "attr": "flask"},
            {"name": "sqlalchemy", "required_for": "database access", "attr": "sqlalchemy"},
            {"name": "requests", "required_for": "API requests", "attr": "requests"},
            {"name": "psycopg2", "required_for": "PostgreSQL connectivity", "attr": "psycopg2"},
            {"name": "fastapi", "required_for": "API development", "attr": "fastapi"}
        ]
        
        for module in module_info:
            module_name = module["name"]
            module_key = module_name.lower()
            required_for = module["required_for"]
            attr_name = module.get("attr", module_name)
            
            try:
                __import__(attr_name)
                mod = sys.modules[attr_name]
                version = getattr(mod, "__version__", "unknown version")
                info[f'{module_key}_status'] = f"Installed (version {version})"
                info[f'{module_key}_class'] = "success"
            except ImportError:
                info[f'{module_key}_status'] = f"Not installed (required for {required_for})"
                info[f'{module_key}_class'] = "warning"
            except Exception as e:
                info[f'{module_key}_status'] = f"Error checking ({str(e)})"
                info[f'{module_key}_class'] = "warning"
        
        # Add Python executable path
        info['python_path'] = sys.executable
        
        return info

def run_server(port=5000):
    """Run the diagnostic server"""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, DiagnosticHandler)
    print(f"Starting diagnostic server on port {port}...")
    print(f"Python version: {sys.version}")
    print(f"Server address: http://0.0.0.0:{port}/")
    print(f"You can access the server at the URL in your browser.")
    try:
        httpd.serve_forever()
    except Exception as e:
        print(f"Server error: {e}")
        raise

if __name__ == '__main__':
    try:
        # Print detailed debugging information
        print("=" * 50)
        print("DIAGNOSTIC SERVER STARTUP")
        print("=" * 50)
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Path: {os.environ.get('PATH', 'Not set')}")
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        print(f"Database URL: {'Set (hidden)' if os.environ.get('DATABASE_URL') else 'Not set'}")
        
        # Print loaded modules for debugging
        print("\nLoaded modules:")
        for i, module_name in enumerate(sorted(sys.modules.keys())[:20], 1):  # Show only first 20
            print(f"  {i}. {module_name}")
        if len(sys.modules) > 20:
            print(f"  ... and {len(sys.modules) - 20} more")
        
        print("\nStarting server...")
        port = int(os.environ.get('PORT', 5000))
        run_server(port)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)