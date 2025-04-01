#!/usr/bin/env python3
"""
Ultra-minimal diagnostic server for BCBS Values using Python's built-in HTTP server
"""

import http.server
import socketserver
import os
import socket
import platform
import datetime
import json
from urllib.parse import parse_qs, urlparse

# Set the port (use PORT environment variable or default to 5000)
PORT = int(os.environ.get('PORT', 5000))

class DiagnosticHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for diagnostic information"""
    
    def generate_html(self):
        """Generate HTML for the diagnostic page"""
        
        # Basic system info
        now = datetime.datetime.now().isoformat()
        hostname = socket.gethostname()
        python_version = platform.python_version()
        system_info = f"{platform.system()} {platform.release()}"
        
        # Try to get memory info
        memory_info = "Not available"
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = f"Total: {memory.total / (1024*1024):.2f} MB, Available: {memory.available / (1024*1024):.2f} MB"
        except ImportError:
            pass
            
        # Get directory info
        current_dir = os.getcwd()
        try:
            files = ", ".join(os.listdir(".")[:20])
            if len(os.listdir(".")) > 20:
                files += "... (truncated)"
        except Exception as e:
            files = f"Error listing files: {e}"
            
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BCBS Diagnostic</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
                h1 {{ color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }}
                h2 {{ color: #0066cc; margin-top: 30px; }}
                pre {{ background: #f5f5f5; padding: 15px; overflow-x: auto; }}
                .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>BCBS Values Diagnostic Server</h1>
            
            <div class="card">
                <h2>System Information</h2>
                <pre>
Time: {now}
Hostname: {hostname}
Platform: {system_info}
Python Version: {python_version}
Memory: {memory_info}
                </pre>
            </div>
            
            <div class="card">
                <h2>File System Access</h2>
                <pre>
Current Directory: {current_dir}
Files: {files}
                </pre>
            </div>
            
            <div class="card">
                <h2>Environment Variables</h2>
                <pre>
PORT: {os.environ.get('PORT', 'Not set')}
PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}
                </pre>
            </div>
            
            <div class="card">
                <h2>Next Steps</h2>
                <p>This is a minimal diagnostic server to confirm that Python is working in this environment.</p>
                <p>For a more complete diagnostic, the full application needs to be properly installed.</p>
            </div>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
                <p>BCBS Values Minimal Diagnostic Server</p>
                <p>Generated: {now}</p>
            </footer>
        </body>
        </html>
        """
        return html
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        # API endpoint for health check
        if parsed_path.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                'status': 'python_diagnostic_mode',
                'message': 'Running in minimal Python diagnostic mode',
                'timestamp': datetime.datetime.now().isoformat(),
                'python_version': platform.python_version(),
                'system': f"{platform.system()} {platform.release()}"
            }
            
            self.wfile.write(json.dumps(health_data).encode())
            return
            
        # Default: serve HTML diagnostic page
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.generate_html().encode())

def run_server():
    """Start the diagnostic server"""
    print(f"Starting minimal diagnostic server on port {PORT}")
    print(f"Server time: {datetime.datetime.now().isoformat()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Open http://localhost:{PORT}/ to view diagnostic information")
    
    with socketserver.TCPServer(("0.0.0.0", PORT), DiagnosticHandler) as httpd:
        print("Server started. Press Ctrl+C to stop.")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()