#!/usr/bin/env python3
"""
Simplified diagnostic server that provides basic environment information
"""
import os
import sys
import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        env_vars = {}
        for key in sorted(os.environ.keys()):
            if key in ['DATABASE_URL', 'PGPASSWORD', 'SESSION_SECRET', 'API_KEY']:
                env_vars[key] = 'Set (value hidden)'
            elif key.startswith('PG') or key in ['PYTHONPATH', 'PATH', 'PORT']:
                env_vars[key] = os.environ[key]
                
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Diagnostic</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #3366cc; }}
                .info {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>BCBS Values Simple Diagnostic</h1>
            <div class="info">
                <p><strong>Current Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Python Version:</strong> {sys.version.split()[0]}</p>
                <p><strong>Python Path:</strong> {sys.executable}</p>
                <p><strong>Current Directory:</strong> {os.getcwd()}</p>
            </div>
            
            <h2>Environment Variables</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Value</th>
                </tr>
        """
        
        for key, value in env_vars.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            
        html += """
            </table>
            <div style="margin-top: 20px">
                <p><strong>Note:</strong> This is a simplified diagnostic server for troubleshooting.</p>
                <p>To run the full diagnostic server, use <code>python quick_diagnostic_server.py</code>.</p>
            </div>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())

def run_server(port=5000):
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    print(f"Starting simple diagnostic server on port {port}")
    print(f"Open http://localhost:{port}/ in your browser")
    server.serve_forever()

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        run_server(port)
    except KeyboardInterrupt:
        print("Server stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)