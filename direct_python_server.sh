#!/bin/bash
# Direct Python server script for BCBS Values Platform
# This script uses a hardcoded Python path for maximum reliability

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting BCBS Values Platform Server"
log "===================================="

# Direct path to Python
PYTHON="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

# Check if Python exists
if [ ! -x "$PYTHON" ]; then
    log "ERROR: Python not found at $PYTHON"
    log "Falling back to Node.js server"
    
    # Create a minimal Node.js server as fallback
    cat > minimal_server.js << 'EOT'
const http = require('http');
const fs = require('fs');
const PORT = 5002;

const server = http.createServer((req, res) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.url}`);
    
    if (req.url === '/' || req.url === '/index.html') {
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.end(`<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .alert { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    <div class="alert">
        <strong>Note:</strong> Running in fallback mode
    </div>
    <p>Server is running in fallback mode using Node.js</p>
    <p>Current time: ${new Date().toISOString()}</p>
</body>
</html>`);
    } else if (req.url.startsWith('/api/')) {
        res.writeHead(200, {'Content-Type': 'application/json'});
        res.end(JSON.stringify({
            status: "success",
            message: "Fallback API response",
            endpoint: req.url.split('/')[2] || '',
            timestamp: new Date().toISOString()
        }));
    } else {
        res.writeHead(404, {'Content-Type': 'text/html'});
        res.end('404 Not Found');
    }
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`[${new Date().toISOString()}] Fallback server running at http://0.0.0.0:${PORT}/`);
});
EOT
    
    # Run the Node.js server
    log "Starting Node.js fallback server"
    exec node minimal_server.js
    exit 1
fi

# Server script to run
log "Python executable found: $PYTHON"

# Create a simple server Python script
cat > temp_server.py << 'EOT'
#!/usr/bin/env python3
"""
Simple HTTP server for BCBS Values Platform
"""
import os
import sys
import http.server
import socketserver
import datetime
import json

PORT = 5002
HOST = "0.0.0.0"

def log(message):
    """Print a log message with timestamp"""
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] {message}")

class Handler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        log(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    <div class="success">
        <strong>Success!</strong> Server is running correctly
    </div>
    <p>Server is running with Python {sys.version}</p>
    <p>Current time: {datetime.datetime.now().isoformat()}</p>
    <p>Available HTML files:</p>
    <ul>
"""
            
            # List available HTML files
            html_files = [f for f in os.listdir('.') if f.endswith('.html')]
            for file in html_files:
                html += f"        <li><a href='/{file}'>{file}</a></li>\n"
            
            html += """    </ul>
    <p>API Endpoints:</p>
    <ul>
        <li><a href='/api/status'>/api/status</a></li>
        <li><a href='/api/agent-status'>/api/agent-status</a></li>
    </ul>
</body>
</html>"""
            
            self.wfile.write(html.encode())
            return
            
        elif self.path.startswith('/api/'):
            endpoint = self.path.split('/')[2] if len(self.path.split('/')) > 2 else ''
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "status": "success",
                "message": f"API response for: {endpoint}",
                "server": "Python Direct Server",
                "python_version": sys.version,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.wfile.write(json.dumps(response).encode())
            return
            
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def run_server():
    """Run the HTTP server"""
    log(f"Starting server on {HOST}:{PORT}")
    log(f"Python version: {sys.version}")
    log(f"Current directory: {os.getcwd()}")
    
    # List HTML files
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    if html_files:
        log(f"Available HTML files: {', '.join(html_files)}")
    else:
        log("No HTML files found")
    
    try:
        with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
            log(f"Server running at http://{HOST}:{PORT}/")
            httpd.serve_forever()
    except KeyboardInterrupt:
        log("Server stopped by user")
    except Exception as e:
        log(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()
EOT

# Run the Python server
log "Starting Python server"
exec $PYTHON temp_server.py