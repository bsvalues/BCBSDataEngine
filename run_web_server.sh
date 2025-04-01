#!/bin/bash

echo "Starting BCBS Values Platform Web Server..."

# Try to find Python in different locations
PYTHON_PATHS=(
  "/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
  "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"
  "/usr/bin/python3"
  "/usr/local/bin/python3"
  "python3"
)

# Find the first working Python
PYTHON_PATH=""
for path in "${PYTHON_PATHS[@]}"; do
  if command -v "$path" > /dev/null 2>&1; then
    echo "Found Python at: $path"
    PYTHON_PATH="$path"
    break
  fi
done

if [ -z "$PYTHON_PATH" ]; then
  echo "Error: Could not find Python in any of the expected locations."
  echo "Creating a very simple Node.js fallback server instead."
  
  # Create a simple Node.js server as fallback
  cat > simple_node_server.js << 'EOF'
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Convert URL to file path
  let filePath = '.' + req.url;
  if (filePath === './') {
    filePath = './index.html';
  }
  
  // Get file extension
  const extname = path.extname(filePath);
  let contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Read the file
  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // Page not found - try to serve static_fallback.html
        fs.readFile('./static_fallback.html', (err, content) => {
          if (err) {
            res.writeHead(404);
            res.end('Page not found');
          } else {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(content, 'utf-8');
          }
        });
      } else {
        // Server error
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      }
    } else {
      // Success
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`BCBS Values Platform Node.js Fallback Server running at http://0.0.0.0:${PORT}/`);
  console.log('Available pages:');
  console.log(`- Home: http://0.0.0.0:${PORT}/`);
  console.log(`- Dashboard: http://0.0.0.0:${PORT}/dashboard.html`);
  console.log(`- Static Fallback: http://0.0.0.0:${PORT}/static_fallback.html`);
});
EOF
  
  # Try to run the Node.js server
  if command -v node > /dev/null 2>&1; then
    echo "Starting Node.js fallback server..."
    exec node simple_node_server.js
  else
    echo "Error: Node.js not found. Cannot start any server."
    exit 1
  fi
else
  # Create a simple Python server script
  cat > simple_bcbs_server.py << 'EOF'
#!/usr/bin/env python3
"""
Simple HTTP server to serve static files for BCBS Values Platform
"""

import http.server
import socketserver
import os
import sys

# Set the port
PORT = 5000

class BCBSHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for BCBS Values Platform"""
    
    def log_message(self, format, *args):
        """Override to provide cleaner logging"""
        print("[%s] %s" % (self.log_date_time_string(), format % args))

# Create the server
Handler = BCBSHandler
httpd = socketserver.TCPServer(("0.0.0.0", PORT), Handler)

print(f"BCBS Values Platform Python Server running at http://0.0.0.0:{PORT}/")
print("Available pages:")
print(f"- Home: http://0.0.0.0:{PORT}/")
print(f"- Dashboard: http://0.0.0.0:{PORT}/dashboard.html")
print(f"- Static Fallback: http://0.0.0.0:{PORT}/static_fallback.html")

# Start the server
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down server...")
    httpd.server_close()
    sys.exit(0)
EOF
  
  # Make it executable
  chmod +x simple_bcbs_server.py
  
  # Run the server
  echo "Starting Python web server..."
  exec "$PYTHON_PATH" simple_bcbs_server.py
fi