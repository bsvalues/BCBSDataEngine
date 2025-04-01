#!/bin/bash

# Simple script to run the Python server with the correct Python binary path
PYTHON_PATH="/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"

if [ -x "$PYTHON_PATH" ]; then
  echo "Python found at $PYTHON_PATH"
  
  # Create a very basic script to serve the files
  cat > simple_python_server.py << 'EOF'
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

print(f"Serving BCBS Values Platform at http://0.0.0.0:{PORT}/")
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
  chmod +x simple_python_server.py
  
  # Run the server
  echo "Starting Python HTTP server..."
  exec "$PYTHON_PATH" simple_python_server.py
else
  echo "Python not found at expected path: $PYTHON_PATH"
  exit 1
fi