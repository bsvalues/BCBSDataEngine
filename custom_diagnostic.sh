#!/bin/bash

# BCBS Values Custom Diagnostic Script - Ultra-Simplified Version
# This script directly runs the Node.js server with minimal checks

echo "=================================================="
echo "BCBS VALUES CUSTOM DIAGNOSTIC - SIMPLIFIED"
echo "=================================================="
echo "Started at: $(date)"
echo ""

# Set default port
export PORT=5000

# Direct execution strategy - try each option in sequence
# with minimal checks to reduce possible points of failure

echo "Attempting to start server with Node.js..."
if [ -f "simple_http_server.js" ] && command -v node >/dev/null 2>&1; then
  echo "Found simple_http_server.js and node is available."
  echo "Starting server at http://0.0.0.0:$PORT/"
  echo "=================================================="
  node simple_http_server.js
  exit $?
fi

echo "Node.js server failed to start. Trying bash-based server..."
if [ -f "simple_bash_server.sh" ]; then
  echo "Found simple_bash_server.sh"
  echo "Starting server at http://0.0.0.0:$PORT/"
  echo "=================================================="
  bash ./simple_bash_server.sh
  exit $?
fi

echo "Bash server failed to start. Trying static HTML with netcat..."
if [ -f "index.html" ] && command -v nc >/dev/null 2>&1; then
  echo "Found index.html and netcat is available."
  echo "Starting static server at http://0.0.0.0:$PORT/"
  echo "=================================================="
  echo "Serving static index.html with netcat"
  echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n$(cat index.html)" | nc -l -p $PORT
  exit $?
fi

# Ultimate fallback - create an inline HTML page
echo "All server options failed. Creating emergency HTML response..."
HTML="<!DOCTYPE html>
<html>
<head>
  <title>BCBS Values Emergency Page</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
    h1 { color: #d9534f; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    pre { background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>BCBS Values - Emergency Diagnostic Page</h1>
  
  <div class='card'>
    <p>This is an emergency diagnostic page created when all other server options failed.</p>
    <p>Generated at: $(date)</p>
  </div>
  
  <div class='card'>
    <h2>Environment Information</h2>
    <pre>
Working Directory: $(pwd)
Date: $(date)
Hostname: $(hostname)
    </pre>
  </div>
</body>
</html>"

echo "Starting emergency server at http://0.0.0.0:$PORT/"
echo "=================================================="
echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n$HTML" | nc -l -p $PORT || \
echo "Failed to start netcat server. All diagnostic options exhausted."

exit 1