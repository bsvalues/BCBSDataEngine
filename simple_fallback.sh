#!/bin/bash

# Simple Fallback Diagnostic Server
# This script provides a basic diagnostic server using only bash capabilities

PORT=${PORT:-5000}
echo "Starting simple fallback diagnostic server on port $PORT"
echo "Server time: $(date)"

# Generate basic HTML content
HTML_CONTENT="
<!DOCTYPE html>
<html>
<head>
    <title>BCBS Simple Diagnostic</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #0066cc; margin-top: 30px; }
        pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; font-weight: bold; }
        .warning { background-color: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <h1>BCBS Values Simple Diagnostic</h1>
    
    <div class='card'>
        <h2>System Status</h2>
        <p><span class='status warning'>BASIC FALLBACK MODE</span> Running in minimal bash-based diagnostic mode.</p>
        <p>The application server could not be started with Python or Node.js.</p>
    </div>
    
    <div class='card'>
        <h2>System Information</h2>
        <pre>
Date: $(date)
Hostname: $(hostname 2>/dev/null || echo 'Unknown')
User: $(whoami 2>/dev/null || echo 'Unknown')
Path: $PATH
PWD: $PWD
</pre>
    </div>
    
    <div class='card'>
        <h2>Environment</h2>
        <pre>
PORT: $PORT
PYTHONPATH: $PYTHONPATH
NODE_PATH: $NODE_PATH
</pre>
    </div>
    
    <div class='card'>
        <h2>System Resources</h2>
        <pre>
$(free -h 2>/dev/null || echo 'Memory info not available')

$(df -h . 2>/dev/null || echo 'Disk info not available')
</pre>
    </div>
    
    <div class='card'>
        <h2>Available Tools</h2>
        <pre>
Python: $(which python3 2>/dev/null || which python 2>/dev/null || echo 'Not found')
Node.js: $(which node 2>/dev/null || echo 'Not found')
Bash: $(which bash 2>/dev/null || echo 'Not found')
</pre>
    </div>
    
    <div class='card'>
        <h2>File System</h2>
        <pre>
Current directory files:
$(ls -la | head -20)
</pre>
    </div>
    
    <div class='card'>
        <h2>Environment Variables</h2>
        <pre>
PATH: $PATH
</pre>
    </div>
    
    <div class='card'>
        <h2>Help Information</h2>
        <p>The server is currently running in fallback mode because neither Python nor Node.js could be started.</p>
        <p>To enable the full application, please ensure that Python 3.11 or Node.js is properly installed and accessible in the PATH.</p>
    </div>
    
    <footer style='margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;'>
        <p>BCBS Values Simple Fallback Diagnostic Server</p>
        <p>Generated: $(date)</p>
    </footer>
</body>
</html>
"

# Check if netcat is available to create a simple web server
if command -v nc >/dev/null 2>&1; then
    echo "Using netcat to serve diagnostic page on port $PORT"
    
    # Loop to keep the server running after each connection
    while true; do
        echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n$HTML_CONTENT" | nc -l -p $PORT || break
        echo "Connection served. Restarting server..."
    done
else
    # Fallback to just showing the HTML
    echo "Netcat not available. Cannot start server."
    echo "Diagnostic information:"
    echo "------------------------------------------------"
    echo "Date: $(date)"
    echo "User: $(whoami 2>/dev/null || echo 'Unknown')"
    echo "Directory: $PWD"
    echo "PATH: $PATH"
    echo "Displaying HTML content for manual inspection:"
    echo "$HTML_CONTENT"
fi