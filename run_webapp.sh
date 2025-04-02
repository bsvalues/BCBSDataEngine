#!/bin/bash
# Improved workflow startup script for BCBS Values Platform
# This script handles Python detection and provides better error handling

# Set up logging with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting BCBS Values Platform Web Application"
log "=============================================="

# Function to find a working Python executable
find_python() {
    log "Searching for Python executable..."
    
    # List of potential Python paths to try - updated order for efficiency
    PYTHON_PATHS=(
        "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"
        "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
        "python3"
        "python"
        "nix-shell -p python3 --run python3"
    )
    
    for path in "${PYTHON_PATHS[@]}"; do
        log "Trying Python path: $path"
        if [[ $path == nix-shell* ]]; then
            # Special case for nix-shell
            eval "$path --version" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                log "Found working Python using nix-shell"
                echo "$path"
                return 0
            fi
        elif [ -x "$path" ]; then
            # Check if the path exists and is executable
            $path --version > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                log "Found working Python at: $path"
                echo "$path"
                return 0
            fi
        fi
    done
    
    log "ERROR: Could not find a working Python executable"
    return 1
}

# Function to check if server is running
check_server() {
    local port=$1
    local timeout=$2
    local start_time=$(date +%s)
    local current_time
    
    log "Waiting for server to start on port $port (timeout: ${timeout}s)..."
    
    while true; do
        # Try to connect to the port (simple connection test)
        (echo > /dev/tcp/localhost/$port) 2>/dev/null
        if [ $? -eq 0 ]; then
            log "Server is running on port $port"
            return 0
        fi
        
        # Check timeout
        current_time=$(date +%s)
        if [ $((current_time - start_time)) -gt $timeout ]; then
            log "ERROR: Server startup timed out after ${timeout}s"
            return 1
        fi
        
        # Check if process is still running
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR: Server process is no longer running"
            return 1
        fi
        
        # Sleep for a short time before checking again
        sleep 1
    done
}

# Main execution
log "Checking environment..."
log "Current directory: $(pwd)"
log "User: $(whoami)"

# Find Python executable
PYTHON_EXEC=$(find_python)
if [ $? -ne 0 ]; then
    log "ERROR: Could not find a working Python executable. Creating fallback server..."
    
    # Create a minimal Node.js server as a fallback
    cat > minimal_server.js << 'EOF'
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5002;

const server = http.createServer((req, res) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    
    if (req.url === '/' || req.url === '/index.html') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>BCBS Values Platform</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    h1 { color: #2c3e50; }
                    .alert { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>BCBS Values Platform</h1>
                <div class="alert">
                    <strong>Warning:</strong> Running in fallback mode due to Python environment issues.
                </div>
                <p>This is a minimal server for the BCBS Values Platform.</p>
                <p>Server time: ${new Date().toISOString()}</p>
                <p>Requested URL: ${req.url}</p>
            </body>
            </html>
        `);
    } else if (req.url.startsWith('/api/')) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            status: "success",
            message: "Fallback server API",
            path: req.url,
            time: new Date().toISOString()
        }));
    } else {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>404 Not Found</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    h1 { color: #2c3e50; }
                </style>
            </head>
            <body>
                <h1>404 Not Found</h1>
                <p>The requested URL ${req.url} was not found on this server.</p>
                <p><a href="/">Go to homepage</a></p>
            </body>
            </html>
        `);
    }
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`Fallback server running at http://0.0.0.0:${PORT}/`);
});
EOF
    
    log "Starting fallback Node.js server..."
    node minimal_server.js &
    SERVER_PID=$!
    
    # Check if server started successfully
    check_server 5002 10
    if [ $? -ne 0 ]; then
        log "ERROR: Fallback server failed to start"
        exit 1
    fi
    
    log "Fallback server is running at http://localhost:5002/"
    wait $SERVER_PID
    exit 0
fi

log "Using Python executable: $PYTHON_EXEC"

# Choose the right script to run based on availability
if [ -f "simple_http_server.py" ]; then
    log "Found simple_http_server.py, using this for better compatibility"
    SERVER_SCRIPT="simple_http_server.py"
elif [ -f "app.py" ]; then
    log "Using app.py as server script"
    SERVER_SCRIPT="app.py"
elif [ -f "start_webapp.py" ]; then
    log "Using start_webapp.py as server script"
    SERVER_SCRIPT="start_webapp.py"
else
    log "ERROR: No suitable server script found. Exiting."
    exit 1
fi

# Make server script executable if it isn't already
chmod +x $SERVER_SCRIPT

# Run the server
log "Starting the web application using $SERVER_SCRIPT..."

# Run the server with the found Python executable
$PYTHON_EXEC $SERVER_SCRIPT > server.log 2>&1 &
SERVER_PID=$!

# Wait for the server to start
check_server 5002 30
if [ $? -ne 0 ]; then
    log "ERROR: Server failed to start properly"
    # Try to kill the process if it's still running
    kill $SERVER_PID > /dev/null 2>&1
    
    # Show logs
    log "Server log output:"
    cat server.log
    
    exit 1
fi

log "Server is running successfully"
log "You can access the web interface at: http://localhost:5002/"

# Keep script running to maintain the server process
log "Press Ctrl+C to stop the server"
wait $SERVER_PID