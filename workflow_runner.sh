#!/usr/bin/env bash
# This script is a specialized entry point for the Replit Workflow runner
# It attempts to locate either Python or netcat and use them to start a simple HTTP server

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Start banner
log "Starting BCBS Values Platform Workflow Server"
log "==========================================="

# Look for Python executable in common locations
PYTHON_PATHS=(
    "/usr/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python"
    "/usr/local/bin/python"
    "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"
)

for PYTHON_PATH in "${PYTHON_PATHS[@]}"; do
    if [ -x "$PYTHON_PATH" ]; then
        log "Found Python at $PYTHON_PATH"
        if [ -f "simple_http_server.py" ]; then
            log "Starting server with simple_http_server.py"
            exec "$PYTHON_PATH" simple_http_server.py
        else
            log "Starting server with built-in HTTP server module"
            exec "$PYTHON_PATH" -m http.server 5002 --bind 0.0.0.0
        fi
    fi
done

# If we get here, Python couldn't be found or started
log "Python not found, trying to start HTTP server with Bash..."

# Check for Bash HTTP server script
if [ -f "bash_http_server.sh" ] && [ -x "bash_http_server.sh" ]; then
    log "Found bash_http_server.sh, executing..."
    exec ./bash_http_server.sh
fi

# If we get here, all attempts failed
log "ERROR: All server startup methods failed!"
log "Working directory contents:"
ls -la

# Create error HTML as a last resort
cat > error.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform - Server Error</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #721c24; }
        .card { border: 1px solid #f5c6cb; border-radius: 8px; padding: 20px; background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>Server Error</h1>
    <div class="card">
        <p>The BCBS Values Platform server failed to start properly.</p>
        <p>Please check the server logs for more information.</p>
        <p>Time of error: <script>document.write(new Date().toLocaleString())</script></p>
    </div>
</body>
</html>
EOF

log "Server failed to start. Error page created."
exit 1