#!/bin/bash
# Node.js Server Runner for Workflow Reliability
# This script locates and runs node with direct_server.js

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Banner
log "=========================================="
log "BCBS VALUES PLATFORM NODE.JS SERVER"
log "=========================================="

# Find Node.js executable
NODE_PATHS=(
    "/usr/bin/node"
    "/usr/local/bin/node"
    "node"
)

for NODE_PATH in "${NODE_PATHS[@]}"; do
    if command -v "$NODE_PATH" &> /dev/null; then
        log "Found Node.js at $NODE_PATH"
        
        # Check if direct_server.js exists
        if [ -f "direct_server.js" ]; then
            log "Starting server with direct_server.js"
            exec "$NODE_PATH" direct_server.js
        else
            log "ERROR: direct_server.js not found!"
            exit 1
        fi
    fi
done

# If we get here, Node.js is not available
log "ERROR: Node.js not found!"

# Try bash-based server as fallback
if [ -f "bash_http_server.sh" ] && [ -x "bash_http_server.sh" ]; then
    log "Falling back to bash_http_server.sh"
    exec ./bash_http_server.sh
else
    log "ERROR: No server alternatives available!"
    exit 1
fi