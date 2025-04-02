#!/bin/bash
# Simple bash script to run a basic HTTP server for the BCBS dashboard

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Set server port
PORT=5002
DOCUMENT_ROOT="."
DEFAULT_FILE="dashboard_static.html"

log "Starting simple Bash HTTP server on port $PORT"
log "Serving content from $DOCUMENT_ROOT"
log "Default file: $DEFAULT_FILE"

# Use built-in shell capabilities for a simple HTTP server
while true; do
  # Listen on the port using /dev/tcp
  { echo -ne "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"; cat "$DOCUMENT_ROOT/$DEFAULT_FILE"; } | nc -l -p $PORT || log "Error: Failed to start server, retrying in 5 seconds..."
  
  # Wait a bit before restarting if there's an error
  sleep 5
done