#!/bin/bash
# Simple HTTP server using Busybox httpd

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Set server port
PORT=5002
DOCUMENT_ROOT="."

log "Starting Busybox HTTP server on port $PORT"
log "Serving content from $DOCUMENT_ROOT"

# Check for busybox
if command -v busybox >/dev/null 2>&1; then
  # Create a temporary directory for httpd configuration
  TEMP_DIR=$(mktemp -d)
  echo "Setting up httpd in $TEMP_DIR"
  
  # Create a simple index.html symlink to our dashboard
  ln -sf "$PWD/dashboard_static.html" "$TEMP_DIR/index.html"
  
  # Copy all necessary files to the temp directory
  cp -r "$PWD"/* "$TEMP_DIR/"
  
  # Change to temp directory
  cd "$TEMP_DIR" || exit 1
  
  # Start the httpd server
  busybox httpd -f -p "$PORT" -h "$TEMP_DIR"
else
  log "Error: Busybox not found"
  exit 1
fi