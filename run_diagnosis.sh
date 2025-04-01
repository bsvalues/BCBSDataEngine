#!/bin/bash

# BCBS Values Platform Diagnosis Script
# This script attempts to start the server using multiple fallback options

echo "=== BCBS Values Platform Diagnosis ==="
echo "Starting diagnostic process..."

# Function to log messages with timestamps
log_message() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if we're in the root directory
if [ ! -f "index.html" ]; then
  log_message "Warning: index.html not found in current directory"
  log_message "This script should be run from the project root"
else
  log_message "Found index.html in current directory"
fi

# Use direct Python path that we know exists
PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

log_message "Attempting to start Python fallback server using direct path..."
if [ -x "$PYTHON_PATH" ]; then
  log_message "Python found at $PYTHON_PATH"
  if [ -f "simple_python_server.py" ]; then
    log_message "Found simple_python_server.py, executing with $PYTHON_PATH..."
    "$PYTHON_PATH" simple_python_server.py
    exit $?
  else
    log_message "simple_python_server.py not found in current directory, checking..."
    ls -la
  fi
else
  log_message "Python not found at expected path: $PYTHON_PATH"
  log_message "Checking directory..."
  ls -la /mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/
fi

# If Python server fails, try direct Node.js path
NODE_PATH="/mnt/nixmodules/nix/store/7f3s06fx9khz3pyipkgj46j6p8mrbmcn-nodejs-18.19.1/bin/node"
if [ -x "$NODE_PATH" ]; then
  log_message "Found Node.js at $NODE_PATH"
  if [ -f "basic_server.js" ]; then
    log_message "Found basic_server.js, executing with direct Node.js path..."
    "$NODE_PATH" basic_server.js
    exit $?
  else
    log_message "basic_server.js not found, trying alternate methods"
    ls -la
  fi
else
  log_message "Node.js not found at expected path: $NODE_PATH"
fi

# Final fallback - echo a message to guide the user
log_message "All server startup methods failed"
log_message "Please try manually accessing static_fallback.html in your browser"
echo ""
echo "===== SERVER STARTUP FAILED ====="
echo "Try navigating to static_fallback.html in your browser"
echo "===== SERVER STARTUP FAILED ====="

exit 1