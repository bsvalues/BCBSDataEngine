#!/bin/bash

# BCBS Values Node.js Diagnostic Server Script
# This script runs the direct Node.js diagnostic server

echo "========================================================"
echo "BCBS VALUES NODE.JS DIAGNOSTIC SERVER"
echo "========================================================"
echo "Timestamp: $(date)"

# Check if Node.js is available
if command -v node > /dev/null 2>&1; then
  echo "Found Node.js: $(node --version)"
  echo "Starting Node.js diagnostic server..."
  
  if [ -f "direct_server.js" ]; then
    echo "Running direct_server.js on port ${PORT:-5000}..."
    node direct_server.js
    exit $?
  else
    echo "Error: direct_server.js not found!"
    exit 1
  fi
else
  echo "Error: Node.js is not available in this environment."
  echo "PATH: $PATH"
  exit 1
fi