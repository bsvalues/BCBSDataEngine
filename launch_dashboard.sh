#!/bin/bash
# Dashboard launch script that runs in the foreground

echo "Starting BCBS Dashboard Server..."

# Find the Python path
PYTHON_PATH="/mnt/nixmodules/nix/store/83mqxa6p0r3f1fh6f70737ifah5ik7qp-python-wrapped-0.1.0/bin/python3.10"

echo "Using Python at: $PYTHON_PATH"

# Use a super simple approach with Python's built-in HTTP server module
echo "Starting HTTP server on port: 8080"
echo "Press Ctrl+C to stop the server"

# Run the HTTP server in the foreground
exec $PYTHON_PATH -m http.server 8080