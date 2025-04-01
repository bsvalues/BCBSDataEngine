#!/bin/bash

# BCBS Values Direct Python Server Script
# This script runs the direct diagnostic server using a specific Python path

echo "========================================================"
echo "BCBS VALUES DIRECT PYTHON SERVER"
echo "========================================================"
echo "Timestamp: $(date)"

# Use the absolute path to Python 3.11
PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

if [ -x "$PYTHON_PATH" ] && [ -f "direct_diagnostic.py" ]; then
  echo "Found Python 3.11 at $PYTHON_PATH"
  echo "Starting direct diagnostic server..."
  chmod +x direct_diagnostic.py
  $PYTHON_PATH direct_diagnostic.py
  exit $?
else
  echo "Error: Either Python 3.11 was not found at $PYTHON_PATH or direct_diagnostic.py doesn't exist."
  echo "Python path exists: $([ -x "$PYTHON_PATH" ] && echo "Yes" || echo "No")"
  echo "direct_diagnostic.py exists: $([ -f "direct_diagnostic.py" ] && echo "Yes" || echo "No")"
  exit 1
fi