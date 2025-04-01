#!/bin/bash

# Direct script to run the server without relying on workflow configuration

echo "Running direct server script..."
PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

if [ -x "$PYTHON_PATH" ]; then
  echo "Python found at $PYTHON_PATH"
  if [ -f "simple_python_server.py" ]; then
    echo "Found simple_python_server.py, executing directly..."
    exec "$PYTHON_PATH" simple_python_server.py
  else
    echo "simple_python_server.py not found"
    exit 1
  fi
else
  echo "Python not found at expected path: $PYTHON_PATH"
  exit 1
fi