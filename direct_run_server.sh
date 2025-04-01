#!/bin/bash

# Direct script to run the server with the specific Python path

PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

if [ -x "$PYTHON_PATH" ]; then
  echo "Python found at $PYTHON_PATH"
  
  if [ -f "start_webapp.py" ]; then
    echo "Found start_webapp.py, executing directly..."
    exec "$PYTHON_PATH" start_webapp.py
  else
    echo "start_webapp.py not found"
    exit 1
  fi
else
  echo "Python not found at expected path: $PYTHON_PATH"
  exit 1
fi