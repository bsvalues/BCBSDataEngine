#!/bin/bash

echo "Starting BCBS Values Platform with Python fallback server..."
echo "Running simple_python_server.py..."

# Try to locate Python
PYTHON_PATHS=(
  "python3"
  "python"
  "/usr/bin/python3"
  "/usr/local/bin/python3"
  "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"
)

for PYTHON_PATH in "${PYTHON_PATHS[@]}"; do
  if command -v "$PYTHON_PATH" > /dev/null 2>&1 || [ -x "$PYTHON_PATH" ]; then
    echo "Using Python at $PYTHON_PATH"
    "$PYTHON_PATH" simple_python_server.py
    exit $?
  fi
done

echo "Python not found in any of the expected locations. Cannot start server."
echo "Trying to use static HTML fallback..."
echo "Please navigate to static_fallback.html in your browser."
exit 1