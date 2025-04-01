#!/bin/bash

# BCBS Values Platform Dashboard Server
echo "BCBS Values Platform Dashboard Server"
echo "===================================="
echo "Starting dashboard server..."

# Make the Python scripts executable
chmod +x start_webapp.py

# Define the Python path that we know works from the test_python.py output
PYTHON_PATH="/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"

# Check if the Python path exists
if [ -x "$PYTHON_PATH" ]; then
  echo "✅ Found Python at: $PYTHON_PATH"
  echo "🔄 Starting Dashboard Server..."
  $PYTHON_PATH start_webapp.py
else
  echo "❌ Python not found at $PYTHON_PATH. Trying alternative paths..."
  
  # Try with exact same path but in /mnt
  ALT_PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
  if [ -x "$ALT_PYTHON_PATH" ]; then
    echo "✅ Found Python at: $ALT_PYTHON_PATH"
    $ALT_PYTHON_PATH start_webapp.py
  else
    echo "❌ Python not found at alternative path."
    
    # Try the command used in the TestPython workflow
    echo "Trying the command used in TestPython workflow..."
    /mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3 start_webapp.py
    
    if [ $? -ne 0 ]; then
      echo "❌ Failed to start Dashboard Server. Please check your environment."
      exit 1
    fi
  fi
fi