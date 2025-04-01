#!/bin/bash

# BCBS Values Diagnostic Script
# This script attempts to run various diagnostics in a hierarchical fallback manner

echo "========================================================"
echo "BCBS VALUES DIAGNOSTIC SCRIPT"
echo "========================================================"
echo "Current directory: $(pwd)"
echo "Date: $(date)"

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Set default port
PORT=${PORT:-5000}
export PORT

# Try our custom diagnostic script first
echo "Trying custom diagnostic script..."
if [ -f "custom_diagnostic.sh" ]; then
  echo "Found custom_diagnostic.sh. Running..."
  if bash custom_diagnostic.sh; then
    echo "Custom diagnostic script completed successfully."
    exit 0
  else
    echo "Custom diagnostic script failed. Trying alternative methods..."
  fi
else
  echo "custom_diagnostic.sh not found. Trying alternative methods..."
fi

# Try Python-based diagnostic server
echo "Trying Python diagnostic server..."
if command_exists python3; then
  echo "Python 3 found. Trying simple_diagnostic.py..."
  if [ -f "simple_diagnostic.py" ]; then
    echo "Found simple_diagnostic.py. Running..."
    if python3 simple_diagnostic.py; then
      echo "Python diagnostic server completed successfully."
      exit 0
    else
      echo "Python diagnostic server failed. Trying Node.js..."
    fi
  else
    echo "simple_diagnostic.py not found. Trying Node.js..."
  fi
elif command_exists python; then
  echo "Python found. Trying simple_diagnostic.py..."
  if [ -f "simple_diagnostic.py" ]; then
    echo "Found simple_diagnostic.py. Running..."
    if python simple_diagnostic.py; then
      echo "Python diagnostic server completed successfully."
      exit 0
    else
      echo "Python diagnostic server failed. Trying Node.js..."
    fi
  else
    echo "simple_diagnostic.py not found. Trying Node.js..."
  fi
else
  echo "Python not found. Trying Node.js..."
fi

# Try Node.js-based server
echo "Trying Node.js diagnostic server..."
if command_exists node; then
  echo "Node.js found. Trying server.js..."
  if [ -f "server.js" ]; then
    echo "Found server.js. Running..."
    if node server.js; then
      echo "Node.js server completed successfully."
      exit 0
    else
      echo "Node.js server failed. Trying simple_http_server.js..."
    fi
  else
    echo "server.js not found. Trying simple_http_server.js..."
  fi

  if [ -f "simple_http_server.js" ]; then
    echo "Found simple_http_server.js. Running..."
    if node simple_http_server.js; then
      echo "Simple Node.js server completed successfully."
      exit 0
    else
      echo "Simple Node.js server failed. Moving to static diagnostic..."
    fi
  else
    echo "simple_http_server.js not found. Moving to static diagnostic..."
  fi
else
  echo "Node.js not found. Moving to static diagnostic..."
fi

# Final fallback: Just show plain text diagnostic info
echo "========================================================"
echo "STATIC DIAGNOSTIC INFORMATION"
echo "========================================================"
echo "No server could be started. Here's some basic diagnostic information:"

echo "System:"
uname -a

echo "Memory:"
free -h || echo "free command not available"

echo "Disk space:"
df -h . || echo "df command not available" 

echo "Environment:"
echo "- PATH: $PATH"
echo "- PWD: $PWD"
echo "- PORT: $PORT"

echo "Files:"
if [ -f "index.html" ]; then
  echo "index.html exists"
  echo "Try accessing index.html directly in your browser."
else 
  echo "index.html does not exist"
fi

echo "========================================================"
echo "DIAGNOSTICS COMPLETE"
echo "========================================================"
echo "All diagnostic options have been exhausted."
echo "Please check the diagnostic information above."

exit 1