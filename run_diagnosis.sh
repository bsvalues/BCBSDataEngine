#!/bin/bash

# BCBS Values Diagnostic Script
# This script attempts to run the diagnostic server, with multiple fallback options

echo "========================================================"
echo "BCBS VALUES DIAGNOSTIC SCRIPT"
echo "========================================================"
echo "Current directory: $(pwd)"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Try to find the Python executable
PYTHON=""

# Check common paths directly (not relying on PATH)
for python_path in \
  /usr/bin/python3 \
  /usr/bin/python3.11 \
  /usr/bin/python3.10 \
  /usr/bin/python3.9 \
  /usr/bin/python \
  /usr/local/bin/python3 \
  /usr/local/bin/python \
  /nix/store/*/bin/python3 \
  /nix/store/*/bin/python \
  /home/runner/.local/bin/python3 \
  /home/runner/.local/bin/python
do
  if [ -x "$python_path" ]; then
    PYTHON="$python_path"
    echo "Found Python at: $PYTHON"
    break
  fi
done

# If not found in common paths, try PATH
if [ -z "$PYTHON" ]; then
  for cmd in python3.11 python3.10 python3.9 python3 python; do
    if command_exists "$cmd"; then
      PYTHON="$cmd"
      echo "Found Python in PATH: $PYTHON ($(which $PYTHON))"
      break
    fi
  done
fi

# Last resort: use env to find python
if [ -z "$PYTHON" ]; then
  PYTHON=$(env python3 -c 'import sys; print(sys.executable)' 2>/dev/null)
  if [ $? -eq 0 ] && [ -n "$PYTHON" ]; then
    echo "Found Python via env: $PYTHON"
  else
    PYTHON=$(env python -c 'import sys; print(sys.executable)' 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$PYTHON" ]; then
      echo "Found Python via env: $PYTHON"
    fi
  fi
fi

if [ -z "$PYTHON" ]; then
  echo "ERROR: No Python executable found in PATH or common locations."
  echo "Paths checked in PATH:"
  which python python3 python3.11 python3.10 python3.9 2>/dev/null || echo "  No Python found via 'which'"
  echo
  echo "Attempt to locate python in /usr:"
  find /usr -name "python*" -type f -executable 2>/dev/null | grep -v "config" | head -n 10 || echo "  No Python found in /usr"
  echo
  echo "Attempt to locate python in /nix/store:"
  find /nix/store -name "python*" -type f -executable 2>/dev/null | head -n 5 || echo "  No Python found in /nix/store"
  echo
  echo "Will continue with shell-only diagnostics."
fi

# Only run Python-based diagnostics if we found Python
if [ -n "$PYTHON" ]; then
  echo "Using Python: $PYTHON"
  echo "Python version: $($PYTHON --version 2>&1)"
  echo

  # Define server port
  PORT=${PORT:-5000}

  # Try to run the unified diagnostic script
  echo "Attempting to run unified diagnostic server..."
  if [ -f "run_diagnostics.py" ]; then
    chmod +x run_diagnostics.py
    $PYTHON run_diagnostics.py
    if [ $? -eq 0 ]; then
      echo "Diagnostic server completed successfully."
      exit 0
    else
      echo "Unified diagnostic server failed, trying alternatives..."
    fi
  else
    echo "Unified diagnostic script not found, trying alternatives..."
  fi

  # Try to run the advanced diagnostic server directly
  echo "Attempting to run advanced diagnostic server directly..."
  if [ -f "quick_diagnostic_server.py" ]; then
    $PYTHON quick_diagnostic_server.py
    if [ $? -eq 0 ]; then
      echo "Advanced diagnostic server completed successfully."
      exit 0
    else
      echo "Advanced diagnostic server failed, trying simple server..."
    fi
  else
    echo "Advanced diagnostic script not found, trying simple server..."
  fi

  # Try to run the simple diagnostic server
  echo "Attempting to run simple diagnostic server..."
  if [ -f "simple_diagnostic.py" ]; then
    chmod +x simple_diagnostic.py
    $PYTHON simple_diagnostic.py
    if [ $? -eq 0 ]; then
      echo "Simple diagnostic server completed successfully."
      exit 0
    else
      echo "Simple diagnostic server failed."
    fi
  else
    echo "Simple diagnostic script not found."
  fi
else
  echo "========================================================"
  echo "WARNING: Python not found, skipping Python-based diagnostics"
  echo "========================================================"
fi

# Last resort: run a direct diagnostic
echo "========================================================"
echo "DIRECT DIAGNOSTIC INFORMATION"
echo "========================================================"
echo "Current time: $(date)"
echo "Python version: $($PYTHON --version 2>&1)"
echo "Python path: $(command -v $PYTHON)"
echo "Current directory: $(pwd)"
echo

echo "Environment variables:"
env | grep -E "^(DATABASE_URL|PG|API_KEY|SESSION_SECRET|PORT|PYTHONPATH)" | grep -v "PASSWORD" | sort

echo
echo "System information:"
uname -a
echo

echo "Disk space:"
df -h .
echo

echo "Directory listing:"
ls -la
echo

echo "========================================================"
echo "DIAGNOSTIC COMPLETE"
echo "========================================================"
echo "All diagnostic options have been tried."
echo "This information may help troubleshoot any issues."
echo "========================================================"

exit 1