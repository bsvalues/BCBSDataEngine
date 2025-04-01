#!/bin/bash

# BCBS Values Direct Diagnostic Script
# This script outputs diagnostic information directly to console
# without requiring a web server

echo "========================================================"
echo "BCBS VALUES DIRECT DIAGNOSTIC OUTPUT"
echo "========================================================"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo

# Use Python 3.11 absolute path to run simple_diagnostic.py
PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

if [ -x "$PYTHON_PATH" ] && [ -f "simple_diagnostic.py" ]; then
  echo "Found Python 3.11 at $PYTHON_PATH"
  echo "Starting simple diagnostic server..."
  $PYTHON_PATH simple_diagnostic.py &
  SERVER_PID=$!
  echo "Server started with PID $SERVER_PID"
  echo "You can access the server at: http://0.0.0.0:5000"
  
  # Let the server run in the background
  # Keep the script alive so the server doesn't terminate
  wait $SERVER_PID
  exit 0
fi

# Check Python versions (if direct execution failed)
echo "Python versions available:"
for pyver in python python3 python3.8 python3.9 python3.10 python3.11; do
  if command -v $pyver > /dev/null 2>&1; then
    echo "  $pyver: $($pyver --version 2>&1) - $(which $pyver)"
  fi
done
echo

# Check common utilities
echo "Checking required utilities:"
for cmd in curl wget jq sqlite3 pg_dump psql pip pip3 git npm node; do
  if command -v $cmd > /dev/null 2>&1; then
    echo "  ✓ $cmd ($(which $cmd))"
  else
    echo "  ✗ $cmd (not found)"
  fi
done
echo

# Environment variables
echo "Environment variables (filtered):"
env | grep -E "^(DATABASE_URL|PG|API_KEY|SESSION_SECRET|PORT|PYTHONPATH|PATH|REPO|REPL)" | grep -v "PASSWORD" | sort
echo

# Directory structure
echo "Directory structure (top level):"
ls -la | head -n 20
if [ $(ls -la | wc -l) -gt 20 ]; then
  echo "  ... and $(( $(ls -la | wc -l) - 20 )) more files/directories"
fi
echo

# Check for critical files
echo "Checking for critical files:"
for file in quick_diagnostic_server.py simple_diagnostic.py run_diagnostics.py api.py basic_api.py app.py main.py server.py valuation.py run_diagnosis.sh; do
  if [ -f "$file" ]; then
    echo "  ✓ $file ($(wc -l < $file) lines, $(stat -c %s $file 2>/dev/null || stat -f %z $file) bytes)"
  else
    echo "  ✗ $file (not found)"
  fi
done
echo

# Check database connection
echo "Testing database connection:"
if [ -n "$DATABASE_URL" ]; then
  echo "  DATABASE_URL is set"
  if command -v psql > /dev/null 2>&1; then
    echo "  Attempting to connect (no password output)..."
    if psql "$DATABASE_URL" -c "SELECT 1" > /dev/null 2>&1; then
      echo "  ✓ Database connection successful"
    else
      echo "  ✗ Database connection failed"
    fi
  else
    echo "  psql not available, skipping connection test"
  fi
else
  echo "  DATABASE_URL is not set"
fi
echo

# System information
echo "System information:"
echo "  Operating System: $(uname -s)"
echo "  Release: $(uname -r)"
echo "  Architecture: $(uname -m)"
if [ -f /etc/os-release ]; then
  source /etc/os-release
  echo "  Distribution: $PRETTY_NAME"
fi
echo

# Python packages
echo "Python packages (if pip available):"
if command -v pip > /dev/null 2>&1; then
  pip list | head -n 20
  if [ $(pip list | wc -l) -gt 20 ]; then
    echo "  ... and $(( $(pip list | wc -l) - 20 )) more packages"
  fi
elif command -v pip3 > /dev/null 2>&1; then
  pip3 list | head -n 20
  if [ $(pip3 list | wc -l) -gt 20 ]; then
    echo "  ... and $(( $(pip3 list | wc -l) - 20 )) more packages"
  fi
else
  echo "  pip not available, skipping package list"
fi
echo

# Disk space
echo "Disk space:"
df -h . 2>/dev/null || echo "  df command failed"
echo

echo "========================================================"
echo "DIAGNOSTIC OUTPUT COMPLETE"
echo "========================================================"

# This helps with automation - exit successfully
exit 0