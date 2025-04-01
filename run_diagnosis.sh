#!/bin/bash

# BCBS Values Platform Dashboard with Micro-animations
echo "BCBS Values Platform Dashboard Server"
echo "===================================="
echo "Starting enhanced dashboard with micro-animations..."

# Make the new Python script executable
chmod +x run_enhanced_dashboard.py

# Try various Python versions
python_found=0
for py_cmd in python3 python python3.11 python3.10 python3.9 python3.8 /nix/store/*python*3.11*/bin/python3.11; do
  if command -v $py_cmd >/dev/null 2>&1; then
    echo "✅ Python found: $($py_cmd --version 2>&1)"
    echo "🔄 Starting Dashboard Server with micro-animations..."
    $py_cmd run_enhanced_dashboard.py
    python_found=1
    break
  fi
done

# If no Python was found or the server couldn't start
if [ $python_found -eq 0 ]; then
  echo "❌ Python not found. Trying direct execution..."
  ./run_enhanced_dashboard.py
  
  if [ $? -ne 0 ]; then
    echo "❌ Failed to start Dashboard Server. Please check your environment."
    exit 1
  fi
fi