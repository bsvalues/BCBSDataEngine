#!/bin/bash
# Run script for the BCBS Values Platform Dashboard with micro-animations

# Make sure the script is executable
chmod +x serve_dashboard.py

# Find Python
for py_cmd in python3 python /nix/store/*python*3.11*/bin/python3.11 /usr/bin/python3.11 /usr/local/bin/python3.11; do
  if command -v $py_cmd >/dev/null 2>&1; then
    echo "Found Python at: $py_cmd"
    $py_cmd serve_dashboard.py
    exit $?
  fi
done

echo "Python not found, trying to run script directly..."
./serve_dashboard.py