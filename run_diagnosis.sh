#!/bin/bash
# This script runs the diagnostic tool using our new wrapper

# Install required packages if not already installed
pip_installed=$(pip list 2>/dev/null | grep -E "^psycopg2-binary|^sqlalchemy" | wc -l)
if [ "$pip_installed" -lt 2 ]; then
  echo "Installing required packages..."
  pip install psycopg2-binary sqlalchemy python-dotenv
fi

# Find available Python executables
echo "Looking for Python executables..."
which python python3 2>/dev/null || echo "No Python found in PATH"

# Try different Python executables
for python_cmd in python3 python; do
  if command -v $python_cmd >/dev/null 2>&1; then
    echo "Found $python_cmd, using it to run diagnostics wrapper..."
    $python_cmd -V
    
    # Use our Python wrapper which properly handles all environment variables
    $python_cmd run_diagnostics.py
    
    exit_code=$?
    echo ""
    echo "Diagnostics completed with exit code: $exit_code"
    
    # Print separator
    echo ""
    echo "=================================================================="
    echo "To run this diagnostic again with different variables, use:"
    echo "DATABASE_URL=your_connection_string ./run_diagnosis.sh"
    echo "=================================================================="
    exit $exit_code
  fi
done

# If we get here, no Python was found
echo "ERROR: No Python executable found. Please install Python 3.10 or later."
exit 1