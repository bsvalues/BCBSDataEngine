#!/bin/bash
# Simple script to run the Flask dashboard server

echo "Starting Flask dashboard server..."
PYTHON_PATH="/mnt/nixmodules/nix/store/83mqxa6p0r3f1fh6f70737ifah5ik7qp-python-wrapped-0.1.0/bin/python3.10"

# Check if Flask is installed
if ! $PYTHON_PATH -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    $PYTHON_PATH -m pip install flask
fi

# Start the server
echo "Running Flask server on port 5002..."
FLASK_APP=serve_dashboard.py $PYTHON_PATH -m flask run --host=0.0.0.0 --port=5002