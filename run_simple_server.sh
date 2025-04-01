#!/bin/bash

# Run the simple Python HTTP server using direct path

# We'll use the full Python path we found previously
PYTHON_CMD="/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"

echo "Starting BCBS Values Platform Simple Web Server"
echo "=============================================="
echo "Using Python: $PYTHON_CMD"

# Run the server
$PYTHON_CMD simple_server.py