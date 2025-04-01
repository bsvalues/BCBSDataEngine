#!/bin/bash
# Direct execution script for the BCBS Values Platform server
# This script avoids any intermediate scripts and directly runs Python

# Make the Python script executable
chmod +x simple_server.py

# Run the Python script directly with the full path to Python
echo "Running BCBS Values Platform server directly..."
/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11 simple_server.py