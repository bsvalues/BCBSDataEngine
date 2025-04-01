#!/bin/bash

# Direct script for running the Python web server
# This script uses the direct Python path and runs the webapp.py file

echo "Starting BCBS Values Platform Web Server"
echo "========================================"

# Use the full path to Python - this works in manual testing
/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11 webapp.py