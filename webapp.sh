#!/bin/bash

# Kill any existing Python processes for our application
pkill -f "python.*simple_server.py" || true
sleep 1

# Run the server with the full path to Python
/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11 simple_server.py