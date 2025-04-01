#!/bin/bash

# Kill any existing Python processes
pkill -f "python.*start_webapp.py" || true
sleep 1

# Run the server with the full path to Python
/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11 start_webapp.py