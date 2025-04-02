#!/bin/bash
# Kill any existing instances of the server
pkill -f "python3.10 simple_http_server.py" || true

# Define Python executable path
PYTHON_BIN="/mnt/nixmodules/nix/store/f5prw9mm064yp8g96fcals3mrsadysig-python3-3.10.16/bin/python3.10"

# Start Python HTTP server to serve dashboard files
echo "Starting Python HTTP server..."
echo "Server start: $(date)" > server.log
$PYTHON_BIN simple_http_server.py >> server.log 2>&1 &
SERVER_PID=$!

# Wait a moment to ensure server starts
sleep 2

# Check if process is running
if ps -p $SERVER_PID > /dev/null; then
    echo "Server started successfully! PID: $SERVER_PID"
    echo "Server log:"
    tail -5 server.log
    echo "Dashboard should be available at http://0.0.0.0:5002/"
else
    echo "Server failed to start. Check server.log for details:"
    cat server.log
fi