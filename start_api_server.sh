#!/bin/bash
# Start the API server in the background and save the PID

# Default configuration
API_PORT=5002
API_HOST="0.0.0.0"
DEBUG_MODE=true
PID_FILE=".api_pid"

# Print header
echo "===== Starting API Server ====="
echo "Host: $API_HOST"
echo "Port: $API_PORT"
echo "Debug Mode: $DEBUG_MODE"

# Check if API server is already running
if [ -f "$PID_FILE" ]; then
    API_PID=$(cat $PID_FILE | grep -o '[0-9]\+')
    if [ -n "$API_PID" ] && ps -p $API_PID > /dev/null; then
        echo "❌ API server is already running with PID: $API_PID"
        echo "Run ./stop_api_server.sh first if you want to restart the server."
        exit 1
    else
        echo "⚠️ Found stale PID file. Cleaning up..."
        rm $PID_FILE
    fi
fi

# Set environment variables
export API_HOST=$API_HOST
export API_PORT=$API_PORT
export DEBUG_MODE=$DEBUG_MODE
export JWT_SECRET="bcbs_jwt_secret_2025"  # Should be set as an environment variable in production

# Start the API server
echo "Starting API server..."
/mnt/nixmodules/nix/store/b03kwd9a5dm53k0z5vfzdhkvaa64c4g7-python3-3.10.13-env/bin/python3 api.py &

# Get the PID of the last background process
API_PID=$!

# Check if the process is running
if ps -p $API_PID > /dev/null; then
    echo "✅ API server started successfully with PID: $API_PID"
    echo $API_PID > $PID_FILE
    echo "API server is running at http://$API_HOST:$API_PORT"
else
    echo "❌ Failed to start API server"
    exit 1
fi

# Wait a moment to see if the server crashes immediately
sleep 2
if ! ps -p $API_PID > /dev/null; then
    echo "❌ API server crashed after starting"
    rm $PID_FILE
    exit 1
fi

echo ""
echo "API server is now running in the background."
echo "Server PID: $API_PID (saved to $PID_FILE)"
echo "API available at: http://$API_HOST:$API_PORT/api"
echo ""
echo "To test the API, run: ./test_api_with_curl.sh"
echo "To stop the server, run: ./stop_api_server.sh"
echo "====================================="