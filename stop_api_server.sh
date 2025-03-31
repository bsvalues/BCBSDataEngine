#!/bin/bash
# Stop the API server

# Check if PID file exists
if [ -f .api_pid ]; then
    # Read PID from file
    API_PID=$(cat .api_pid | grep -o '[0-9]\+')
    
    if [ -n "$API_PID" ]; then
        echo "Stopping API server with PID: $API_PID"
        
        # Check if process is running
        if ps -p $API_PID > /dev/null; then
            # Send SIGTERM to process
            kill $API_PID
            echo "Termination signal sent, waiting for server to stop..."
            
            # Wait for process to terminate
            for i in {1..5}; do
                if ! ps -p $API_PID > /dev/null; then
                    echo "✅ API server stopped successfully."
                    rm .api_pid
                    exit 0
                fi
                sleep 1
            done
            
            # If process is still running, force kill
            if ps -p $API_PID > /dev/null; then
                echo "Server did not terminate gracefully, sending SIGKILL..."
                kill -9 $API_PID
                
                if ! ps -p $API_PID > /dev/null; then
                    echo "✅ API server forcefully terminated."
                    rm .api_pid
                    exit 0
                else
                    echo "❌ Failed to terminate API server."
                    exit 1
                fi
            fi
        else
            echo "No API server running with PID: $API_PID"
            rm .api_pid
            exit 0
        fi
    else
        echo "Invalid PID file format."
        rm .api_pid
        exit 1
    fi
else
    echo "No API server PID file found."
    
    # Check for any Python process running api.py
    API_PID=$(ps aux | grep "python3 api.py" | grep -v grep | awk '{print $2}')
    
    if [ -n "$API_PID" ]; then
        echo "Found API server process: $API_PID"
        echo "Stopping API server..."
        kill $API_PID
        echo "✅ API server stopped."
    else
        echo "No API server process found."
    fi
    
    exit 0
fi