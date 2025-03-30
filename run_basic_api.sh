#!/bin/bash

# run_basic_api.sh
# This script starts the FastAPI server using uvicorn

# Set variables
API_FILE="basic_api.py"   # The Python file containing the FastAPI app
HOST="0.0.0.0"            # Host address (0.0.0.0 makes it available on all network interfaces)
PORT="5000"               # Port number for the server
LOG_LEVEL="info"          # Logging level (debug, info, warning, error, critical)

# Display startup message
echo "Starting Basic API Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Press Ctrl+C to stop the server"
echo "-----------------------------------"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 to continue."
    exit 1
fi

# Check if API file exists
if [ ! -f "$API_FILE" ]; then
    echo "Error: API file '$API_FILE' not found."
    exit 1
fi

# Check if uvicorn is installed
if ! python3 -c "import uvicorn" &> /dev/null; then
    echo "Warning: uvicorn module not found. Attempting to install..."
    pip install uvicorn
    
    # Verify installation
    if ! python3 -c "import uvicorn" &> /dev/null; then
        echo "Error: Failed to install uvicorn. Please install it manually with 'pip install uvicorn'."
        exit 1
    fi
    echo "uvicorn installed successfully."
fi

# Check if FastAPI is installed
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "Warning: fastapi module not found. Attempting to install..."
    pip install fastapi
    
    # Verify installation
    if ! python3 -c "import fastapi" &> /dev/null; then
        echo "Error: Failed to install fastapi. Please install it manually with 'pip install fastapi'."
        exit 1
    fi
    echo "fastapi installed successfully."
fi

# Start the server using uvicorn
# --reload: Restarts the server when code changes (useful for development)
# --host: Set the host address
# --port: Set the port number
# --log-level: Set the logging level
echo "Starting uvicorn server..."
python3 -m uvicorn basic_api:app --reload --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"

# Note: This script won't reach this point unless uvicorn is stopped
echo "Server has been stopped."