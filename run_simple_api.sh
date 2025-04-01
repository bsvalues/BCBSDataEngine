#!/bin/bash

# Script to run the Simple Valuation API with uvicorn for FastAPI support
# This script handles virtual environment activation and proper server configuration

# Set error handling to stop on any error
set -e

# Print colorful banner for better visibility
echo -e "\033[1;36m============================================\033[0m"
echo -e "\033[1;36m=== Starting BCBS_Values Simple Valuation API ===\033[0m"
echo -e "\033[1;36m============================================\033[0m"

# Set environment variables with defaults
export PORT=${PORT:-8000}
export HOST=${HOST:-"0.0.0.0"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export MODULE_NAME=${MODULE_NAME:-"api:app"}
export WORKERS=${WORKERS:-4}
export RELOAD=${RELOAD:-"--reload"}

# Check if we're in a Python virtual environment, activate if not
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "\033[0;33mNo active Python virtual environment detected\033[0m"
    
    # Check if there's a venv directory and activate it if it exists
    if [ -d "venv" ]; then
        echo -e "\033[0;32mActivating virtual environment from ./venv\033[0m"
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        echo -e "\033[0;32mActivating virtual environment from ./.venv\033[0m"
        source .venv/bin/activate
    else
        echo -e "\033[0;33mNo virtual environment found. Using system Python...\033[0m"
    fi
else
    echo -e "\033[0;32mUsing active virtual environment: $VIRTUAL_ENV\033[0m"
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo -e "\033[0;31mUvicorn not found. Attempting to install...\033[0m"
    pip install uvicorn
fi

# Check if FastAPI is installed
if ! pip list | grep -i fastapi &> /dev/null; then
    echo -e "\033[0;31mFastAPI not found. Attempting to install...\033[0m"
    pip install fastapi
fi

# Display setup information
echo ""
echo -e "\033[1mServer Configuration:\033[0m"
echo -e "- \033[1mHost:\033[0m $HOST"
echo -e "- \033[1mPort:\033[0m $PORT"
echo -e "- \033[1mModule:\033[0m $MODULE_NAME"
echo -e "- \033[1mWorkers:\033[0m $WORKERS"
echo -e "- \033[1mLog Level:\033[0m $LOG_LEVEL"
echo ""

echo -e "\033[1mAPI will be available at:\033[0m http://$HOST:$PORT"
echo ""
echo -e "\033[1mAPI Endpoints:\033[0m"
echo "- GET /api/health - Health check endpoint"
echo "- POST /api/valuation - Generate a property valuation"
echo "- GET /api/neighborhoods - Get neighborhood quality ratings"
echo "- GET /api/reference-points - Get GIS reference points"
echo "- GET /api/etl-status - Get ETL pipeline status"
echo "- GET /api/agent-status - Get agent status information"
echo ""
echo -e "\033[1mPress Ctrl+C to stop the API server\033[0m"
echo "============================================="

# Start the server with error handling
echo -e "\033[0;32mStarting FastAPI server with uvicorn...\033[0m"
echo ""

# Use try-catch block for better error reporting
{
    # Start the uvicorn server with the specified configuration
    uvicorn $MODULE_NAME --host $HOST --port $PORT --workers $WORKERS --log-level $LOG_LEVEL $RELOAD
} || {
    # Print error message if server fails to start
    echo -e "\033[0;31mError: Failed to start the FastAPI server\033[0m"
    echo -e "\033[0;31mPlease check the error messages above\033[0m"
    exit 1
}