#!/bin/bash

# Set up logging for API server
LOG_FILE="logs/api_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Start the API server with logging
echo "Starting API server..."
exec python run_api.py 2>&1 | tee $LOG_FILE