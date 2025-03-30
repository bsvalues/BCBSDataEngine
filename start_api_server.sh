#!/bin/bash

# Start the API server with FastAPI and authentication
echo "Starting BCBS Values API Server with authentication..."

# Check if Python environment is active
python -c "import uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing uvicorn..."
    pip install uvicorn fastapi
fi

# Start the server with uvicorn
echo "Starting API server on port 5000..."
uvicorn api:app --host 0.0.0.0 --port 5000 --reload