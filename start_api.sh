#!/bin/bash
# Start script for the BCBS Values API
echo "Starting BCBS Values API server on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload