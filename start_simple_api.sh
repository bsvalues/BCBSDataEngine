#!/bin/bash

echo "=== Starting the BCBS_Values Simple Valuation API ==="

# Set environment variables
export PORT=${PORT:-5002}
export WORKERS=${WORKERS:-4}
export TIMEOUT=${TIMEOUT:-120}
export LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "Starting API server with ${WORKERS} workers on port ${PORT}"

# Start the gunicorn server
exec gunicorn simple_valuation_api:app \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --timeout ${TIMEOUT} \
    --log-level ${LOG_LEVEL} \
    --access-logfile - \
    --error-logfile - \
    --preload