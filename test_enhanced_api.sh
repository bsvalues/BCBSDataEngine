#!/bin/bash

# Script to run the Enhanced API test

# Set environment variables for testing
export BCBS_VALUES_API_KEY="test_api_key_1234"

# Check if the API is already running
PORT=8000
if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/ > /dev/null 2>&1; then
    echo "API server already running on port $PORT"
else
    echo "Starting API server..."
    python api.py > api_server.log 2>&1 &
    API_PID=$!
    
    # Wait for server to start
    echo "Waiting for API server to start..."
    for i in {1..20}; do
        sleep 1
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/ > /dev/null 2>&1; then
            echo "API server started successfully"
            break
        fi
        if [ $i -eq 20 ]; then
            echo "Failed to start API server within timeout"
            cat api_server.log
            kill $API_PID 2>/dev/null
            exit 1
        fi
    done
fi

# Run the test script
echo "Running enhanced API tests..."
python test_enhanced_api.py

# Capture the exit code
TEST_EXIT_CODE=$?

# Print the summary
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n✅ All tests passed!"
else
    echo -e "\n❌ Some tests failed. See output above for details."
fi

# Don't stop the API server automatically to allow for further testing
echo -e "\nAPI server is still running. Press Ctrl+C to stop."

exit $TEST_EXIT_CODE