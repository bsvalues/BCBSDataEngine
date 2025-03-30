#!/bin/bash
# Test script for the new property valuation API endpoint

# Set default values
API_PORT=${API_PORT:-8000}
API_KEY=${BCBS_VALUES_API_KEY:-"sample_test_key"}
API_URL="http://localhost:${API_PORT}"
TEST_MODE=${TEST_MODE:-"basic"}

# ANSI color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored headers
print_header() {
  echo -e "${BLUE}===${NC} $1 ${BLUE}===${NC}"
}

# Function to print success messages
print_success() {
  echo -e "${GREEN}✓${NC} $1"
}

# Function to print error messages
print_error() {
  echo -e "${RED}✗${NC} $1"
}

# Function to print warning messages
print_warning() {
  echo -e "${YELLOW}!${NC} $1"
}

# Check if we have a valuation engine
if [ ! -f "src/valuation.py" ]; then
  print_error "Valuation engine not found at src/valuation.py"
  exit 1
fi

# Set environment variables for API key if not already set
export BCBS_VALUES_API_KEY=${API_KEY}

# Start the API server
print_header "Starting FastAPI Server"
echo "API URL: ${API_URL}"
echo "API Key: ${API_KEY}"

# Start the API server in the background
python api.py &
API_PID=$!

# Wait for the server to start
echo "Waiting for API server to start..."
sleep 5

# Verify the server is running
if ! curl -s "${API_URL}" > /dev/null; then
  print_error "Failed to connect to API server at ${API_URL}"
  kill $API_PID
  exit 1
fi

print_success "API server is running"

# Run the test script
print_header "Testing Property Valuation API Endpoint"

if [ "$TEST_MODE" == "debug" ]; then
  python test_valuation_endpoint.py --url ${API_URL} --key ${API_KEY} --debug
else
  python test_valuation_endpoint.py --url ${API_URL} --key ${API_KEY}
fi

TEST_EXIT_CODE=$?

# Cleanup
print_header "Stopping API Server"
kill $API_PID
print_success "API server stopped"

# Final result
if [ $TEST_EXIT_CODE -eq 0 ]; then
  print_header "Test Result: SUCCESS"
  exit 0
else
  print_header "Test Result: FAILED"
  exit 1
fi