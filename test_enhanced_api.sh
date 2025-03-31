#!/bin/bash
# Set Python path
PYTHON_PATH="/mnt/nixmodules/nix/store/b03kwd9a5dm53k0z5vfzdhkvaa64c4g7-python3-3.10.13-env/bin/python3"
# Script to test the enhanced API functionality

# Configuration
API_BASE_URL="http://localhost:5002/api"
API_KEY=${API_KEY:-bcbs_demo_key_2023}
AGENT_ID="bash-test-agent"
AGENT_TYPE="bash_test"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print test header
echo -e "${BLUE}===== Enhanced API Test =====${NC}"
echo "Date: $(date)"
echo "API Base URL: $API_BASE_URL"
echo "API Key: ${API_KEY:0:4}...${API_KEY: -4}"
echo

# Function to check if jq is installed
check_jq() {
  if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed${NC}"
    echo "This script uses jq to parse JSON responses."
    echo "Please install jq using your package manager (e.g., apt-get install jq)."
    exit 1
  fi
}

# Run a test and check the result
run_test() {
  local name=$1
  local command=$2
  local expected_status=$3
  local expected_key=$4
  
  echo -e "${BLUE}Test: $name${NC}"
  echo "Command: $command"
  
  # Run the command and capture output
  output=$(eval $command)
  status=$?
  
  # Check if command succeeded
  if [ $status -ne 0 ]; then
    echo -e "${RED}Error: Command failed with status $status${NC}"
    echo "$output"
    return 1
  fi
  
  # Check if output is valid JSON
  if ! echo "$output" | jq -e . > /dev/null 2>&1; then
    echo -e "${RED}Error: Invalid JSON response${NC}"
    echo "$output"
    return 1
  fi
  
  # Check status code if expected_status is provided
  if [ -n "$expected_status" ]; then
    response_status=$(echo "$output" | jq -r '.status // .statusCode // 200')
    if [ "$response_status" != "$expected_status" ]; then
      echo -e "${RED}Error: Expected status $expected_status, got $response_status${NC}"
      echo "$output" | jq
      return 1
    fi
  fi
  
  # Check for expected key if provided
  if [ -n "$expected_key" ]; then
    if ! echo "$output" | jq -e ".$expected_key" > /dev/null 2>&1; then
      echo -e "${RED}Error: Expected key '$expected_key' not found in response${NC}"
      echo "$output" | jq
      return 1
    fi
  fi
  
  echo -e "${GREEN}✅ Test passed${NC}"
  echo
  return 0
}

# Check if the API server is running
check_api_server() {
  echo -e "${BLUE}Checking if API server is running...${NC}"
  if ! curl -s -o /dev/null -w "%{http_code}" "${API_BASE_URL}/health" | grep -q "200"; then
    echo -e "${RED}Error: API server is not running${NC}"
    echo "Please start the API server using ./start_api_server.sh"
    exit 1
  fi
  echo -e "${GREEN}API server is running${NC}"
  echo
}

# Main test sequence
main() {
  check_jq
  check_api_server
  
  # Step 1: Get authentication token
  echo -e "${BLUE}===== Step 1: Get authentication token =====${NC}"
  token_command="curl -s -X POST -H 'Content-Type: application/json' -d '{\"agent_id\":\"$AGENT_ID\",\"agent_type\":\"$AGENT_TYPE\",\"api_key\":\"$API_KEY\"}' $API_BASE_URL/auth/token | jq"
  run_test "Authentication Token" "$token_command" "" "token"
  
  # Extract token
  token=$(eval $token_command | jq -r '.token')
  if [ -z "$token" ] || [ "$token" = "null" ]; then
    echo -e "${RED}Error: Could not extract token${NC}"
    exit 1
  fi
  echo -e "${GREEN}Token: ${token:0:20}...${NC}"
  
  # Step 2: Test valuations endpoint
  echo -e "${BLUE}===== Step 2: Test valuations endpoint =====${NC}"
  valuations_command="curl -s -H 'Authorization: Bearer $token' $API_BASE_URL/valuations | jq"
  run_test "Valuations" "$valuations_command" "" "valuations"
  
  # Step 3: Test valuations filtering
  echo -e "${BLUE}===== Step 3: Test valuations filtering =====${NC}"
  filtering_command="curl -s -H 'Authorization: Bearer $token' '$API_BASE_URL/valuations?limit=5&sort_by=valuation_date&sort_dir=desc' | jq"
  run_test "Valuations Filtering" "$filtering_command" "" "valuations"
  
  # Step 4: Test ETL status endpoint
  echo -e "${BLUE}===== Step 4: Test ETL status endpoint =====${NC}"
  etl_command="curl -s -H 'Authorization: Bearer $token' $API_BASE_URL/etl-status | jq"
  run_test "ETL Status" "$etl_command" "" "jobs"
  
  # Step 5: Test agent status endpoint
  echo -e "${BLUE}===== Step 5: Test agent status endpoint =====${NC}"
  agent_command="curl -s -H 'Authorization: Bearer $token' $API_BASE_URL/agent-status | jq"
  run_test "Agent Status" "$agent_command" "" "agents"
  
  # Step 6: Test agent status filtering
  echo -e "${BLUE}===== Step 6: Test agent status filtering =====${NC}"
  agent_filter_command="curl -s -H 'Authorization: Bearer $token' '$API_BASE_URL/agent-status?active_only=true' | jq"
  run_test "Agent Status Filtering" "$agent_filter_command" "" "agents"
  
  # Step 7: Test agent logs endpoint (if agents exist)
  echo -e "${BLUE}===== Step 7: Test agent logs endpoint =====${NC}"
  # First get an agent ID
  agent_id=$(eval $agent_command | jq -r '.agents[0].id // "none"')
  if [ "$agent_id" != "none" ]; then
    logs_command="curl -s -H 'Authorization: Bearer $token' $API_BASE_URL/agent-logs/$agent_id | jq"
    run_test "Agent Logs" "$logs_command" "" "logs"
  else
    echo -e "${YELLOW}Skipping agent logs test, no agents found${NC}"
  fi
  
  # Step 8: Test invalid token
  echo -e "${BLUE}===== Step 8: Test invalid token =====${NC}"
  invalid_token_command="curl -s -H 'Authorization: Bearer invalid.token.here' $API_BASE_URL/valuations | jq"
  run_test "Invalid Token" "$invalid_token_command" "401" "error"
  
  # Step 9: Test invalid API key
  echo -e "${BLUE}===== Step 9: Test invalid API key =====${NC}"
  invalid_key_command="curl -s -H 'X-API-Key: invalid_api_key' $API_BASE_URL/valuations | jq"
  run_test "Invalid API Key" "$invalid_key_command" "401" "error"
  
  # Print summary
  echo -e "${GREEN}===== All tests completed successfully =====${NC}"
}

# Run the main test sequence
main