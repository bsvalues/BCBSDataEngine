#!/bin/bash
# Test the API endpoints using curl

# Configuration
API_BASE_URL="http://localhost:5002/api"
API_KEY=${API_KEY:-bcbs_demo_key_2023}
TOKEN=""

# Print header
echo "===== Testing API with curl ====="
echo "Date: $(date)"
echo "API Base URL: $API_BASE_URL"
echo "API Key: ${API_KEY:0:4}...${API_KEY: -4}"
echo

# Test authentication token endpoint
echo "----- Testing authentication token endpoint -----"
TOKEN_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "{\"agent_id\":\"curl-test-agent\",\"agent_type\":\"curl_test\",\"api_key\":\"$API_KEY\"}" \
  $API_BASE_URL/auth/token)

echo "Response: $TOKEN_RESPONSE"
# Extract token if available
if echo "$TOKEN_RESPONSE" | grep -q "token"; then
    TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"token":"[^"]*"' | sed 's/"token":"//;s/"//')
    echo "Acquired token: ${TOKEN:0:20}..."
fi
echo

# Test enhanced valuations endpoint
echo "----- Testing enhanced valuations endpoint -----"
if [ -n "$TOKEN" ]; then
  echo "Using JWT token for authentication"
  curl -s -H "Authorization: Bearer $TOKEN" "$API_BASE_URL/valuations?limit=3" | jq
else
  echo "Using API key for authentication"
  curl -s -H "X-API-Key: $API_KEY" "$API_BASE_URL/valuations?limit=3" | jq
fi
echo

# Test ETL status endpoint
echo "----- Testing ETL status endpoint -----"
if [ -n "$TOKEN" ]; then
  curl -s -H "Authorization: Bearer $TOKEN" "$API_BASE_URL/etl-status" | jq
else
  curl -s -H "X-API-Key: $API_KEY" "$API_BASE_URL/etl-status" | jq
fi
echo

# Test agent status endpoint
echo "----- Testing agent status endpoint -----"
if [ -n "$TOKEN" ]; then
  curl -s -H "Authorization: Bearer $TOKEN" "$API_BASE_URL/agent-status" | jq
else
  curl -s -H "X-API-Key: $API_KEY" "$API_BASE_URL/agent-status" | jq
fi
echo

# Test applying filters to valuations endpoint
echo "----- Testing valuations endpoint with filters -----"
if [ -n "$TOKEN" ]; then
  curl -s -H "Authorization: Bearer $TOKEN" "$API_BASE_URL/valuations?method=enhanced_regression&min_confidence=0.8&sort_by=estimated_value&sort_dir=desc&limit=3" | jq
else
  curl -s -H "X-API-Key: $API_KEY" "$API_BASE_URL/valuations?method=enhanced_regression&min_confidence=0.8&sort_by=estimated_value&sort_dir=desc&limit=3" | jq
fi
echo

# Test agent logs endpoint if we have agents
echo "----- Testing agent logs endpoint -----"
if [ -n "$TOKEN" ]; then
  AGENT_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$API_BASE_URL/agent-status")
else
  AGENT_RESPONSE=$(curl -s -H "X-API-Key: $API_KEY" "$API_BASE_URL/agent-status")
fi

AGENT_ID=$(echo "$AGENT_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | sed 's/"id"://')

if [ -n "$AGENT_ID" ]; then
  echo "Found agent ID: $AGENT_ID, retrieving logs"
  if [ -n "$TOKEN" ]; then
    curl -s -H "Authorization: Bearer $TOKEN" "$API_BASE_URL/agent-logs/$AGENT_ID" | jq
  else
    curl -s -H "X-API-Key: $API_KEY" "$API_BASE_URL/agent-logs/$AGENT_ID" | jq
  fi
else
  echo "No agents found to test logs endpoint"
fi
echo

echo "===== API tests completed ====="