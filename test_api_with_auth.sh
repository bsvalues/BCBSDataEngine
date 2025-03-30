#!/bin/bash

# Test API Endpoints with Authentication
API_KEY="sample_test_key"  # Default sample key, in production use environment variable
BASE_URL="http://127.0.0.1:5000"

# Test health check endpoint (does not require auth)
echo "Testing health check endpoint..."
curl -s "${BASE_URL}/api/health" | jq .

# Test valuations endpoint with authentication
echo -e "\nTesting valuations endpoint with authentication..."
curl -s -H "X-API-KEY: ${API_KEY}" "${BASE_URL}/api/valuations?limit=2" 

# Test ETL status endpoint with authentication
echo -e "\nTesting ETL status endpoint with authentication..."
curl -s -H "X-API-KEY: ${API_KEY}" "${BASE_URL}/api/etl-status"

# Test agent status endpoint with authentication
echo -e "\nTesting agent status endpoint with authentication..."
curl -s -H "X-API-KEY: ${API_KEY}" "${BASE_URL}/api/agent-status"

# Test creating a property valuation with authentication
echo -e "\nTesting property valuation creation with authentication..."
curl -s -X POST "${BASE_URL}/api/valuations" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "address": "123 Main St",
    "city": "Richland",
    "state": "WA",
    "zip_code": "99352",
    "property_type": "Single Family",
    "bedrooms": 3,
    "bathrooms": 2,
    "square_feet": 1800,
    "lot_size": 8500,
    "year_built": 1995,
    "use_gis": true,
    "model_type": "basic"
  }'

echo -e "\nAll tests completed!"