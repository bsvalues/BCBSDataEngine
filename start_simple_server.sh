#!/bin/bash

# Create a simple mock HTTP server using netcat
echo "Starting mock HTTP server on port 8080..."

# Function to handle HTTP requests
handle_request() {
  read request
  
  # Extract the requested path
  path=$(echo "$request" | head -1 | cut -d' ' -f2)
  
  echo "HTTP/1.1 200 OK"
  echo "Content-Type: application/json"
  echo "Access-Control-Allow-Origin: *"
  echo ""
  
  if [[ "$path" == "/" || "$path" == "/index.html" ]]; then
    echo '{"status":"running","message":"BCBS Values API is running!","version":"1.0"}'
  elif [[ "$path" == "/api/properties" || "$path" == "/api/valuations" ]]; then
    echo '[{"property_id":"BENT-12345","address":"123 Main St, Richland, WA 99352","estimated_value":345000,"confidence_score":0.85,"bedrooms":3,"bathrooms":2,"square_feet":1850,"year_built":2005,"property_type":"Single Family"},{"property_id":"BENT-67890","address":"456 Oak Ave, Kennewick, WA 99336","estimated_value":289000,"confidence_score":0.78,"bedrooms":2,"bathrooms":1.5,"square_feet":1550,"year_built":1995,"property_type":"Single Family"},{"property_id":"BENT-23456","address":"789 Pine Ln, Pasco, WA 99301","estimated_value":425000,"confidence_score":0.92,"bedrooms":4,"bathrooms":3,"square_feet":2200,"year_built":2018,"property_type":"Single Family"}]'
  elif [[ "$path" == "/api/validation" ]]; then
    echo '{"validation_passed":true,"total_records":150,"valid_records":148,"invalid_records":2,"validation_timestamp":"2025-03-29 22:30:00","validation_results":{"missing_values":{"count":1,"details":["Property BENT-45678 missing square_feet value"]},"invalid_values":{"count":1,"details":["Property BENT-98765 has invalid year_built (value: 3005)"]}}}'
  else
    echo '{"error":"Not found"}'
  fi
}

# Main loop to handle connections
while true; do
  handle_request | nc -l 8080
  echo "Request handled..."
done