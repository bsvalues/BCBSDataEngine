#!/bin/bash

# Start the valuation API in the background
echo "Starting Valuation API server..."
python simple_valuation_api.py > api_server.log 2>&1 &
API_PID=$!

# Give the server time to start
echo "Waiting for server to start..."
sleep 5

# Test the API with a simple health check
echo -e "\nTesting API Health:"
curl -s http://localhost:5002/api/health | jq .

# Get neighborhood ratings
echo -e "\nRetrieving neighborhood ratings:"
curl -s http://localhost:5002/api/neighborhoods | jq .

# Test valuation endpoint with a sample property
echo -e "\nTesting property valuation with GIS data:"
curl -s -X POST http://localhost:5002/api/valuation \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 2200,
    "bedrooms": 4,
    "bathrooms": 2.5,
    "year_built": 2010,
    "latitude": 46.2804,
    "longitude": -119.2752,
    "city": "Richland",
    "neighborhood": "Meadow Springs",
    "use_gis": true
  }' | jq .

# Test valuation with different neighborhoods
echo -e "\nComparing valuations across different neighborhoods:"

for location in "Richland" "Kennewick" "West Richland"; do
  echo -e "\n$location valuation:"
  curl -s -X POST http://localhost:5002/api/valuation \
    -H "Content-Type: application/json" \
    -d "{
      \"square_feet\": 2000,
      \"bedrooms\": 3,
      \"bathrooms\": 2,
      \"year_built\": 2005,
      \"city\": \"$location\",
      \"use_gis\": true
    }" | jq '.valuation.predicted_value, .valuation.gis_metrics.adjustment_factor'
done

# Kill the server process
echo -e "\nTests complete. Stopping API server..."
kill $API_PID

echo "Done!"