#!/bin/bash
# Test script for the enhanced GIS valuation API

# Check if the API key is set
API_KEY="${BCBS_VALUES_API_KEY:-sample_test_key}"

# Define the API URL (uses the default URL if not set)
API_URL="${API_HOST:-http://localhost:8000}/api/valuations"

# Define sample property data with enhanced GIS model
read -r -d '' DATA << EOM
{
  "address": "123 Enhanced GIS Drive",
  "city": "Richland",
  "state": "WA",
  "zip_code": "99352",
  "property_type": "Single Family",
  "bedrooms": 4,
  "bathrooms": 2.5,
  "square_feet": 2400,
  "lot_size": 9500,
  "year_built": 2005,
  "latitude": 46.2804,
  "longitude": -119.2752,
  "use_gis": true,
  "model_type": "enhanced_gis"
}
EOM

# Make the API request with enhanced GIS model
echo "Making API request to $API_URL with enhanced GIS model..."
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d "$DATA" | python3 -m json.tool

echo -e "\n\nNow trying with basic model for comparison..."

# Define the same property with basic model
read -r -d '' BASIC_DATA << EOM
{
  "address": "123 Basic Model Drive",
  "city": "Richland",
  "state": "WA",
  "zip_code": "99352",
  "property_type": "Single Family",
  "bedrooms": 4,
  "bathrooms": 2.5,
  "square_feet": 2400,
  "lot_size": 9500,
  "year_built": 2005,
  "latitude": 46.2804,
  "longitude": -119.2752,
  "use_gis": true,
  "model_type": "basic"
}
EOM

# Make the API request with basic model
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d "$BASIC_DATA" | python3 -m json.tool