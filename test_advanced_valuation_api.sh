#!/bin/bash

# Test script for the advanced valuation API endpoint

echo "Testing advanced property valuation API endpoint..."
curl -X POST "http://localhost:5001/api/v1/valuations/advanced" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: sample_test_key" \
  -d '{
    "square_feet": 2400,
    "bedrooms": 4,
    "bathrooms": 2.5,
    "year_built": 1995,
    "latitude": 46.2415,
    "longitude": -119.2756,
    "address": "1234 Test St",
    "city": "Richland",
    "state": "WA",
    "property_type": "single_family",
    "use_gis": true,
    "model_type": "enhanced_gis"
  }'

echo -e "\n\nDone."