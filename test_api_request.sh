#!/bin/bash

echo "=== Testing the BCBS_Values Simple Valuation API ==="

# Define the API URL
API_URL="http://localhost:5002"

# Function to test the health endpoint
test_health() {
    echo "Testing health endpoint..."
    RESPONSE=$(curl -s "$API_URL/api/health")
    echo "Response: $RESPONSE"
    echo ""
}

# Function to test the valuation endpoint
test_valuation() {
    echo "Testing valuation endpoint..."
    RESPONSE=$(curl -s -X POST "$API_URL/api/valuation" \
        -H "Content-Type: application/json" \
        -d '{
            "square_feet": 2000,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "year_built": 2005,
            "latitude": 46.2804,
            "longitude": -119.2752,
            "city": "Richland",
            "neighborhood": "Meadow Springs",
            "use_gis": true
        }')
    echo "Response: $RESPONSE"
    echo ""
}

# Function to test the neighborhoods endpoint
test_neighborhoods() {
    echo "Testing neighborhoods endpoint..."
    RESPONSE=$(curl -s "$API_URL/api/neighborhoods")
    echo "Response: $RESPONSE"
    echo ""
}

# Function to test the reference points endpoint
test_reference_points() {
    echo "Testing reference points endpoint..."
    RESPONSE=$(curl -s "$API_URL/api/reference-points")
    echo "Response: $RESPONSE"
    echo ""
}

# Run all tests
test_health
test_valuation
test_neighborhoods
test_reference_points

echo "=== API Testing Complete ==="