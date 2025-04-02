#!/bin/bash

# Script to run the Simple Valuation API
echo "=== Starting BCBS_Values Simple Valuation API ==="
echo ""
echo "The API will be available at http://localhost:5002"
echo ""
echo "API Endpoints:"
echo "- GET /api/health - Health check endpoint"
echo "- POST /api/valuation - Generate a property valuation"
echo "- GET /api/neighborhoods - Get neighborhood quality ratings"
echo "- GET /api/reference-points - Get GIS reference points"
echo ""
echo "Press Ctrl+C to stop the API server"
echo "==========================================="

# Run the Simple Valuation API
./start_simple_api.sh