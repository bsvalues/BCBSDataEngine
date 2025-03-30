#!/bin/bash

# Enhanced API Endpoints Test Script
# This script runs the new API endpoint tests added to test_integration_advanced.py

echo "Running enhanced API endpoint tests..."
echo "======================================="

# Run specific tests for the newly added API endpoints
python -m pytest tests/test_integration_advanced.py::TestAPIEndpoints::test_api_neighborhoods_endpoint tests/test_integration_advanced.py::TestAPIEndpoints::test_api_property_search_endpoint tests/test_integration_advanced.py::TestAPIEndpoints::test_api_valuation_history_endpoint tests/test_integration_advanced.py::TestAPIEndpoints::test_api_market_trends_endpoint -v

# Get the exit status
STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo ""
  echo "✅ All tests passed successfully!"
else
  echo ""
  echo "❌ Some tests failed. Check the output above for details."
fi

exit $STATUS