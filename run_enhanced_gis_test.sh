#!/bin/bash

# Run Enhanced GIS Features Test
# This script runs the enhanced GIS features test and outputs the result

echo "Running Enhanced GIS Features Test..."
python run_enhanced_gis_test.py

echo ""
echo "Running unit tests for Enhanced GIS Features..."
python -m unittest tests/test_enhanced_gis_features.py

echo ""
echo "Results saved to enhanced_gis_valuation_result.log and gis_features_metadata.json"
echo "Sample data saved to enhanced_gis_sample_data.csv"