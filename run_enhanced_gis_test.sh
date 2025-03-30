#!/bin/bash
# Run the enhanced GIS test

echo "Running enhanced GIS test..."
python test_enhanced_gis.py | tee gis_enhancement_test.log

echo -e "\nTest completed. Check gis_enhancement_test.log for details."