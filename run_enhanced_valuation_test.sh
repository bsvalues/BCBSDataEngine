#!/bin/bash
# Run the enhanced valuation test script

echo "Running Enhanced Valuation Tests..."
echo "==================================="

# Create a virtual environment for clean testing
python3 -m venv test_env

# Activate virtual environment
source test_env/bin/activate

# Install required packages
pip install numpy pandas scikit-learn

# Run the test
python test_enhanced_valuation.py

# Deactivate virtual environment
deactivate

echo "==================================="
echo "Test execution complete."