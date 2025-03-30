#!/bin/bash

# Set API Key in Environment
# This script sets the BCBS_VALUES_API_KEY environment variable

# Check if API key is provided as argument
if [ $# -eq 0 ]; then
    echo "Please provide an API key as an argument."
    echo "Usage: ./set_api_key.sh YOUR_API_KEY"
    exit 1
fi

API_KEY=$1

# Export API key to environment
export BCBS_VALUES_API_KEY="$API_KEY"

# Verify the key is set
echo "API key has been set in the environment."
echo "Test with: echo \$BCBS_VALUES_API_KEY"

# Update .env file if it exists
if [ -f .env ]; then
    # Check if key already exists in .env
    if grep -q "BCBS_VALUES_API_KEY" .env; then
        # Replace existing key
        sed -i "s/BCBS_VALUES_API_KEY=.*/BCBS_VALUES_API_KEY=$API_KEY/" .env
    else
        # Add new key
        echo "BCBS_VALUES_API_KEY=$API_KEY" >> .env
    fi
    echo "Updated API key in .env file."
else
    # Create new .env file
    echo "BCBS_VALUES_API_KEY=$API_KEY" > .env
    echo "Created .env file with API key."
fi

echo "API key configuration complete."