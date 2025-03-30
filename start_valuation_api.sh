#!/bin/bash

echo "Starting BCBS_Values Valuation API server..."

# Start the valuation API with Gunicorn
# This provides better process management and reliability
gunicorn --bind 0.0.0.0:5002 --timeout 90 --workers 2 --log-level info "simple_valuation_api:app"