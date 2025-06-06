#!/usr/bin/env python3
"""
Test script for the simple property valuation API.
This script verifies the functionality of the POST /api/valuation endpoint.
"""
import json
import sys
import requests
import time
from datetime import datetime

def test_simple_valuation_api(base_url, debug=False):
    """
    Test the simple valuation API endpoint by sending a sample request.
    
    Args:
        base_url (str): The base URL of the API (e.g., http://localhost:5002)
        debug (bool, optional): Whether to print debug information
    
    Returns:
        dict: The API response and metadata
    """
    endpoint = f"{base_url}/api/valuation"
    
    # Sample property data for valuation
    sample_property = {
        "square_feet": 2200,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "year_built": 2005,
        "latitude": 46.2804,
        "longitude": -119.2752,
        "city": "Richland",
        "neighborhood": "Meadow Springs",
        "use_gis": True
    }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Start the timer for performance measurement
    start_time = time.time()
    
    try:
        if debug:
            print(f"Sending POST request to {endpoint}")
            print(f"Headers: {headers}")
            print(f"Request payload: {json.dumps(sample_property, indent=2)}")
        
        # Send the POST request to the API
        response = requests.post(
            endpoint,
            headers=headers,
            json=sample_property,
            timeout=30  # 30-second timeout
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Print debug information if requested
        if debug:
            print(f"Status code: {response.status_code}")
            print(f"Response time: {response_time:.2f} seconds")
            
            if response.status_code == 200:
                print(f"Response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"Error response: {response.text}")
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and return the response data
        result = {
            "status": "success",
            "status_code": response.status_code,
            "response_time": response_time,
            "data": response.json(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n{'='*40}")
        print(f"API Test Summary ({result['status']}):")
        print(f"Status Code: {result['status_code']}")
        print(f"Response Time: {result['response_time']:.2f} seconds")
        predicted_value = result['data']['valuation']['predicted_value']
        print(f"Predicted Value: ${predicted_value:,.2f}")
        if 'r2_score' in result['data']['valuation']:
            print(f"R² Score: {result['data']['valuation']['r2_score']:.4f}")
        print(f"Model Type: {result['data']['valuation']['model_type']}")
        print(f"{'='*40}\n")
        
        return result
        
    except requests.exceptions.RequestException as e:
        # Handle request errors
        error_time = time.time() - start_time
        
        result = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "response_time": error_time,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n{'='*40}")
        print(f"API Test ERROR:")
        print(f"Error Type: {result['error_type']}")
        print(f"Error Message: {result['error_message']}")
        print(f"Response Time: {result['response_time']:.2f} seconds")
        print(f"{'='*40}\n")
        
        return result

if __name__ == "__main__":
    # Get command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the simple valuation API")
    parser.add_argument("--url", default="http://localhost:5002", help="Base URL of the API")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    # Run the test
    result = test_simple_valuation_api(args.url, args.debug)
    
    # Set exit code based on result
    sys.exit(0 if result["status"] == "success" else 1)