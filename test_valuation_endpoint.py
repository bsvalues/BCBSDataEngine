#!/usr/bin/env python3
"""
Test script for the new property valuation API endpoint.
This script verifies the functionality of the POST /api/valuations endpoint.
"""
import json
import os
import sys
import requests
import time
import argparse
from datetime import datetime

def test_valuation_api(base_url, api_key=None, debug=False):
    """
    Test the property valuation API endpoint by sending a sample request.
    
    Args:
        base_url (str): The base URL of the API (e.g., http://localhost:8000)
        api_key (str, optional): The API key for authentication
        debug (bool, optional): Whether to print debug information
    
    Returns:
        dict: The API response and metadata
    """
    endpoint = f"{base_url}/api/valuations"
    
    # Sample property data for valuation
    sample_property = {
        "address": "123 Test St",
        "city": "Richland",
        "state": "WA",
        "zip_code": "99352",
        "property_type": "Single Family",
        "bedrooms": 4,
        "bathrooms": 2.5,
        "square_feet": 2200,
        "lot_size": 8500,
        "year_built": 2005,
        "latitude": 46.2804,
        "longitude": -119.2752,
        "use_gis": True
    }
    
    # Prepare headers with API key if provided
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["X-API-KEY"] = api_key
    
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
        print(f"Estimated Value: ${result['data']['estimated_value']:,.2f}")
        print(f"Confidence Score: {result['data']['confidence_score']:.4f}")
        print(f"Model Used: {result['data']['model_used']}")
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

def main():
    """Parse command line arguments and run the test"""
    parser = argparse.ArgumentParser(description="Test the property valuation API endpoint")
    
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--key", 
        default=os.environ.get("BCBS_VALUES_API_KEY", "sample_test_key"),
        help="API key for authentication (default: value from BCBS_VALUES_API_KEY env var)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Print debug information"
    )
    
    args = parser.parse_args()
    
    # Run the test
    result = test_valuation_api(args.url, args.key, args.debug)
    
    # Exit with appropriate code
    if result["status"] == "error":
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()