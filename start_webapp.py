#!/usr/bin/env python3
import os
import requests
import json
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# API base URL
API_BASE_URL = "http://localhost:5000"  # API server running on port 5000

# API Key for backend communication
API_KEY = os.environ.get("BCBS_VALUES_API_KEY", "")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/valuation')
def valuation():
    return render_template('index.html')  # For now, use the same template

# Proxy API endpoints to avoid CORS issues
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """
    API proxy to forward requests to the API server and return the response.
    This avoids CORS issues by having the frontend and API on the same domain.
    """
    # Build the target URL
    url = f"{API_BASE_URL}/api/{path}"
    
    # Forward the request headers
    headers = {
        key: value for key, value in request.headers if key != 'Host'
    }
    
    # Add API key for authentication
    headers["X-API-KEY"] = API_KEY
    
    try:
        # Forward the request with appropriate method
        if request.method == 'GET':
            resp = requests.get(url, headers=headers, params=request.args)
        elif request.method == 'POST':
            resp = requests.post(url, headers=headers, json=request.json)
        elif request.method == 'PUT':
            resp = requests.put(url, headers=headers, json=request.json)
        elif request.method == 'DELETE':
            resp = requests.delete(url, headers=headers)
        else:
            return jsonify({"error": "Method not supported"}), 405
        
        # If the API is down or not functioning, provide development data
        if resp.status_code >= 500 or resp.status_code == 404:
            if path == "valuations":
                # Return sample data for development
                return jsonify([
                    {
                        "property_id": "dev-sample-1",
                        "address": "123 Main St, Richland, WA 99352",
                        "estimated_value": 350000.0,
                        "confidence_score": 0.85,
                        "model_used": "development_model",
                        "valuation_date": "2025-03-30T00:00:00",
                        "features_used": {
                            "bedrooms": 3,
                            "bathrooms": 2,
                            "square_feet": 1800,
                            "year_built": 2005
                        }
                    },
                    {
                        "property_id": "dev-sample-2",
                        "address": "456 Oak Ave, Kennewick, WA 99336",
                        "estimated_value": 420000.0,
                        "confidence_score": 0.82,
                        "model_used": "development_model",
                        "valuation_date": "2025-03-30T00:00:00",
                        "features_used": {
                            "bedrooms": 4,
                            "bathrooms": 2.5,
                            "square_feet": 2200,
                            "year_built": 2010
                        }
                    }
                ])
            else:
                return jsonify({"error": "API endpoint not available"}), 503
        
        # Return the API response
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type', 'application/json')
        )
    except requests.RequestException:
        # Handle connection errors (API server down)
        if path == "valuations":
            # Return sample data if the valuations endpoint is requested
            return jsonify([
                {
                    "property_id": "dev-sample-1",
                    "address": "123 Main St, Richland, WA 99352",
                    "estimated_value": 350000.0,
                    "confidence_score": 0.85,
                    "model_used": "development_model",
                    "valuation_date": "2025-03-30T00:00:00",
                    "features_used": {
                        "bedrooms": 3,
                        "bathrooms": 2,
                        "square_feet": 1800,
                        "year_built": 2005
                    }
                },
                {
                    "property_id": "dev-sample-2",
                    "address": "456 Oak Ave, Kennewick, WA 99336",
                    "estimated_value": 420000.0,
                    "confidence_score": 0.82,
                    "model_used": "development_model",
                    "valuation_date": "2025-03-30T00:00:00",
                    "features_used": {
                        "bedrooms": 4,
                        "bathrooms": 2.5,
                        "square_feet": 2200,
                        "year_built": 2010
                    }
                }
            ])
        else:
            return jsonify({"error": "API server not available"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
