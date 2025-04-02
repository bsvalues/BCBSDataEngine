#!/usr/bin/env python3
"""
Test script for the enhanced BCBS_Values API with advanced metrics.

This script tests the enhanced API endpoints to verify that advanced metrics
are properly returned in the response for both GET and POST operations.
"""
import json
import requests
import time
import sys
import os

# Constants
API_BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("BCBS_VALUES_API_KEY", "sample_test_key")

def test_get_valuations():
    """Test the GET /api/valuations endpoint to ensure it returns enhanced metrics."""
    url = f"{API_BASE_URL}/api/valuations"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": API_KEY
    }
    params = {
        "limit": 3,
        "property_type": "Single Family"
    }
    
    print(f"Testing GET {url} with params {params}...")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        valuations = response.json()
        if not valuations:
            print("No valuations found. Is the database populated?")
            return False
            
        # Check for enhanced metrics in the response
        has_enhanced_metrics = False
        
        print("\nReceived valuations:")
        for valuation in valuations:
            property_id = valuation.get('property_id')
            address = valuation.get('address')
            estimated_value = valuation.get('estimated_value')
            print(f"- Property {property_id}: {address}, Value: ${estimated_value:,.2f}")
            
            # Check if enhanced metrics are included
            if (valuation.get('adj_r2_score') is not None or 
                valuation.get('rmse') is not None or
                valuation.get('feature_importance') or
                valuation.get('gis_factors')):
                has_enhanced_metrics = True
                print("  ✓ Enhanced metrics found for this property")
            
        if has_enhanced_metrics:
            print("\n✓ GET /api/valuations endpoint returns enhanced metrics")
            return True
        else:
            print("\n✗ GET /api/valuations endpoint does not return enhanced metrics")
            return False
            
    except requests.RequestException as e:
        print(f"✗ Error testing GET /api/valuations: {e}")
        return False

def test_get_valuation_by_id():
    """Test the GET /api/valuations/{property_id} endpoint for enhanced metrics."""
    # First get a list of valuation IDs
    try:
        list_response = requests.get(
            f"{API_BASE_URL}/api/valuations", 
            headers={"Accept": "application/json", "X-API-KEY": API_KEY},
            params={"limit": 1}
        )
        list_response.raise_for_status()
        
        valuations = list_response.json()
        if not valuations:
            print("No valuations found to test detail endpoint.")
            return False
            
        # Get the first property ID
        test_property_id = valuations[0].get('property_id')
        
        # Now test the detail endpoint
        url = f"{API_BASE_URL}/api/valuations/{test_property_id}"
        headers = {
            "Accept": "application/json",
            "X-API-KEY": API_KEY
        }
        
        print(f"Testing GET {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        valuation = response.json()
        
        # Check for enhanced metrics in the response
        print(f"\nReceived property details for {valuation.get('address')}:")
        print(f"- Estimated value: ${valuation.get('estimated_value'):,.2f}")
        
        # Print all the advanced metrics
        enhanced_metrics = []
        
        if valuation.get('adj_r2_score') is not None:
            enhanced_metrics.append(f"Adjusted R² Score: {valuation.get('adj_r2_score'):.4f}")
            
        if valuation.get('rmse') is not None:
            enhanced_metrics.append(f"RMSE: {valuation.get('rmse'):.2f}")
            
        if valuation.get('mae') is not None:
            enhanced_metrics.append(f"MAE: {valuation.get('mae'):.2f}")
        
        # Print feature importance
        if valuation.get('feature_importance'):
            importance = valuation.get('feature_importance')
            if isinstance(importance, dict) and importance:
                enhanced_metrics.append("Feature Importance:")
                for feature, value in sorted(
                    importance.items(), 
                    key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0,
                    reverse=True
                )[:5]:  # Top 5 features
                    try:
                        if isinstance(value, (int, float)):
                            enhanced_metrics.append(f"  - {feature}: {value:.4f}")
                        else:
                            enhanced_metrics.append(f"  - {feature}: {value}")
                    except (ValueError, TypeError):
                        enhanced_metrics.append(f"  - {feature}: {value}")
        
        # Print p-values for statistical significance
        if valuation.get('p_values'):
            p_values = valuation.get('p_values')
            if isinstance(p_values, dict) and p_values:
                enhanced_metrics.append("Statistical Significance (p-values):")
                for feature, value in sorted(
                    p_values.items(), 
                    key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 1,
                )[:5]:  # Top 5 most significant features
                    try:
                        if isinstance(value, (int, float)):
                            enhanced_metrics.append(f"  - {feature}: {value:.4f}")
                        else:
                            enhanced_metrics.append(f"  - {feature}: {value}")
                    except (ValueError, TypeError):
                        enhanced_metrics.append(f"  - {feature}: {value}")
        
        # Print GIS factors
        if valuation.get('gis_factors'):
            enhanced_metrics.append("GIS Adjustment Factors:")
            for factor, value in valuation.get('gis_factors').items():
                try:
                    if isinstance(value, (int, float)):
                        enhanced_metrics.append(f"  - {factor}: {value:.4f}")
                    else:
                        enhanced_metrics.append(f"  - {factor}: {value}")
                except (ValueError, TypeError):
                    enhanced_metrics.append(f"  - {factor}: {value}")
                    
        # Print location quality
        if valuation.get('location_quality') is not None:
            enhanced_metrics.append(f"Location Quality Score: {valuation.get('location_quality'):.4f}")
            
        # Print model metrics
        if valuation.get('model_metrics'):
            model_metrics = valuation.get('model_metrics')
            if model_metrics.get('prediction_interval'):
                interval = model_metrics.get('prediction_interval')
                if interval and interval[0] is not None and interval[1] is not None:
                    enhanced_metrics.append(f"Prediction Interval: ${interval[0]:,.2f} - ${interval[1]:,.2f}")
        
        # Print found metrics
        if enhanced_metrics:
            print("\nEnhanced Metrics Found:")
            for metric in enhanced_metrics:
                print(f"  {metric}")
            print("\n✓ GET /api/valuations/{property_id} endpoint returns enhanced metrics")
            return True
        else:
            print("\n✗ GET /api/valuations/{property_id} endpoint does not return enhanced metrics")
            return False
            
    except requests.RequestException as e:
        print(f"✗ Error testing GET /api/valuations/[id]: {e}")
        return False

def test_create_valuation():
    """Test the POST /api/valuations endpoint to ensure it returns enhanced metrics."""
    url = f"{API_BASE_URL}/api/valuations"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-KEY": API_KEY
    }
    
    # Test data for each model type
    model_types = [
        "basic",
        "advanced_linear",
        "advanced_lightgbm",
        "advanced_ensemble",
        "enhanced_gis"
    ]
    
    # Base property data
    property_data = {
        "address": "123 Test St",
        "city": "Richland",
        "state": "WA",
        "zip_code": "99352",
        "property_type": "Single Family",
        "bedrooms": 3,
        "bathrooms": 2,
        "square_feet": 1800,
        "lot_size": 8500,
        "year_built": 1995,
        "latitude": 46.2804,
        "longitude": -119.2752,
        "use_gis": True
    }
    
    success = True
    
    for model_type in model_types:
        # Deep copy property data and add model type
        payload = property_data.copy()
        payload["model_type"] = model_type
        payload["address"] = f"{model_type.capitalize()} Test Property"
        
        print(f"\nTesting POST {url} with model_type={model_type}...")
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            valuation = response.json()
            
            # Check for enhanced metrics in the response
            print(f"Received valuation for {valuation.get('address')}:")
            print(f"- Estimated value: ${valuation.get('estimated_value'):,.2f}")
            print(f"- Model used: {valuation.get('model_used')}")
            
            # Check if enhanced metrics are included based on model type
            has_appropriate_metrics = False
            
            if model_type == "basic":
                # Basic should have at least confidence score
                has_appropriate_metrics = valuation.get('confidence_score') is not None
                print(f"- Confidence score: {valuation.get('confidence_score'):.4f}")
                
            elif model_type in ["advanced_linear", "advanced_lightgbm", "advanced_ensemble", "enhanced_gis"]:
                # Advanced models should have metrics like adj_r2_score, rmse, feature importance
                metrics_found = []
                
                if valuation.get('adj_r2_score') is not None:
                    metrics_found.append(f"Adjusted R² Score: {valuation.get('adj_r2_score'):.4f}")
                    
                if valuation.get('rmse') is not None:
                    metrics_found.append(f"RMSE: {valuation.get('rmse'):.2f}")
                    
                if valuation.get('mae') is not None:
                    metrics_found.append(f"MAE: {valuation.get('mae'):.2f}")
                
                if valuation.get('feature_importance'):
                    metrics_found.append("Feature Importance: Available")
                    
                if valuation.get('feature_coefficients'):
                    metrics_found.append("Feature Coefficients: Available")
                    
                if valuation.get('p_values'):
                    metrics_found.append("P-Values: Available")
                    
                if model_type in ["enhanced_gis"] and valuation.get('gis_factors'):
                    metrics_found.append("GIS Factors: Available")
                    
                if model_type in ["enhanced_gis"] and valuation.get('location_quality') is not None:
                    metrics_found.append(f"Location Quality: {valuation.get('location_quality'):.4f}")
                
                # Print found metrics
                if metrics_found:
                    print("\nEnhanced Metrics Found:")
                    for metric in metrics_found:
                        print(f"  - {metric}")
                        
                # Check if we have appropriate metrics for the model type
                if model_type == "enhanced_gis":
                    has_appropriate_metrics = any(m.startswith("GIS Factors") or m.startswith("Location Quality") for m in metrics_found)
                else:
                    has_appropriate_metrics = len(metrics_found) >= 3  # At least 3 advanced metrics
            
            if has_appropriate_metrics:
                print(f"✓ POST /api/valuations with {model_type} returns appropriate enhanced metrics")
            else:
                print(f"✗ POST /api/valuations with {model_type} does not return appropriate enhanced metrics")
                success = False
                
        except requests.RequestException as e:
            print(f"✗ Error testing POST /api/valuations with {model_type}: {e}")
            success = False
    
    return success

def main():
    """Run all tests for the enhanced API endpoints."""
    print("=== Testing Enhanced BCBS_Values API ===\n")
    
    results = []
    
    # Test GET list endpoint
    results.append(("GET /api/valuations", test_get_valuations()))
    
    # Test GET detail endpoint
    results.append(("GET /api/valuations/{property_id}", test_get_valuation_by_id()))
    
    # Test POST endpoint with different model types
    results.append(("POST /api/valuations", test_create_valuation()))
    
    # Print summary
    print("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        print(f"{test_name}: {status}")
    
    # Set exit code based on results
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()