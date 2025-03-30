#!/usr/bin/env python3
"""
Test script for enhanced GIS integration in the BCBS_Values system.

This script tests the integration of the enhanced GIS module with the property
valuation system. It creates sample property data, generates reference points,
and tests both the basic and advanced valuation models with GIS features enabled.
"""
import logging
import numpy as np
import pandas as pd
from src.valuation import (
    estimate_property_value,
    advanced_property_valuation,
    calculate_gis_features
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_enhanced_gis')

def generate_test_data(num_samples=50):
    """Generate realistic sample property data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Base property features
    data = {
        'square_feet': np.random.normal(2000, 500, num_samples).astype(int),
        'bedrooms': np.random.choice([2, 3, 4, 5], num_samples, p=[0.1, 0.5, 0.3, 0.1]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], num_samples),
        'year_built': np.random.randint(1950, 2023, num_samples),
        'lot_size': np.random.normal(10000, 3000, num_samples).astype(int),
    }
    
    # Sample coordinates for Benton County, WA area
    # Roughly: 46.2°N to 46.3°N, -119.35°W to -119.25°W
    data['latitude'] = np.random.uniform(46.2, 46.3, num_samples)
    data['longitude'] = np.random.uniform(-119.35, -119.25, num_samples)
    
    # Generate prices based on features with some random noise
    # Base price calculation
    base_price = (
        150000 +  # Base price
        data['square_feet'] * 100 +  # $100 per sq ft
        data['bedrooms'] * 15000 +   # $15k per bedroom
        data['bathrooms'] * 25000 +  # $25k per bathroom
        (2023 - data['year_built']) * -500  # -$500 per year of age
    )
    
    # Add location-based price variation (properties near center are more valuable)
    center_lat, center_lon = 46.25, -119.30
    distance_from_center = np.sqrt(
        (data['latitude'] - center_lat)**2 + 
        (data['longitude'] - center_lon)**2
    ) * 111  # Approximate conversion to kilometers
    
    # Location affects price (closer to center = higher price)
    location_factor = 1 - (distance_from_center / 10)  # Normalize to 0-1 range
    
    # Apply location factor and add random noise
    data['list_price'] = (
        base_price * (0.8 + 0.4 * location_factor) + 
        np.random.normal(0, 25000, num_samples)
    ).astype(int)
    
    # Create DataFrame and add a unique ID
    df = pd.DataFrame(data)
    df['property_id'] = [f'PROP-{i:04d}' for i in range(num_samples)]
    
    return df

def create_reference_points():
    """Create reference points for GIS feature calculation."""
    return {
        'downtown': {
            'lat': 46.25,
            'lon': -119.30,
            'weight': 1.0
        },
        'mall': {
            'lat': 46.28,
            'lon': -119.28,
            'weight': 0.7
        },
        'school': {
            'lat': 46.23,
            'lon': -119.32,
            'weight': 0.8
        },
        'hospital': {
            'lat': 46.27,
            'lon': -119.33,
            'weight': 0.6
        },
        'park': {
            'lat': 46.26,
            'lon': -119.27,
            'weight': 0.5
        }
    }

def create_neighborhood_ratings():
    """Create sample neighborhood ratings."""
    return {
        'West Richland': 0.85,
        'Kennewick': 0.78,
        'Richland': 0.82,
        'Pasco': 0.75
    }

def create_enhanced_gis_data():
    """Create sample enhanced GIS datasets."""
    # Create flood zone data (random points with risk level)
    flood_zones = pd.DataFrame({
        'latitude': np.random.uniform(46.2, 46.3, 20),
        'longitude': np.random.uniform(-119.35, -119.25, 20),
        'risk_level': np.random.choice([1, 2, 3, 4, 5], 20, p=[0.4, 0.3, 0.2, 0.08, 0.02]),
        'zone_name': np.random.choice(['Zone A', 'Zone B', 'Zone C', 'Zone X'], 20)
    })
    
    # Create school quality data
    schools = pd.DataFrame({
        'latitude': np.random.uniform(46.2, 46.3, 8),
        'longitude': np.random.uniform(-119.35, -119.25, 8),
        'quality_score': np.random.uniform(5, 10, 8),
        'name': [f'School {i}' for i in range(8)]
    })
    
    # Create amenity data
    amenities = pd.DataFrame({
        'latitude': np.random.uniform(46.2, 46.3, 30),
        'longitude': np.random.uniform(-119.35, -119.25, 30),
        'type': np.random.choice(['grocery', 'restaurant', 'shopping', 'cafe', 'gym'], 30),
        'rating': np.random.uniform(3, 5, 30)
    })
    
    return {
        'flood_zones': flood_zones,
        'schools': schools,
        'amenities': amenities
    }

def test_basic_gis_features():
    """Test basic GIS feature calculation."""
    logger.info("=== Testing Basic GIS Feature Calculation ===")
    
    # Generate test data
    properties = generate_test_data(30)
    ref_points = create_reference_points()
    neighborhood_ratings = create_neighborhood_ratings()
    
    # Calculate GIS features
    logger.info("Calculating basic GIS features...")
    properties_with_gis = calculate_gis_features(
        properties, 
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings
    )
    
    # Check which GIS features were added
    original_cols = set(properties.columns)
    new_cols = set(properties_with_gis.columns) - original_cols
    
    logger.info(f"Added GIS features: {sorted(list(new_cols))}")
    
    # Run basic valuation with GIS features
    logger.info("Running basic valuation with GIS features...")
    target_property = properties.iloc[[0]].copy()
    
    result = estimate_property_value(
        properties_with_gis,
        target_property=target_property,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        use_gis_features=True
    )
    
    logger.info(f"Predicted value: ${result['predicted_value']:,.2f}")
    if 'gis_metrics' in result:
        logger.info(f"GIS metrics: {result['gis_metrics']}")
    
    return properties_with_gis, result

def test_enhanced_gis_features():
    """Test enhanced GIS feature calculation."""
    logger.info("\n=== Testing Enhanced GIS Feature Calculation ===")
    
    # Generate test data
    properties = generate_test_data(30)
    ref_points = create_reference_points()
    neighborhood_ratings = create_neighborhood_ratings()
    enhanced_gis_data = create_enhanced_gis_data()
    
    # Calculate enhanced GIS features
    logger.info("Calculating enhanced GIS features...")
    properties_with_enhanced_gis = calculate_gis_features(
        properties, 
        gis_data=enhanced_gis_data,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings
    )
    
    # Check which enhanced GIS features were added
    original_cols = set(properties.columns)
    new_cols = set(properties_with_enhanced_gis.columns) - original_cols
    
    logger.info(f"Added enhanced GIS features: {sorted(list(new_cols))}")
    
    # Run advanced valuation with enhanced GIS features
    logger.info("Running advanced valuation with enhanced GIS features...")
    target_property = properties.iloc[[0]].copy()
    
    result = advanced_property_valuation(
        properties_with_enhanced_gis,
        target_property=target_property,
        gis_data=enhanced_gis_data,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        use_gis_features=True,
        model_type='linear'  # Using linear model for simplicity
    )
    
    logger.info(f"Predicted value: ${result['predicted_value']:,.2f}")
    if 'gis_metrics' in result:
        logger.info(f"GIS metrics: {result['gis_metrics']}")
    
    return properties_with_enhanced_gis, result

def compare_basic_and_enhanced():
    """Compare basic and enhanced GIS feature calculation."""
    logger.info("\n=== Comparing Basic and Enhanced GIS Features ===")
    
    # Generate consistent test data
    properties = generate_test_data(30)
    target_property = properties.iloc[[0]].copy()
    ref_points = create_reference_points()
    neighborhood_ratings = create_neighborhood_ratings()
    enhanced_gis_data = create_enhanced_gis_data()
    
    # Calculate basic GIS features
    logger.info("Running valuation with basic GIS features...")
    basic_result = estimate_property_value(
        properties,
        target_property=target_property,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        use_gis_features=True
    )
    
    # Calculate enhanced GIS features
    logger.info("Running valuation with enhanced GIS features...")
    enhanced_result = advanced_property_valuation(
        properties,
        target_property=target_property,
        gis_data=enhanced_gis_data,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        use_gis_features=True,
        model_type='ensemble'  # Using ensemble model for better accuracy
    )
    
    # Compare the results
    basic_value = basic_result['predicted_value']
    enhanced_value = enhanced_result['predicted_value']
    difference = enhanced_value - basic_value
    percent_diff = (difference / basic_value) * 100
    
    logger.info(f"Basic GIS valuation: ${basic_value:,.2f}")
    logger.info(f"Enhanced GIS valuation: ${enhanced_value:,.2f}")
    logger.info(f"Difference: ${difference:,.2f} ({percent_diff:.2f}%)")
    
    return basic_result, enhanced_result

if __name__ == "__main__":
    logger.info("Starting enhanced GIS integration tests")
    
    # Run tests
    basic_properties, basic_result = test_basic_gis_features()
    enhanced_properties, enhanced_result = test_enhanced_gis_features()
    basic_comparison, enhanced_comparison = compare_basic_and_enhanced()
    
    logger.info("GIS integration tests completed")