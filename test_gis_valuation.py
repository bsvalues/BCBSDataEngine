#!/usr/bin/env python3
"""
Test script for the enhanced GIS-enabled property valuation model.

This script tests the updated estimate_property_value function that includes
GIS data integration, demonstrating how spatial features can influence 
property valuations with location-based adjustments.
"""

import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from src.valuation import estimate_property_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gis_valuation_result.log')
    ]
)

logger = logging.getLogger(__name__)

def create_sample_property_data(num_properties=100, with_gis=True):
    """
    Create sample property data for testing, including GIS coordinates if specified.
    
    This function generates realistic property data for Benton County, WA with 
    prices correlated to features and location.
    
    Args:
        num_properties (int): Number of properties to generate
        with_gis (bool): Whether to include GIS coordinates and attributes
        
    Returns:
        pd.DataFrame: DataFrame containing generated property data
    """
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate basic property features
    square_feet = np.random.normal(1800, 400, num_properties).astype(int)
    bedrooms = np.random.choice([2, 3, 4, 5], num_properties, p=[0.1, 0.5, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], num_properties, p=[0.1, 0.15, 0.4, 0.25, 0.1])
    year_built = np.random.randint(1950, 2023, num_properties)
    property_types = np.random.choice(['Single Family', 'Condo', 'Townhouse'], 
                                    num_properties, p=[0.8, 0.1, 0.1])
    cities = np.random.choice(['Richland', 'Kennewick', 'West Richland'], 
                             num_properties, p=[0.4, 0.4, 0.2])
    
    # Calculate base price (without location factor)
    # Formula: $100/sqft base + bedrooms*$15k + bathrooms*$25k + newer construction premium
    base_price = (
        square_feet * 100 + 
        bedrooms * 15000 + 
        bathrooms * 25000 + 
        (year_built - 1950) * 500
    )
    
    # Create DataFrame
    properties = pd.DataFrame({
        'property_id': [f"P{i:03d}" for i in range(num_properties)],
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'property_type': property_types,
        'city': cities
    })
    
    # Add GIS data if requested
    if with_gis:
        # Generate realistic coordinates for Benton County, WA
        # Approximate bounding box for the area
        min_lat, max_lat = 46.1, 46.4  # Latitude range
        min_lon, max_lon = -119.5, -119.1  # Longitude range
        
        # Generate random coordinates within the bounding box
        latitude = np.random.uniform(min_lat, max_lat, num_properties)
        longitude = np.random.uniform(min_lon, max_lon, num_properties)
        
        # Add coordinates to the DataFrame
        properties['latitude'] = latitude
        properties['longitude'] = longitude
        
        # Create neighborhoods based on geographic clusters
        # Simplified approach: divide area into 9 neighborhoods
        lat_bins = pd.cut(latitude, 3, labels=['North', 'Central', 'South'])
        lon_bins = pd.cut(longitude, 3, labels=['West', 'Mid', 'East'])
        
        # Convert categorical to string before concatenation
        properties['neighborhood'] = (
            lat_bins.astype(str) + "-" + lon_bins.astype(str)
        )
        
        # Create location-based price factors (property values vary by location)
        # Define quality ratings for each area (higher = more desirable/expensive)
        neighborhood_ratings = {
            'North-West': 0.9,    # Good area, slightly above average
            'North-Mid': 1.1,     # Premium area
            'North-East': 1.0,    # Above average
            'Central-West': 0.8,  # Average area
            'Central-Mid': 1.2,   # Best area, most desirable
            'Central-East': 0.95, # Slightly above average
            'South-West': 0.75,   # Below average
            'South-Mid': 0.85,    # Average
            'South-East': 0.9,    # Slightly above average
            'Richland': 1.15,     # Premium city
            'Kennewick': 1.0,     # Average city
            'West Richland': 0.85 # Below average city
        }
        
        # Apply location factors to base price
        # First ensure all neighborhood names are in the ratings dictionary
        for neighborhood in properties['neighborhood'].unique():
            if neighborhood not in neighborhood_ratings:
                # Assign an average value (1.0) for unknown neighborhoods
                neighborhood_ratings[neighborhood] = 1.0
                
        properties['location_factor'] = properties['neighborhood'].map(neighborhood_ratings)
        properties['list_price'] = (base_price * properties['location_factor']).astype(int)
        
        # Create reference points (city centers, schools, etc.)
        ref_points = {
            'downtown': {
                'lat': 46.28, 
                'lon': -119.28, 
                'weight': 1.0  # Importance weight
            },
            'shopping_mall': {
                'lat': 46.22,
                'lon': -119.23,
                'weight': 0.7
            },
            'park': {
                'lat': 46.32,
                'lon': -119.32,
                'weight': 0.5
            }
        }
        
        # Store reference points for later use
        properties.attrs['ref_points'] = ref_points
        properties.attrs['neighborhood_ratings'] = neighborhood_ratings
    else:
        # Without GIS data, use a simpler price model
        properties['list_price'] = base_price.astype(int)
    
    logger.info(f"Created sample dataset with {len(properties)} properties")
    logger.info(f"Price range: ${properties['list_price'].min():,} - ${properties['list_price'].max():,}")
    
    # Display a few sample properties
    logger.info("\nSample properties:")
    for i in range(5):
        prop = properties.iloc[i]
        logger.info(f"ID: {prop['property_id']}, {prop['bedrooms']}bd/{prop['bathrooms']}ba, "
                  f"{prop['square_feet']} sqft, Built: {prop['year_built']}, "
                  f"City: {prop['city']}, Type: {prop['property_type']}, "
                  f"Price: ${prop['list_price']:,}")
    
    return properties

def main():
    """Test the GIS-enhanced property valuation model with sample data."""
    logger.info("=== Testing GIS-Enhanced Property Valuation Model ===")
    
    # Create sample property data with GIS attributes
    properties = create_sample_property_data(num_properties=120, with_gis=True)
    
    # Extract reference points and neighborhood ratings from the generated data
    ref_points = properties.attrs.get('ref_points', None)
    neighborhood_ratings = properties.attrs.get('neighborhood_ratings', None)
    
    # Create a target property to evaluate
    target_property = pd.DataFrame({
        'property_id': ['TARGET'],
        'square_feet': [2000],
        'bedrooms': [3],
        'bathrooms': [2.5],
        'year_built': [2005],
        'latitude': [46.25],
        'longitude': [-119.22],
        'city': ['Kennewick']
    })
    
    logger.info(f"\nTarget property: {target_property['square_feet'][0]} sqft, "
              f"{target_property['bedrooms'][0]}bd/{target_property['bathrooms'][0]}ba, "
              f"Built: {target_property['year_built'][0]}")
    
    # === Test 1: Valuation without GIS data ===
    logger.info("\n=== Test 1: Standard valuation (without GIS data) ===")
    standard_result = estimate_property_value(
        properties, 
        target_property,
        use_gis_features=False  # Disable GIS features
    )
    
    if 'error' in standard_result:
        logger.error(f"Valuation error: {standard_result['error']}")
    else:
        logger.info(f"Standard predicted value: ${standard_result['predicted_value']:,.2f}")
        logger.info(f"Model R² score: {standard_result['r2_score']:.4f}")
        logger.info("\nTop 5 features by importance:")
        for i, feature in enumerate(standard_result['feature_importance'][:5]):
            logger.info(f"  {i+1}. {feature['feature']}: {feature['importance']:.4f}")
    
    # === Test 2: Valuation with GIS data ===
    logger.info("\n=== Test 2: GIS-enhanced valuation ===")
    gis_result = estimate_property_value(
        properties, 
        target_property,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        use_gis_features=True  # Enable GIS features
    )
    
    if 'error' in gis_result:
        logger.error(f"Valuation error: {gis_result['error']}")
    else:
        logger.info(f"GIS-enhanced predicted value: ${gis_result['predicted_value']:,.2f}")
        logger.info(f"Model R² score: {gis_result['r2_score']:.4f}")
        
        # Show GIS-specific metrics if available
        if 'gis_metrics' in gis_result:
            logger.info("\nGIS metrics:")
            logger.info(f"  GIS features used: {gis_result['gis_metrics']['features_used']}")
            logger.info(f"  Location adjustment factor: {gis_result['gis_metrics'].get('adjustment_factor', 'None')}")
        
        logger.info("\nTop 5 features by importance:")
        for i, feature in enumerate(gis_result['feature_importance'][:5]):
            logger.info(f"  {i+1}. {feature['feature']}: {feature['importance']:.4f}")
    
    # === Test 3: Demonstrate impact of different neighborhoods ===
    logger.info("\n=== Test 3: Impact of different neighborhoods on valuation ===")
    
    # Update neighborhood ratings to differentiate locations better
    if neighborhood_ratings:
        neighborhood_ratings['Richland'] = 1.15  # Premium location, 15% higher value
        neighborhood_ratings['Kennewick'] = 1.0  # Average location, no adjustment
        neighborhood_ratings['West Richland'] = 0.85  # Below average, 15% lower value
    
    # Create identical properties in different neighborhoods
    locations = [
        {'name': 'Premium Location', 'lat': 46.29, 'lon': -119.27, 'city': 'Richland'},
        {'name': 'Average Location', 'lat': 46.22, 'lon': -119.31, 'city': 'Kennewick'},
        {'name': 'Below Average Location', 'lat': 46.17, 'lon': -119.42, 'city': 'West Richland'}
    ]
    
    for location in locations:
        # Create a test property with this location
        test_property = pd.DataFrame({
            'property_id': ['TEST'],
            'square_feet': [1900],  # Same features for all test properties
            'bedrooms': [3],
            'bathrooms': [2],
            'year_built': [2000],
            'latitude': [location['lat']],
            'longitude': [location['lon']],
            'city': [location['city']]
        })
        
        # Run valuation with GIS data
        result = estimate_property_value(
            properties, 
            test_property,
            ref_points=ref_points,
            neighborhood_ratings=neighborhood_ratings,
            use_gis_features=True
        )
        
        if 'error' not in result:
            adjustment = result.get('gis_metrics', {}).get('adjustment_factor', 1.0)
            if adjustment is None:
                adjustment = 1.0
            
            logger.info(f"{location['name']}: ${result['predicted_value']:,.2f} "
                      f"(GIS adjustment: {(adjustment-1)*100:+.1f}%)")
    
    logger.info("\n=== GIS Valuation Testing Complete ===")

if __name__ == "__main__":
    main()