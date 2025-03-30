"""
Test script for the enhanced valuation engine.

This script generates some sample property data and tests the enhanced
valuation engine with various model types and spatial adjustments.
"""

import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the valuation engine
from src.valuation import estimate_property_value

def generate_sample_data(n_samples=100):
    """Generate sample property data for testing."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate basic property features
    square_feet = np.random.normal(2000, 500, n_samples).astype(int)
    square_feet = np.clip(square_feet, 800, 4000)  # Clip to reasonable range
    
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.5, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                               p=[0.05, 0.1, 0.25, 0.3, 0.2, 0.05, 0.05])
    
    # Generate year built (between 1950 and 2020)
    year_built = np.random.randint(1950, 2021, n_samples)
    
    # Generate locations (centered around Richland, WA)
    latitude = np.random.normal(46.2800, 0.05, n_samples)
    longitude = np.random.normal(-119.2800, 0.05, n_samples)
    
    # Create neighborhoods
    neighborhoods = np.random.choice(
        ['Meadow Springs', 'Horn Rapids', 'Queensgate', 'Southridge', 'Downtown Richland'],
        n_samples,
        p=[0.3, 0.2, 0.2, 0.15, 0.15]
    )
    
    # Create cities
    cities = np.random.choice(
        ['Richland', 'Kennewick', 'West Richland', 'Pasco'],
        n_samples,
        p=[0.4, 0.3, 0.15, 0.15]
    )
    
    # Generate property IDs
    property_ids = [f"TEST{i:06d}" for i in range(n_samples)]
    
    # Calculate synthetic property prices based on features
    # Basic pricing model: $100 per sq ft + bedroom and bathroom premiums
    base_price = square_feet * 100
    bedroom_premium = bedrooms * 10000
    bathroom_premium = bathrooms * 15000
    
    # Age discount: newer homes are worth more
    age = 2025 - year_built
    age_factor = np.clip(1 - (age / 100), 0.7, 1.0)
    
    # Location adjustment
    neighborhood_premium = np.zeros(n_samples)
    for i, neighborhood in enumerate(neighborhoods):
        if neighborhood == 'Meadow Springs':
            neighborhood_premium[i] = 50000
        elif neighborhood == 'Horn Rapids':
            neighborhood_premium[i] = 40000
        elif neighborhood == 'Queensgate':
            neighborhood_premium[i] = 30000
        elif neighborhood == 'Southridge':
            neighborhood_premium[i] = 25000
        elif neighborhood == 'Downtown Richland':
            neighborhood_premium[i] = 20000
    
    # Final price calculation with some random noise
    list_price = (base_price + bedroom_premium + bathroom_premium) * age_factor + neighborhood_premium
    list_price = list_price * np.random.normal(1, 0.1, n_samples)  # Add noise
    list_price = np.round(list_price, -3)  # Round to nearest thousand
    
    # Create DataFrame
    data = pd.DataFrame({
        'property_id': property_ids,
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'latitude': latitude,
        'longitude': longitude,
        'neighborhood': neighborhoods,
        'city': cities,
        'list_price': list_price
    })
    
    return data

def test_basic_valuation():
    """Test the basic valuation functionality."""
    
    logger.info("=== Testing Basic Linear Regression Valuation ===")
    
    # Generate sample data
    property_data = generate_sample_data(n_samples=200)
    logger.info(f"Generated {len(property_data)} sample properties")
    
    # Create a target property for valuation
    target_property = {
        'square_feet': 2200,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'year_built': 2010,
        'latitude': 46.2804,
        'longitude': -119.2752,
        'neighborhood': 'Meadow Springs',
        'city': 'Richland'
    }
    
    # Run basic valuation
    result = estimate_property_value(
        property_data=property_data,
        target_property=target_property,
        use_multiple_regression=False,
        include_advanced_metrics=True
    )
    
    # Print results
    logger.info(f"Estimated value: ${result['estimated_value']:,.2f}")
    logger.info(f"R-squared: {result['r2_score']:.4f}")
    logger.info(f"Adjusted R-squared: {result['adj_r2_score']:.4f}")
    logger.info("Feature importance:")
    for feature in result['feature_importance']:
        logger.info(f"  - {feature['feature']}: {feature['importance']:.4f} (coef: {feature['coefficient']:.4f})")
        
    # Check if confidence interval is present
    if 'confidence_interval' in result:
        logger.info(f"95% Confidence interval: ${result['confidence_interval'][0]:,.2f} - ${result['confidence_interval'][1]:,.2f}")
    
    return result

def test_advanced_valuation():
    """Test the advanced valuation functionality with multiple models."""
    
    logger.info("=== Testing Advanced Multiple Regression Valuation ===")
    
    # Generate sample data
    property_data = generate_sample_data(n_samples=200)
    
    # Create a target property for valuation
    target_property = {
        'square_feet': 2200,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'year_built': 2010,
        'latitude': 46.2804,
        'longitude': -119.2752,
        'neighborhood': 'Meadow Springs',
        'city': 'Richland'
    }
    
    # Define reference points for GIS integration
    ref_points = {
        'downtown_richland': {
            'lat': 46.2804, 
            'lon': -119.2752, 
            'weight': 1.0
        },
        'downtown_kennewick': {
            'lat': 46.2112, 
            'lon': -119.1367, 
            'weight': 0.9
        },
        'downtown_pasco': {
            'lat': 46.2395, 
            'lon': -119.1005, 
            'weight': 0.8
        }
    }
    
    # Define neighborhood ratings
    neighborhood_ratings = {
        'Richland': 1.15,
        'West Richland': 1.05,
        'Kennewick': 1.0,
        'Pasco': 0.95,
        'Meadow Springs': 1.2,
        'Horn Rapids': 1.1,
        'Queensgate': 1.15,
        'Southridge': 1.05,
        'Downtown Richland': 1.1,
        'Unknown': 1.0
    }
    
    # Test different model types
    model_types = ['linear', 'ridge', 'gbr']
    for model_type in model_types:
        logger.info(f"\nTesting model type: {model_type}")
        
        # Run valuation with GIS integration
        result = estimate_property_value(
            property_data=property_data,
            target_property=target_property,
            use_multiple_regression=True,
            include_advanced_metrics=True,
            use_gis_features=True,
            ref_points=ref_points,
            neighborhood_ratings=neighborhood_ratings,
            model_type=model_type,
            feature_selection_method='all',
            spatial_adjustment_method='multiplicative',
            handle_outliers=True,
            handle_missing_values=True
        )
        
        # Print results
        logger.info(f"Model: {result.get('model_name', 'Unknown')}")
        if 'estimated_value' in result:
            logger.info(f"Estimated value: ${result['estimated_value']:,.2f}")
        
        if 'r_squared' in result:
            logger.info(f"R-squared: {result['r_squared']:.4f}")
        elif 'r2_score' in result:
            logger.info(f"R-squared: {result['r2_score']:.4f}")
            
        if 'adjusted_r_squared' in result:
            logger.info(f"Adjusted R-squared: {result['adjusted_r_squared']:.4f}")
        elif 'adj_r2_score' in result:
            logger.info(f"Adjusted R-squared: {result['adj_r2_score']:.4f}")
        
        # Print top 5 feature importances
        if 'feature_importances' in result:
            logger.info("Top 5 feature importances:")
            for i, feature in enumerate(result['feature_importances'][:5]):
                logger.info(f"  {i+1}. {feature['feature']}: {feature['importance']:.4f}")
        
        # Print confidence interval if available
        if 'confidence_interval' in result:
            ci = result['confidence_interval']
            logger.info(f"Confidence interval: ${ci[0]:,.2f} - ${ci[1]:,.2f}")
    
    return result

def test_what_if_scenarios():
    """Test what-if scenarios with spatial adjustments."""
    
    logger.info("=== Testing What-If Scenarios with Spatial Adjustments ===")
    
    # Generate sample data
    property_data = generate_sample_data(n_samples=200)
    
    # Create a base target property
    base_property = {
        'square_feet': 2200,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'year_built': 2010,
        'latitude': 46.2804,
        'longitude': -119.2752,
        'neighborhood': 'Meadow Springs',
        'city': 'Richland'
    }
    
    # Define neighborhood ratings
    neighborhood_ratings = {
        'Richland': 1.15,
        'West Richland': 1.05,
        'Kennewick': 1.0,
        'Pasco': 0.95,
        'Meadow Springs': 1.2,
        'Horn Rapids': 1.1,
        'Queensgate': 1.15,
        'Southridge': 1.05,
        'Downtown Richland': 1.1,
        'Unknown': 1.0
    }
    
    # Get baseline valuation
    baseline = estimate_property_value(
        property_data=property_data,
        target_property=base_property,
        model_type='linear',
        use_gis_features=True,
        neighborhood_ratings=neighborhood_ratings
    )
    
    # Check for error in result
    if 'error' in baseline:
        logger.error(f"Error in baseline valuation: {baseline['error']}")
        return
        
    if 'estimated_value' not in baseline or baseline['estimated_value'] is None:
        logger.error("No estimated value in baseline result")
        return
        
    logger.info(f"Baseline property value: ${baseline['estimated_value']:,.2f}")
    
    # Test what-if scenarios
    scenarios = [
        {"name": "Larger House", "changes": {"square_feet": 2500}},
        {"name": "Extra Bedroom", "changes": {"bedrooms": 4}},
        {"name": "Extra Bathroom", "changes": {"bathrooms": 3.5}},
        {"name": "Newer Construction", "changes": {"year_built": 2020}},
        {"name": "Different Neighborhood", "changes": {"neighborhood": "Horn Rapids"}},
        {"name": "Different City", "changes": {"city": "Kennewick"}},
    ]
    
    # Run each scenario
    for scenario in scenarios:
        # Create a copy of the base property with changes
        property_copy = base_property.copy()
        property_copy.update(scenario["changes"])
        
        # Get valuation for this scenario
        result = estimate_property_value(
            property_data=property_data,
            target_property=property_copy,
            model_type='linear',
            use_gis_features=True,
            neighborhood_ratings=neighborhood_ratings
        )
        
        # Print results
        logger.info(f"\nScenario: {scenario['name']}")
        logger.info(f"Changes: {scenario['changes']}")
        
        # Check for error in result
        if 'error' in result:
            logger.error(f"Error in scenario valuation: {result['error']}")
            continue
            
        if 'estimated_value' not in result or result['estimated_value'] is None:
            logger.error("No estimated value in scenario result")
            continue
        
        # Calculate difference
        value_diff = result['estimated_value'] - baseline['estimated_value']
        percent_diff = (value_diff / baseline['estimated_value']) * 100
        
        logger.info(f"New estimated value: ${result['estimated_value']:,.2f}")
        logger.info(f"Difference: ${value_diff:,.2f} ({percent_diff:+.2f}%)")

def main():
    """Main function to run all tests."""
    logger.info("Starting valuation engine tests")
    logger.info("=" * 80)
    
    # Run tests
    test_basic_valuation()
    logger.info("=" * 80)
    
    test_advanced_valuation()
    logger.info("=" * 80)
    
    test_what_if_scenarios()
    logger.info("=" * 80)
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main()