"""
Test script for the enhanced property valuation model.

This script demonstrates how to use the updated estimate_property_value function
with multiple regression capabilities and GIS integration.
"""
import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_valuation_result.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the valuation function
try:
    from src.valuation import estimate_property_value, has_lightgbm
    logger.info(f"Successfully imported valuation functions from src package. LightGBM available: {has_lightgbm}")
except ImportError:
    logger.error("Failed to import valuation functions. Make sure they are accessible in your PYTHONPATH.")
    raise

def create_sample_data(n_samples=50):
    """Create sample property data for testing."""
    logger.info(f"Creating sample dataset with {n_samples} properties")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate basic property features
    square_feet = np.random.normal(2000, 500, n_samples).astype(int).clip(min=800)
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.5, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                              p=[0.05, 0.1, 0.4, 0.2, 0.15, 0.05, 0.05])
    year_built = np.random.randint(1960, 2022, n_samples)
    
    # Generate locations (centered around Richland, WA)
    center_lat, center_lon = 46.2804, -119.2752
    radius = 0.05  # Approx 3-4 miles
    
    # Create random points in a circle
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    distances = radius * np.sqrt(np.random.uniform(0, 1, n_samples))
    
    latitudes = center_lat + distances * np.cos(angles)
    longitudes = center_lon + distances * np.sin(angles)
    
    # Generate neighborhoods
    neighborhoods = np.random.choice(
        ['South Richland', 'North Richland', 'West Richland', 'Kennewick', 'Pasco'], 
        n_samples,
        p=[0.3, 0.2, 0.2, 0.15, 0.15]
    )
    
    # Calculate base property values using a simple formula
    # Base value depends on square footage, bedrooms, bathrooms, and age
    base_values = (
        square_feet * 100 +                     # $100 per sq ft
        bedrooms * 15000 +                      # $15k per bedroom
        bathrooms * 25000 +                     # $25k per bathroom
        (2022 - year_built) * -500 +            # -$500 per year of age
        np.random.normal(0, 20000, n_samples)   # Random noise
    )
    
    # Add neighborhood effects
    neighborhood_multipliers = {
        'South Richland': 1.15,  # Premium area
        'North Richland': 1.0,   # Average
        'West Richland': 1.05,   # Slightly premium
        'Kennewick': 0.9,        # Slightly below average
        'Pasco': 0.85            # Below average
    }
    
    # Apply neighborhood multipliers
    list_prices = np.array([
        base_values[i] * neighborhood_multipliers[neighborhoods[i]]
        for i in range(n_samples)
    ])
    
    # Create DataFrame
    df = pd.DataFrame({
        'property_id': [f'P{i:03d}' for i in range(n_samples)],
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'neighborhood': neighborhoods,
        'latitude': latitudes,
        'longitude': longitudes,
        'list_price': list_prices.clip(min=150000) # Ensure minimum price
    })
    
    logger.info(f"Created sample dataset with price range: ${df['list_price'].min():,.0f} to ${df['list_price'].max():,.0f}")
    return df

def test_basic_valuation():
    """Test the updated valuation engine without GIS data."""
    logger.info("=" * 80)
    logger.info("Testing basic valuation (without GIS)")
    logger.info("=" * 80)
    
    # Create sample data
    properties = create_sample_data(50)
    
    # Create a sample target property 
    target_property = pd.DataFrame({
        'square_feet': [1850],
        'bedrooms': [3],
        'bathrooms': [2.5],
        'year_built': [2005],
    })
    
    # Run simple valuation without GIS
    result = estimate_property_value(
        properties, 
        target_property=target_property,
        use_gis_features=False,
        use_multiple_regression=True,
        include_advanced_metrics=True
    )
    
    # Print results
    logger.info(f"Predicted value: ${result['predicted_value']:,.2f}")
    logger.info(f"R² Score: {result['r2_score']:.4f}")
    
    if 'adj_r2_score' in result:
        logger.info(f"Adjusted R² Score: {result['adj_r2_score']:.4f}")
    
    if 'rmse' in result:
        logger.info(f"RMSE: ${result['rmse']:,.2f}")
    
    if 'mae' in result:
        logger.info(f"MAE: ${result['mae']:,.2f}")
    
    # Print top features by importance
    logger.info("Top features by importance:")
    features = sorted(result['feature_importance'], key=lambda x: x['importance'], reverse=True)
    for feature in features[:5]:  # Top 5 features
        logger.info(f"  - {feature['feature']}: {feature['importance']:.4f} (coef: {feature['coefficient']:.4f})")
    
    # Print statistically significant features
    if 'statistically_significant_features' in result:
        logger.info("Statistically significant features (p < 0.05):")
        for feature in result['statistically_significant_features']:
            p_value = result['p_values'][feature]
            logger.info(f"  - {feature}: p={p_value:.4f}")
    
    return result

def test_enhanced_gis_valuation():
    """Test the updated valuation engine with GIS data."""
    logger.info("=" * 80)
    logger.info("Testing enhanced valuation (with GIS)")
    logger.info("=" * 80)
    
    # Create sample data
    properties = create_sample_data(50)
    
    # Create a sample target property with location
    target_property = pd.DataFrame({
        'square_feet': [1850],
        'bedrooms': [3],
        'bathrooms': [2.5],
        'year_built': [2005],
        'neighborhood': ['South Richland'],
        'latitude': [46.2743],
        'longitude': [-119.2698]
    })
    
    # Define reference points (key locations in the area)
    ref_points = {
        'downtown': {'lat': 46.2804, 'lon': -119.2752, 'weight': 1.0},
        'columbia_river': {'lat': 46.2694, 'lon': -119.2871, 'weight': 0.8},
        'shopping_center': {'lat': 46.2682, 'lon': -119.2546, 'weight': 0.6},
        'hospital': {'lat': 46.2835, 'lon': -119.2834, 'weight': 0.5}
    }
    
    # Define neighborhood ratings
    neighborhood_ratings = {
        'South Richland': 0.9,
        'North Richland': 0.8,
        'West Richland': 0.85,
        'Kennewick': 0.7,
        'Pasco': 0.65
    }
    
    # Run enhanced valuation with GIS
    result = estimate_property_value(
        properties, 
        target_property=target_property,
        use_gis_features=True,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        use_multiple_regression=True,
        include_advanced_metrics=True
    )
    
    # Print results
    logger.info(f"Predicted value: ${result['predicted_value']:,.2f}")
    logger.info(f"R² Score: {result['r2_score']:.4f}")
    
    if 'adj_r2_score' in result:
        logger.info(f"Adjusted R² Score: {result['adj_r2_score']:.4f}")
    
    # Print GIS-specific information if available
    if 'gis_metrics' in result:
        logger.info("GIS metrics:")
        for key, value in result['gis_metrics'].items():
            if isinstance(value, (list, dict)):
                logger.info(f"  - {key}: {value}")
            else:
                logger.info(f"  - {key}: {value}")
    
    return result

def main():
    """Run the test script with multiple valuation scenarios."""
    logger.info("Starting enhanced valuation engine test")
    
    try:
        # Test without GIS
        basic_result = test_basic_valuation()
        
        # Test with GIS
        gis_result = test_enhanced_gis_valuation()
        
        # Compare results
        basic_value = basic_result['predicted_value']
        gis_value = gis_result['predicted_value']
        
        if basic_value and gis_value:
            difference = gis_value - basic_value
            percent_diff = (difference / basic_value) * 100
            
            logger.info("=" * 80)
            logger.info("Results Comparison")
            logger.info("=" * 80)
            logger.info(f"Basic valuation: ${basic_value:,.2f}")
            logger.info(f"GIS-enhanced valuation: ${gis_value:,.2f}")
            logger.info(f"Difference: ${difference:+,.2f} ({percent_diff:+.2f}%)")
            
            # Effect of location factors
            if 'gis_metrics' in gis_result and 'adjustment_factor' in gis_result['gis_metrics']:
                adjustment = gis_result['gis_metrics']['adjustment_factor']
                if adjustment:
                    logger.info(f"GIS location adjustment factor: {adjustment:.4f}")
                    logger.info(f"Location impact: {(adjustment-1)*100:+.2f}%")
        
        logger.info("Enhanced valuation engine test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during valuation test: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main()