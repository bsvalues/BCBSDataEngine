"""
Simple test for advanced valuation.
"""

import logging
import numpy as np
import pandas as pd

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
    
    # Generate basic property features
    square_feet = np.random.normal(2000, 500, n_samples).astype(int)
    square_feet = np.clip(square_feet, 800, 4000)  # Clip to reasonable range
    
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.5, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                               p=[0.05, 0.1, 0.25, 0.3, 0.2, 0.05, 0.05])
    
    # Generate year built (between 1950 and 2022)
    year_built = np.random.randint(1950, 2023, n_samples)
    
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
    
    # Generate synthetic property prices based on features
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

def test_advanced_valuation():
    """Test the advanced valuation functionality."""
    
    logger.info("=== Testing Advanced Regression Valuation ===")
    
    # Generate sample data
    property_data = generate_sample_data(n_samples=50)
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
    
    # Define reference points
    ref_points = {
        'downtown': {
            'lat': 46.2804, 
            'lon': -119.2752, 
            'weight': 1.0
        }
    }
    
    # Run advanced valuation
    result = estimate_property_value(
        property_data=property_data,
        target_property=target_property,
        use_multiple_regression=True,
        include_advanced_metrics=True,
        ref_points=ref_points,
        model_type='linear',
        feature_selection_method='all',
        handle_outliers=True,
        handle_missing_values=True,
        cross_validation_folds=3
    )
    
    # Check for error in result
    if 'error' in result:
        logger.error(f"Error in valuation: {result['error']}")
        return
        
    # Print results
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

    # Print top features
    if 'feature_importances' in result:
        logger.info("Top 5 feature importances:")
        for i, feature in enumerate(result['feature_importances'][:5]):
            logger.info(f"  {i+1}. {feature['feature']}: {feature['importance']:.4f}")
    
    # Print confidence interval if available
    if 'confidence_interval' in result:
        ci = result['confidence_interval']
        logger.info(f"Confidence interval: ${ci[0]:,.2f} - ${ci[1]:,.2f}")
    
    # Print cross-validation results if available
    if 'cross_validation' in result:
        cv = result['cross_validation']
        logger.info(f"Cross-validation R²: {cv['mean_r2']:.4f} (±{cv['std_r2']:.4f})")
    
    logger.info("Advanced valuation test completed successfully")
    
    return result

if __name__ == "__main__":
    test_advanced_valuation()