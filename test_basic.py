"""
Simple test for basic valuation.
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
    
    # Generate synthetic property prices based on features
    # Basic pricing model: $100 per sq ft + bedroom and bathroom premiums
    base_price = square_feet * 100
    bedroom_premium = bedrooms * 10000
    bathroom_premium = bathrooms * 15000
    
    # Final price calculation with some random noise
    list_price = base_price + bedroom_premium + bathroom_premium
    list_price = list_price * np.random.normal(1, 0.1, n_samples)  # Add noise
    list_price = np.round(list_price, -3)  # Round to nearest thousand
    
    # Create DataFrame
    data = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'list_price': list_price
    })
    
    return data

def test_basic_valuation():
    """Test the basic valuation functionality."""
    
    logger.info("=== Testing Basic Linear Regression Valuation ===")
    
    # Generate sample data
    property_data = generate_sample_data(n_samples=50)
    logger.info(f"Generated {len(property_data)} sample properties")
    
    # Create a target property for valuation
    target_property = {
        'square_feet': 2200,
        'bedrooms': 3,
        'bathrooms': 2.5
    }
    
    # Run basic valuation
    result = estimate_property_value(
        property_data=property_data,
        target_property=target_property,
        use_multiple_regression=False,
        include_advanced_metrics=True
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
    
    # Print feature importances
    if 'feature_importance' in result:
        logger.info("Feature importance:")
        for feature in result['feature_importance']:
            logger.info(f"  - {feature['feature']}: {feature['importance']:.4f} (coef: {feature['coefficient']:.4f})")
        
    # Print confidence interval if available
    if 'confidence_interval' in result:
        ci = result['confidence_interval']
        logger.info(f"95% Confidence interval: ${ci[0]:,.2f} - ${ci[1]:,.2f}")
    
    logger.info("Basic valuation test completed successfully")
    
    return result

if __name__ == "__main__":
    test_basic_valuation()