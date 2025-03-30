"""
Simple test script for the updated property valuation model.
"""
import pandas as pd
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simple_valuation_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the valuation function
try:
    from src.valuation import estimate_property_value
    logger.info("Successfully imported valuation functions from src package")
except ImportError:
    logger.error("Failed to import valuation functions")
    raise

def create_sample_data(n_samples=15):
    """Create sample property data for testing."""
    logger.info(f"Creating sample dataset with {n_samples} properties")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate property IDs
    property_ids = [f'P{i:03d}' for i in range(n_samples)]
    
    # Generate property features
    square_feet = np.random.normal(2000, 500, n_samples).astype(int).clip(min=800)
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.5, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                p=[0.05, 0.1, 0.4, 0.2, 0.15, 0.05, 0.05])
    year_built = np.random.randint(1970, 2020, n_samples)
    
    # Calculate property values based on a simplified formula
    list_prices = (
        square_feet * 120 +                # $120 per sq ft base
        bedrooms * 12000 +                 # $12k per bedroom
        bathrooms * 18000 +                # $18k per bathroom
        (2023 - year_built) * -800 +       # Depreciation by age
        np.random.normal(0, 15000, n_samples)  # Random variation
    ).astype(int).clip(min=150000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'property_id': property_ids,
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'list_price': list_prices
    })
    
    logger.info(f"Created sample dataset with price range: ${df['list_price'].min():,.0f} to ${df['list_price'].max():,.0f}")
    return df

def main():
    """Test the updated valuation engine with a larger test case."""
    logger.info("Starting simple valuation test")
    
    # Create a larger test dataset
    properties = create_sample_data(15)  # 15 properties should be enough
    
    # Create a sample target property
    target_property = pd.DataFrame({
        'square_feet': [1750],
        'bedrooms': [3],
        'bathrooms': [2],
        'year_built': [2000]
    })
    
    try:
        # Run basic valuation without GIS
        logger.info("Running property valuation with multiple regression")
        result = estimate_property_value(
            properties, 
            target_property=target_property,
            use_gis_features=False,  # No GIS for simplicity
            use_multiple_regression=True,
            include_advanced_metrics=True
        )
        
        # Check if we got an error in the result
        if 'error' in result:
            logger.error(f"Valuation returned an error: {result['error']}")
            for key, value in result.items():
                if key != 'error' and value is not None:
                    logger.info(f"{key}: {value}")
            return
        
        # Print results
        if result['predicted_value'] is not None:
            logger.info(f"Predicted value: ${result['predicted_value']:,.2f}")
        else:
            logger.info("No predicted value returned")
            
        if result['r2_score'] is not None:
            logger.info(f"R² Score: {result['r2_score']:.4f}")
        
        if 'adj_r2_score' in result and result['adj_r2_score'] is not None:
            logger.info(f"Adjusted R² Score: {result['adj_r2_score']:.4f}")
            
        if 'rmse' in result and result['rmse'] is not None:
            logger.info(f"RMSE: ${result['rmse']:,.2f}")
            
        if 'p_values' in result and result['p_values'] is not None:
            logger.info("P-values for features:")
            for feature, p_value in result['p_values'].items():
                if feature != 'const':  # Skip the intercept
                    logger.info(f"  - {feature}: {p_value:.4f}")
        
        if result['feature_importance'] is not None:
            logger.info("Feature importance:")
            for feature in result['feature_importance']:
                logger.info(f"  - {feature['feature']}: {feature['importance']:.4f}")
        
        logger.info("Simple valuation test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during valuation test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()