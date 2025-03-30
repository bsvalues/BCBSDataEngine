"""
Test script for the basic property valuation model.

This script demonstrates how to use the train_basic_valuation_model function
to train a basic linear regression model for property valuation.
"""
import pandas as pd
import numpy as np
import logging
import sys
from src.valuation import train_basic_valuation_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('basic_valuation_result.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Test the basic property valuation model with sample data."""
    logger.info("Creating sample property data")
    
    # Create a DataFrame with sample property data
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    
    # Generate sample property data
    property_data = {
        'property_id': [f'P{i:03}' for i in range(1, n_samples + 1)],
        'square_feet': np.random.normal(1800, 500, n_samples).clip(min=800, max=3500).astype(int),
        'bedrooms': np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.4, 0.4, 0.1]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'lot_size': np.random.normal(7500, 2500, n_samples).clip(min=2500).astype(int),
        'garage_spaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.2, 0.6, 0.1]),
        'latitude': np.random.uniform(46.25, 46.35, n_samples),  # Benton County, WA area
        'longitude': np.random.uniform(-119.35, -119.25, n_samples),  # Benton County, WA area
    }
    
    # Create a price model with some noise
    # Base price + square footage contribution + bedroom contribution + bathroom contribution
    # + age contribution + lot size contribution + garage contribution + random noise
    base_price = 150000
    price_per_sqft = 125
    bedroom_value = 15000
    bathroom_value = 20000
    age_penalty = -1000  # per year from 2023
    lot_size_value = 5  # per square foot
    garage_value = 10000
    noise_scale = 30000
    
    current_year = 2023
    
    property_data['list_price'] = (
        base_price +
        property_data['square_feet'] * price_per_sqft +
        property_data['bedrooms'] * bedroom_value +
        property_data['bathrooms'] * bathroom_value +
        (property_data['year_built'] - current_year) * age_penalty +
        property_data['lot_size'] * lot_size_value +
        property_data['garage_spaces'] * garage_value +
        np.random.normal(0, noise_scale, n_samples)
    ).clip(min=100000).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(property_data)
    
    logger.info(f"Created sample dataset with {len(df)} properties")
    logger.info(f"Price range: ${df['list_price'].min():,} - ${df['list_price'].max():,}")
    
    # Display a few sample properties
    logger.info("\nSample properties:")
    for _, row in df.head(5).iterrows():
        logger.info(
            f"ID: {row['property_id']}, "
            f"{row['bedrooms']}bd/{row['bathrooms']}ba, "
            f"{row['square_feet']} sqft, "
            f"Built: {row['year_built']}, "
            f"Price: ${row['list_price']:,}"
        )
    
    # Train the basic valuation model
    logger.info("\nTraining basic valuation model...")
    result = train_basic_valuation_model(df)
    
    # Display model performance
    logger.info(f"\nModel performance:")
    logger.info(f"R² Score: {result['r2_score']:.4f}")
    logger.info(f"Mean Absolute Error: ${result['mae']:,.2f}")
    logger.info(f"Root Mean Squared Error: ${result['rmse']:,.2f}")
    
    # Display feature importance
    logger.info("\nFeature importance:")
    for feature, importance in sorted(
        result['feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Display model coefficients
    logger.info("\nModel coefficients:")
    for feature, coef in sorted(
        result['feature_coefficients'].items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    ):
        logger.info(f"  {feature}: {coef:,.2f}")
    
    # Display predictions for a few properties
    logger.info("\nSample predictions:")
    predictions = result['predictions']
    for _, row in predictions.head(5).iterrows():
        logger.info(
            f"ID: {row['property_id']}, "
            f"Actual: ${row['list_price']:,}, "
            f"Predicted: ${row['predicted_value']:,.2f}, "
            f"Error: ${row['prediction_error']:,.2f} ({row['error_percentage']:.2f}%)"
        )
    
    # Calculate average error percentage
    avg_error_pct = predictions['error_percentage'].abs().mean()
    logger.info(f"\nAverage absolute error percentage: {avg_error_pct:.2f}%")
    
    # Show properties with highest error
    logger.info("\nProperties with highest prediction errors:")
    highest_errors = predictions.loc[predictions['error_percentage'].abs().nlargest(3).index]
    for _, row in highest_errors.iterrows():
        logger.info(
            f"ID: {row['property_id']}, "
            f"{row['bedrooms']}bd/{row['bathrooms']}ba, "
            f"{row['square_feet']} sqft, "
            f"Built: {row['year_built']}, "
            f"Actual: ${row['list_price']:,}, "
            f"Predicted: ${row['predicted_value']:,.2f}, "
            f"Error: {row['error_percentage']:.2f}%"
        )
    
    logger.info("\nBasic valuation model test completed")
    return result

if __name__ == "__main__":
    main()