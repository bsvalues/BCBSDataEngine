"""
Test script for the enhanced multiple regression property valuation model.

This script demonstrates how to use the train_multiple_regression_model function
to train a more sophisticated model for property valuation with statistical significance
reporting and feature selection.
"""
import pandas as pd
import numpy as np
import logging
import sys
from src.valuation import train_multiple_regression_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multiple_regression_result.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Test the multiple regression model with sample data."""
    logger.info("Creating enhanced sample property data for multiple regression")
    
    # Create a DataFrame with more comprehensive property data
    np.random.seed(42)  # For reproducibility
    n_samples = 150
    
    # Generate sample property data with more features
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
        'property_type': np.random.choice(['Single Family', 'Townhouse', 'Condo', 'Duplex'], n_samples, p=[0.7, 0.1, 0.15, 0.05]),
        'city': np.random.choice(['Richland', 'Kennewick', 'West Richland', 'Pasco'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'neighborhood': np.random.choice(['Downtown', 'Uptown', 'Westside', 'Southside', 'North Hills', 'East End'], n_samples),
        'school_district': np.random.choice(['Richland SD', 'Kennewick SD', 'Pasco SD'], n_samples, p=[0.5, 0.3, 0.2]),
        'stories': np.random.choice([1, 1.5, 2, 3], n_samples, p=[0.3, 0.1, 0.5, 0.1]),
        'basement': np.random.choice(['None', 'Unfinished', 'Finished'], n_samples, p=[0.6, 0.2, 0.2]),
        'pool': np.random.choice(['No', 'Yes'], n_samples, p=[0.9, 0.1]),
        'view': np.random.choice(['None', 'Limited', 'Good', 'Excellent'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    }
    
    # Create more sophisticated price model with feature interactions and nonlinear effects
    # Base price components
    base_price = 150000
    price_per_sqft = 130
    bedroom_value = np.array([0, 0, 15000, 25000, 30000, 32000])  # Diminishing returns for more bedrooms
    bathroom_value = 20000
    
    # Age effects (newer homes worth more, but vintage homes from certain eras also valuable)
    def age_effect(year):
        if year >= 2010:
            return 0  # New homes (no penalty)
        elif year >= 2000:
            return -500 * (2023 - year)  # Slight depreciation for homes from 2000-2009
        elif year >= 1990:
            return -750 * (2023 - year)  # More depreciation for 1990s homes
        elif year >= 1970:
            return -1000 * (2023 - year)  # More for 1970-1989
        elif year >= 1950:
            return -800 * (2023 - year)  # Slightly less for mid-century homes (more desirable)
        else:
            return -600 * (2023 - year)  # Historic homes depreciate less
    
    # Location value multipliers
    city_multipliers = {
        'Richland': 1.1,
        'Kennewick': 1.0,
        'West Richland': 1.05,
        'Pasco': 0.9
    }
    
    # Neighborhood quality ratings
    neighborhood_ratings = {
        'Downtown': 0.9,
        'Uptown': 1.1,
        'Westside': 1.05,
        'Southside': 0.95,
        'North Hills': 1.15,
        'East End': 1.0
    }
    
    # Property type adjustments
    property_type_adjustments = {
        'Single Family': 0,  # baseline
        'Townhouse': -20000,
        'Condo': -30000,
        'Duplex': +25000  # income potential
    }
    
    # View premium
    view_premium = {
        'None': 0,
        'Limited': 5000,
        'Good': 15000,
        'Excellent': 35000
    }
    
    # Calculate base prices
    property_data['list_price'] = [
        # Base components
        base_price +
        row['square_feet'] * price_per_sqft +
        bedroom_value[min(row['bedrooms'], 5)] +
        row['bathrooms'] * bathroom_value +
        age_effect(row['year_built']) +
        row['lot_size'] * 2 +
        row['garage_spaces'] * 15000 +
        
        # Property characteristic adjustments
        property_type_adjustments[row['property_type']] +
        view_premium[row['view']] +
        (25000 if row['pool'] == 'Yes' else 0) +
        (15000 if row['basement'] == 'Finished' else 5000 if row['basement'] == 'Unfinished' else 0) +
        
        # Location adjustments - multiply by city and neighborhood factors
        (city_multipliers[row['city']] * neighborhood_ratings[row['neighborhood']] - 1) * 50000 +
        
        # Add interaction term - larger houses with more bedrooms get premium beyond sum of parts
        (0.5 * row['square_feet'] * row['bedrooms'] / 1000) +
        
        # Add nonlinear effect - houses with balanced bed/bath ratio more valuable
        (10000 if 0.8 <= row['bedrooms'] / row['bathrooms'] <= 1.5 else 0) +
        
        # Add noise
        np.random.normal(0, 20000)
        
        for _, row in pd.DataFrame(property_data).iterrows()
    ]
    
    # Ensure prices are positive and reasonable
    property_data['list_price'] = np.array(property_data['list_price']).clip(min=150000).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(property_data)
    
    logger.info(f"Created enhanced dataset with {len(df)} properties and {len(df.columns)} features")
    logger.info(f"Price range: ${df['list_price'].min():,} - ${df['list_price'].max():,}")
    
    # Display a few sample properties
    logger.info("\nSample properties:")
    for _, row in df.sample(5).iterrows():
        logger.info(
            f"ID: {row['property_id']}, "
            f"{row['bedrooms']}bd/{row['bathrooms']}ba, "
            f"{row['square_feet']} sqft, "
            f"Built: {row['year_built']}, "
            f"City: {row['city']}, "
            f"Type: {row['property_type']}, "
            f"Price: ${row['list_price']:,}"
        )
    
    # Train the multiple regression model
    logger.info("\nTraining multiple regression model with feature selection...")
    result = train_multiple_regression_model(df, feature_selection=True, max_features=12)
    
    # Display model performance
    logger.info(f"\nModel performance metrics:")
    logger.info(f"R² Score (test set): {result['r2_score']:.4f}")
    logger.info(f"Adjusted R² Score: {result['adj_r2_score']:.4f}")
    logger.info(f"Mean Absolute Error: ${result['mae']:,.2f}")
    logger.info(f"Root Mean Squared Error: ${result['rmse']:,.2f}")
    logger.info(f"Cross-validation R² scores: {', '.join([f'{score:.4f}' for score in result['cross_val_scores']])}")
    logger.info(f"Mean cross-validation R² score: {result['cross_val_scores'].mean():.4f}")
    
    # Display features and their importance
    logger.info("\nTop 10 features by importance:")
    for feature, importance in sorted(
        result['feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]:
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Display statistically significant features
    logger.info("\nStatistically significant features (p < 0.05):")
    significant_count = 0
    for feature, p_value in sorted(result['p_values'].items(), key=lambda x: x[1]):
        if p_value < 0.05:
            logger.info(f"  {feature}: p={p_value:.4f}")
            significant_count += 1
        if significant_count >= 10:  # Limit to top 10 most significant
            break
            
    # Print the full StatsModels summary if available
    if 'summary_text' in result:
        logger.info(f"\nDetailed StatsModels summary:")
        for line in result['summary_text'].split('\n')[:30]:  # Show first 30 lines
            logger.info(line)
    
    # Display predictions for a few random properties
    logger.info("\nSample predictions:")
    predictions = result['predictions']
    for _, row in predictions.sample(5).iterrows():
        logger.info(
            f"ID: {row['property_id']}, "
            f"City: {row['city']}, "
            f"Type: {row['property_type']}, "
            f"Actual: ${row['list_price']:,}, "
            f"Predicted: ${row['predicted_value']:,.2f}, "
            f"Error: ${row['prediction_error']:,.2f} ({row['error_percentage']:.2f}%)"
        )
    
    # Calculate average error percentage
    avg_error_pct = predictions['error_percentage'].abs().mean()
    logger.info(f"\nAverage absolute error percentage: {avg_error_pct:.2f}%")
    
    logger.info("\nMultiple regression model test completed")
    return result

if __name__ == "__main__":
    main()