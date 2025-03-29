"""
Basic test script for simple property valuation.
"""
import logging
import pandas as pd
import numpy as np
from db.database import Database
from src.valuation import estimate_property_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Test the basic property valuation model with real property data."""
    try:
        # Connect to database
        logger.info("Connecting to database")
        db = Database()
        
        # Retrieve Benton County properties
        logger.info("Retrieving Benton County property data")
        properties = db.get_all_properties(benton_county_only=True)
        
        if properties.empty:
            logger.error("No property data available")
            return
            
        logger.info(f"Retrieved {len(properties)} properties")
        
        # Select a sample property to estimate its value
        sample_property_index = 0
        sample_property = properties.iloc[[sample_property_index]].copy()
        
        # Remove the price fields from the sample property to simulate
        # trying to predict an unknown property value
        target_property = sample_property.copy()
        for price_col in ['list_price', 'estimated_value', 'last_sale_price']:
            if price_col in target_property.columns:
                target_property[price_col] = None
        
        # Log original property details for comparison
        original_price = None
        price_source = None
        for price_col in ['list_price', 'estimated_value', 'last_sale_price']:
            if price_col in sample_property.columns and not pd.isna(sample_property[price_col].iloc[0]):
                original_price = sample_property[price_col].iloc[0]
                price_source = price_col
                break
                
        if original_price and price_source:
            logger.info(f"Original property {price_source}: ${original_price:,.2f}")
            
        logger.info(f"Property details:")
        logger.info(f"  - Address: {sample_property['address'].iloc[0]}")
        logger.info(f"  - City: {sample_property['city'].iloc[0]}, {sample_property['state'].iloc[0]}")
        logger.info(f"  - Square Feet: {sample_property['square_feet'].iloc[0]}")
        logger.info(f"  - Bedrooms: {sample_property['bedrooms'].iloc[0]}")
        logger.info(f"  - Bathrooms: {sample_property['bathrooms'].iloc[0]}")
        logger.info(f"  - Year Built: {sample_property['year_built'].iloc[0]}")
        
        # Use the remaining properties as training data
        training_data = properties.drop(sample_property.index)
        
        # Run the basic valuation model
        logger.info("\n===== RUNNING BASIC VALUATION MODEL =====")
        result = estimate_property_value(training_data, target_property)
        
        # Display results
        if 'error' in result:
            logger.error(f"Valuation error: {result['error']}")
            return
        
        predicted_value = result.get('predicted_value')
        r2_score = result.get('r2_score')
        
        if predicted_value is not None:
            logger.info(f"\nPredicted property value: ${predicted_value:,.2f}")
            
            # Compare with original price
            if original_price:
                difference = predicted_value - original_price
                percent_diff = (difference / original_price) * 100
                logger.info(f"Difference from original {price_source}: ${difference:,.2f} ({percent_diff:+.2f}%)")
        
        # Performance metrics
        if r2_score is not None:
            logger.info(f"\nModel Performance:")
            logger.info(f"R² score: {r2_score:.4f}")
        
        # Feature importance
        if 'feature_importance' in result and result['feature_importance']:
            logger.info(f"\nFeature Importance:")
            if isinstance(result['feature_importance'], list):
                for i, feat_imp in enumerate(result['feature_importance']):
                    if isinstance(feat_imp, dict):
                        feature = feat_imp.get('feature', 'unknown')
                        importance = feat_imp.get('importance', 0)
                        logger.info(f"  {i+1}. {feature}: {importance:.4f}")
                    else:
                        logger.info(f"  {i+1}. {feat_imp}")
            else:
                logger.info(f"  {result['feature_importance']}")
                
        # Close database connection
        db.close()
        
    except Exception as e:
        logger.error(f"Error testing valuation model: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()