"""
Test script for the property valuation model.
This script demonstrates how to use the property valuation functions
with the property data in our database.
"""
import logging
import pandas as pd
from db.database import Database
from src.valuation import estimate_property_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Test the property valuation model with real property data."""
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
        # Here we'll use the first property in our dataset as an example
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
        
        # Run the valuation model
        logger.info("Running property valuation model")
        result = estimate_property_value(training_data, target_property)
        
        # Display results
        logger.info("\n===== VALUATION RESULTS =====")
        
        if 'error' in result:
            logger.error(f"Valuation error: {result['error']}")
        else:
            predicted_value = result['predicted_value']
            r2_score = result['r2_score']
            
            logger.info(f"Predicted property value: ${predicted_value:,.2f}")
            logger.info(f"Model R-squared score: {r2_score:.4f}")
            
            # Compare with original price if available
            if original_price and price_source:
                difference = predicted_value - original_price
                percent_diff = (difference / original_price) * 100
                logger.info(f"Difference from original {price_source}: ${difference:,.2f} ({percent_diff:+.2f}%)")
            
            # Show feature importance
            logger.info("\nFeature Importance:")
            for feature in result['feature_importance']:
                logger.info(f"  - {feature['feature']}: {feature['importance']:.4f}")
            
            # Show data metrics
            logger.info("\nData Metrics:")
            metrics = result['data_metrics']
            logger.info(f"  - Sample size: {metrics['sample_size']} properties")
            logger.info(f"  - Mean price: ${metrics['mean_price']:,.2f}")
            logger.info(f"  - Median price: ${metrics['median_price']:,.2f}")
            logger.info(f"  - Price range: ${metrics['price_range'][0]:,.2f} to ${metrics['price_range'][1]:,.2f}")
        
        # Close database connection
        db.close()
        
    except Exception as e:
        logger.error(f"Error testing valuation model: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()