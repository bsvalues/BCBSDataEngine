"""
Simplified test script for the advanced property valuation model.
This script tests just one valuation model configuration.
"""
import logging
import pandas as pd
import numpy as np
from db.database import Database
from src.valuation import advanced_property_valuation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Test the advanced property valuation model with real property data."""
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
        
        # Run the valuation model with optimized configuration
        logger.info("\n===== RUNNING OPTIMIZED VALUATION MODEL =====")
        result = advanced_property_valuation(
            training_data, 
            target_property,
            feature_selection='auto',
            poly_degree=1,
            regularization='ridge',
            alpha=0.5
        )
        
        # Display comprehensive results
        display_model_results(result, original_price, price_source)
        
        # Close database connection
        db.close()
        
    except Exception as e:
        logger.error(f"Error testing valuation model: {str(e)}", exc_info=True)


def display_model_results(result, original_price, price_source):
    """Display comprehensive results from a valuation model."""
    if 'error' in result:
        logger.error(f"Valuation error: {result['error']}")
        return
    
    # Basic prediction results
    predicted_value = result.get('predicted_value')
    
    logger.info("\n")
    logger.info("="*60)
    logger.info(f"VALUATION RESULTS")
    logger.info("="*60)
    
    if predicted_value is not None:
        logger.info(f"PREDICTED PROPERTY VALUE: ${predicted_value:,.2f}")
    
    # Compare with original price if available
    if original_price and predicted_value:
        difference = predicted_value - original_price
        percent_diff = (difference / original_price) * 100
        logger.info(f"DIFFERENCE FROM ORIGINAL {price_source.upper()}: ${difference:,.2f} ({percent_diff:+.2f}%)")
    
    # Prediction interval if available
    if 'prediction_interval' in result and result['prediction_interval']:
        interval = result['prediction_interval']
        logger.info(f"95% PREDICTION INTERVAL: ${interval[0]:,.2f} to ${interval[1]:,.2f}")
        
        interval_width = interval[1] - interval[0]
        interval_pct = (interval_width / predicted_value) * 100 if predicted_value else 0
        logger.info(f"Interval width: ${interval_width:,.2f} ({interval_pct:.1f}% of predicted value)")
    
    logger.info("="*60)
    
    # Performance metrics
    logger.info(f"\nModel Performance Metrics:")
    if 'r2_train' in result:
        logger.info(f"R² (train): {result['r2_train']:.4f}")
    if 'r2_score' in result:
        logger.info(f"R² (test): {result['r2_score']:.4f}")
    if 'cross_val_r2' in result and result['cross_val_r2'] is not None:
        logger.info(f"Cross-validation R²: {result['cross_val_r2']:.4f} (±{result.get('cross_val_std', 0):.4f})")
    if 'rmse' in result:
        logger.info(f"RMSE: ${result['rmse']:,.2f}")
    if 'mae' in result:
        logger.info(f"MAE: ${result['mae']:,.2f}")
    
    # Statistical significance if available
    if 'model_statistics' in result:
        stats = result['model_statistics']
        logger.info(f"\nStatistical Significance:")
        if 'f_statistic' in stats and 'f_pvalue' in stats:
            logger.info(f"F-statistic: {stats['f_statistic']:.4f} (p={stats['f_pvalue']:.6f})")
        if 'adj_r2' in stats:
            logger.info(f"Adjusted R²: {stats['adj_r2']:.4f}")
    
    # Data metrics if available
    if 'data_metrics' in result:
        metrics = result['data_metrics']
        logger.info(f"\nData Metrics:")
        logger.info(f"Sample size: {metrics.get('sample_size', 0)}")
        logger.info(f"Training samples: {metrics.get('train_size', 0)}")
        logger.info(f"Test samples: {metrics.get('test_size', 0)}")
        
        if 'mean_price' in metrics:
            logger.info(f"Mean price: ${metrics['mean_price']:,.2f}")
        if 'median_price' in metrics:
            logger.info(f"Median price: ${metrics['median_price']:,.2f}")
        if 'price_range' in metrics:
            price_range = metrics['price_range']
            logger.info(f"Price range: ${price_range[0]:,.2f} to ${price_range[1]:,.2f}")
    
    # Model configuration
    if 'model_config' in result:
        config = result['model_config']
        logger.info(f"\nModel Configuration:")
        for key, value in config.items():
            if value is not None:
                logger.info(f"{key}: {value}")
    
    # Top feature importance
    if 'feature_importance' in result and result['feature_importance']:
        logger.info(f"\nTop Feature Importance:")
        # Show top 10 features
        features = result['feature_importance'][:10] if len(result['feature_importance']) > 10 else result['feature_importance']
        
        for i, feature in enumerate(features):
            feature_name = feature.get('feature', 'Unknown')
            importance = feature.get('importance', 0)
            coefficient = feature.get('coefficient', 0)
            
            significance = ""
            if 'p_value' in feature and feature['p_value'] is not None:
                if feature.get('significant', False):
                    significance = f"(p={feature['p_value']:.4f}) *"
                else:
                    significance = f"(p={feature['p_value']:.4f})"
            
            logger.info(f"{i+1}. {feature_name}: {importance:.4f} (coef: {coefficient:.4f}) {significance}")

            
if __name__ == "__main__":
    main()