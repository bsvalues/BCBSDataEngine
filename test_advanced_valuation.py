"""
Test script for the advanced property valuation model.
This script demonstrates how to use the advanced multiple regression analysis
valuation function with real property data.
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
        
        # Run the valuation model with different configurations
        logger.info("\n===== RUNNING BASIC VALUATION =====")
        # Basic with no feature selection, linear features
        result1 = advanced_property_valuation(
            training_data, 
            target_property,
            # Use 'none' instead of None to avoid type errors
            feature_selection='none',
            poly_degree=1,
            regularization=None
        )
        
        logger.info("\n===== RUNNING ADVANCED VALUATION WITH FEATURE SELECTION =====")
        # Advanced with automatic feature selection
        result2 = advanced_property_valuation(
            training_data, 
            target_property,
            feature_selection='auto',
            poly_degree=1,
            regularization=None
        )
        
        logger.info("\n===== RUNNING ADVANCED VALUATION WITH POLYNOMIAL FEATURES =====")
        # Advanced with polynomial features
        result3 = advanced_property_valuation(
            training_data, 
            target_property,
            feature_selection='auto',
            poly_degree=2,
            regularization=None
        )
        
        logger.info("\n===== RUNNING ADVANCED VALUATION WITH REGULARIZATION =====")
        # Advanced with regularization (Ridge)
        result4 = advanced_property_valuation(
            training_data, 
            target_property,
            feature_selection='auto',
            poly_degree=2,
            regularization='ridge',
            alpha=1.0
        )
        
        # Display results for each model
        display_model_results(result1, original_price, price_source, "Basic Model")
        display_model_results(result2, original_price, price_source, "Feature Selection Model")
        display_model_results(result3, original_price, price_source, "Polynomial Features Model")
        display_model_results(result4, original_price, price_source, "Regularized Model")
        
        # Comparison of all models
        logger.info("\n===== MODEL COMPARISON =====")
        compare_models([
            ("Basic", result1),
            ("Feature Selection", result2),
            ("Polynomial", result3),
            ("Regularized", result4)
        ], original_price)
        
        # Close database connection
        db.close()
        
    except Exception as e:
        logger.error(f"Error testing valuation model: {str(e)}", exc_info=True)


def display_model_results(result, original_price, price_source, model_name):
    """Display key results from a valuation model."""
    logger.info(f"\n===== {model_name.upper()} RESULTS =====")
    
    if 'error' in result:
        logger.error(f"Valuation error: {result['error']}")
        return
    
    predicted_value = result.get('predicted_value')
    r2_score = result.get('r2_score')
    r2_train = result.get('r2_train')
    
    if predicted_value is not None:
        logger.info(f"Predicted property value: ${predicted_value:,.2f}")
    
    if r2_score is not None:
        logger.info(f"Model R-squared (test) score: {r2_score:.4f}")
    
    if r2_train is not None:
        logger.info(f"Model R-squared (train) score: {r2_train:.4f}")
    
    # Advanced metrics if available
    if 'rmse' in result:
        logger.info(f"RMSE: ${result['rmse']:,.2f}")
        
    if 'mae' in result:
        logger.info(f"MAE: ${result['mae']:,.2f}")
    
    if 'cross_val_r2' in result and result['cross_val_r2'] is not None:
        logger.info(f"Cross-validation R²: {result['cross_val_r2']:.4f} (±{result.get('cross_val_std', 0):.4f})")
    
    if 'prediction_interval' in result and result['prediction_interval']:
        interval = result['prediction_interval']
        logger.info(f"Prediction interval: ${interval[0]:,.2f} to ${interval[1]:,.2f}")
    
    # Compare with original price if available
    if original_price and predicted_value:
        difference = predicted_value - original_price
        percent_diff = (difference / original_price) * 100
        logger.info(f"Difference from original {price_source}: ${difference:,.2f} ({percent_diff:+.2f}%)")
    
    # Show model configuration details
    if 'model_config' in result:
        config = result['model_config']
        logger.info("\nModel Configuration:")
        for key, value in config.items():
            if value is not None:
                logger.info(f"  - {key}: {value}")
    
    # Data metrics if available
    if 'data_metrics' in result:
        metrics = result['data_metrics']
        logger.info("\nData Metrics:")
        logger.info(f"  - Sample size: {metrics.get('sample_size', 0)}")
        logger.info(f"  - Training samples: {metrics.get('train_size', 0)}")
        logger.info(f"  - Test samples: {metrics.get('test_size', 0)}")
        if 'mean_price' in metrics:
            logger.info(f"  - Mean price: ${metrics['mean_price']:,.2f}")
        if 'median_price' in metrics:
            logger.info(f"  - Median price: ${metrics['median_price']:,.2f}")
    
    # Statistical significance if available
    if 'model_statistics' in result:
        stats = result['model_statistics']
        logger.info("\nStatistical Significance:")
        if 'f_statistic' in stats and 'f_pvalue' in stats:
            logger.info(f"  - F-statistic: {stats['f_statistic']:.4f} (p={stats['f_pvalue']:.6f})")
        if 'adj_r2' in stats:
            logger.info(f"  - Adjusted R²: {stats['adj_r2']:.4f}")
    
    # Show feature importance for top features
    if 'feature_importance' in result and result['feature_importance']:
        logger.info("\nTop Feature Importance:")
        # Show at most 10 top features
        for i, feature in enumerate(result['feature_importance']):
            if i >= 10:
                break
            significance = ""
            if 'p_value' in feature and feature['p_value'] is not None:
                if feature['p_value'] < 0.05:
                    significance = f"(p={feature['p_value']:.4f}) *"
                else:
                    significance = f"(p={feature['p_value']:.4f})"
            
            logger.info(f"  - {feature['feature']}: {feature['importance']:.4f} (coef: {feature.get('coefficient', 0):.4f}) {significance}")


def compare_models(model_results, original_price):
    """Compare multiple valuation models."""
    models = []
    
    for name, result in model_results:
        if 'error' in result:
            continue
        
        predicted = result.get('predicted_value')
        r2 = result.get('r2_score')
        rmse = result.get('rmse')
        
        # Calculate error if original price is available
        error = np.nan
        error_percent = np.nan
        if original_price and predicted:
            error = predicted - original_price
            error_percent = (error / original_price) * 100
        
        models.append({
            'name': name,
            'predicted': predicted,
            'r2': r2,
            'rmse': rmse,
            'error': error,
            'error_percent': error_percent
        })
    
    # If no valid models, exit
    if not models:
        logger.warning("No valid model results to compare")
        return
    
    # Create comparison table using tabulate
    headers = ["Model", "Predicted Value", "R²", "RMSE", "Error", "Error %"]
    table = []
    
    for model in models:
        row = [
            model['name'],
            f"${model['predicted']:,.2f}" if model['predicted'] else 'N/A',
            f"{model['r2']:.4f}" if model['r2'] is not None else 'N/A',
            f"${model['rmse']:,.2f}" if model['rmse'] is not None else 'N/A',
            f"${model['error']:,.2f}" if not np.isnan(model['error']) else 'N/A',
            f"{model['error_percent']:+.2f}%" if not np.isnan(model['error_percent']) else 'N/A'
        ]
        table.append(row)
    
    # Print table manually since we might not have tabulate
    col_widths = [max(len(str(row[i])) for row in table + [headers]) for i in range(len(headers))]
    
    # Print header
    header_row = " | ".join(f"{headers[i]:{col_widths[i]}}" for i in range(len(headers)))
    logger.info(header_row)
    logger.info("-" * len(header_row))
    
    # Print rows
    for row in table:
        logger.info(" | ".join(f"{row[i]:{col_widths[i]}}" for i in range(len(row))))
    
    # Find best model based on R² or error
    best_r2_model = max(models, key=lambda m: m['r2'] if m['r2'] is not None else -float('inf'))
    best_rmse_model = min(models, key=lambda m: m['rmse'] if m['rmse'] is not None else float('inf'))
    
    if original_price:
        best_error_model = min(models, key=lambda m: abs(m['error']) if not np.isnan(m['error']) else float('inf'))
        logger.info(f"\nBest model by error: {best_error_model['name']} (${abs(best_error_model['error']):,.2f} off)")
    
    logger.info(f"Best model by R²: {best_r2_model['name']} ({best_r2_model['r2']:.4f})")
    logger.info(f"Best model by RMSE: {best_rmse_model['name']} (${best_rmse_model['rmse']:,.2f})")


if __name__ == "__main__":
    main()