"""
Property valuation module for the BCBS_Values system.
Implements valuation models for estimating property values based on features.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)

def estimate_property_value(property_data, target_property=None, test_size=0.2, random_state=42):
    """
    Estimates property value using linear regression based on property features.
    
    This function trains a simple linear regression model on property data
    to predict property values based on key features like square footage,
    number of bedrooms, bathrooms, and property age.
    
    Args:
        property_data (pd.DataFrame): DataFrame containing property data with features and prices
        target_property (pd.DataFrame, optional): Single property to predict value for.
            If None, returns model performance metrics only.
        test_size (float, optional): Proportion of data to use for testing (default: 0.2)
        random_state (int, optional): Random seed for reproducibility (default: 42)
    
    Returns:
        dict: Dictionary containing predicted value (if target_property provided),
              model performance metrics, and feature importance
    
    Example:
        >>> df = pd.DataFrame({
        ...     'square_feet': [1500, 2000, 1800, 2200],
        ...     'bedrooms': [3, 4, 3, 4],
        ...     'bathrooms': [2, 2.5, 2, 3],
        ...     'year_built': [1990, 2005, 2000, 2010],
        ...     'list_price': [300000, 400000, 350000, 450000]
        ... })
        >>> target = pd.DataFrame({
        ...     'square_feet': [1750],
        ...     'bedrooms': [3],
        ...     'bathrooms': [2],
        ...     'year_built': [1995]
        ... })
        >>> estimate_property_value(df, target)
    """
    try:
        # Make a copy to avoid modifying original data
        df = property_data.copy()
        
        # Step 1: Data preparation and cleaning
        logger.info("Preparing property data for valuation model")
        
        # Drop rows with missing values in key columns
        key_columns = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        if 'list_price' in df.columns:
            target_column = 'list_price'
        elif 'estimated_value' in df.columns:
            target_column = 'estimated_value'
        elif 'last_sale_price' in df.columns:
            target_column = 'last_sale_price'
        else:
            raise ValueError("No price column found in data (list_price, estimated_value, or last_sale_price required)")
        
        key_columns.append(target_column)
        df = df.dropna(subset=key_columns)
        
        if len(df) < 5:  # Reduced minimum sample size for testing purposes
            logger.warning(f"Insufficient data for reliable model training (< 5 samples, found {len(df)})")
            return {
                'error': 'Insufficient data for model training',
                'predicted_value': None,
                'r2_score': None,
                'feature_importance': None
            }
            
        logger.info(f"Training model with {len(df)} samples")
        
        # Step 2: Feature engineering
        logger.info("Performing feature engineering")
        
        # Calculate property age from year_built
        current_year = pd.Timestamp.now().year
        df['property_age'] = current_year - df['year_built']
        
        # Create derived features
        df['beds_baths_ratio'] = df['bedrooms'] / df['bathrooms'].clip(lower=0.5)
        df['sqft_per_room'] = df['square_feet'] / (df['bedrooms'] + df['bathrooms']).clip(lower=1.0)
        
        # Select features for model
        features = ['square_feet', 'bedrooms', 'bathrooms', 'property_age', 
                   'beds_baths_ratio', 'sqft_per_room']
        
        # Prepare feature matrix X and target vector y
        X = df[features]
        y = df[target_column]
        
        # Step 3: Data normalization with StandardScaler
        logger.info("Normalizing features")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Step 4: Split data into training and testing sets
        logger.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Step 5: Train the linear regression model
        logger.info("Training linear regression model")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Step 6: Evaluate model performance
        logger.info("Evaluating model performance")
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test, y_pred)
        logger.info(f"Model R-squared score: {model_r2:.4f}")
        
        # Step 7: Calculate feature importance (coefficients normalized by feature scale)
        # Coefficients in a linear model represent how much y changes for a unit change in each feature
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': np.abs(model.coef_),
            'coefficient': model.coef_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance:")
        for _, row in feature_importance.iterrows():
            logger.info(f"  - {row['feature']}: {row['importance']:.4f} (coef: {row['coefficient']:.4f})")
        
        # Step 8: Predict value for target property if provided
        predicted_value = None
        if target_property is not None:
            logger.info("Predicting value for target property")
            
            # Prepare target property features
            target_df = target_property.copy()
            
            # Apply same feature engineering to target property
            target_df['property_age'] = current_year - target_df['year_built']
            target_df['beds_baths_ratio'] = target_df['bedrooms'] / target_df['bathrooms'].clip(lower=0.5)
            target_df['sqft_per_room'] = target_df['square_feet'] / (target_df['bedrooms'] + target_df['bathrooms']).clip(lower=1.0)
            
            # Extract features and scale them
            target_features = target_df[features]
            target_scaled = scaler.transform(target_features)
            
            # Make prediction
            predicted_value = float(model.predict(target_scaled)[0])
            logger.info(f"Predicted property value: ${predicted_value:,.2f}")
        
        # Return results
        return {
            'predicted_value': predicted_value,
            'r2_score': model_r2,
            'feature_importance': feature_importance.to_dict('records'),
            'model_params': {
                'intercept': float(model.intercept_),
                'coefficients': dict(zip(features, model.coef_))
            },
            'data_metrics': {
                'sample_size': len(df),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'mean_price': float(df[target_column].mean()),
                'median_price': float(df[target_column].median()),
                'price_range': [float(df[target_column].min()), float(df[target_column].max())]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in property valuation: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'predicted_value': None,
            'r2_score': None,
            'feature_importance': None
        }


def advanced_property_valuation(property_data, target_property=None, location_weight=0.3, include_location=True):
    """
    Advanced property valuation that incorporates location data and weighted features.
    This is a placeholder for a more sophisticated valuation model.
    
    Args:
        property_data (pd.DataFrame): DataFrame containing property data
        target_property (pd.DataFrame, optional): Single property to predict value for
        location_weight (float, optional): Weight to give to location factors (0-1)
        include_location (bool, optional): Whether to include location in the model
        
    Returns:
        dict: Dictionary with valuation results
    """
    # This is a stub for future implementation
    logger.info("Advanced property valuation not yet implemented")
    
    # For now, call the basic model
    return estimate_property_value(property_data, target_property)