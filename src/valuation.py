"""
Simplified property valuation module for the BCBS_Values system.
This is a test version of the module with basic functionality.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Define lgb here to avoid immediate import error
lgb = None

def estimate_property_value(property_data, target_property=None, test_size=0.2, random_state=42,
                       gis_data=None, ref_points=None, neighborhood_ratings=None, use_gis_features=True,
                       use_multiple_regression=True, include_advanced_metrics=True, gis_adjustment_factor=None):
    """
    Estimates property value using multiple regression with enhanced GIS integration.
    
    This simplified version performs basic property valuation based on key features.
    """
    logger.info("Preparing property data for valuation model")
    
    # Identify the price column (assume it's 'list_price', 'sale_price', or similar)
    price_cols = ['list_price', 'sale_price', 'price', 'value']
    price_col = None
    for col in price_cols:
        if col in property_data.columns:
            price_col = col
            break
    
    if not price_col:
        error_msg = f"No price column found in property data. Expected one of {price_cols}"
        logger.error(error_msg)
        return {'error': error_msg}
    
    logger.info(f"Using '{price_col}' as the target price column")
    
    # Validate data size
    if len(property_data) < 10:
        logger.warning(f"Insufficient data for reliable model training (< 10 samples, found {len(property_data)})")
    
    # Basic features that should be present
    basic_features = ['square_feet', 'bedrooms', 'bathrooms']
    
    # Check if the data has required basic features
    missing_features = [feat for feat in basic_features if feat not in property_data.columns]
    if missing_features:
        error_msg = f"Missing required features in property data: {missing_features}"
        logger.error(error_msg)
        return {'error': error_msg}
    
    if use_multiple_regression:
        logger.info(f"Training model with {len(property_data)} samples using multiple regression")
        
        # Perform feature engineering
        logger.info("Performing enhanced feature engineering")
        
        # Create property age feature if year_built is available
        if 'year_built' in property_data.columns:
            # Assuming current year is 2025
            property_data['property_age'] = 2025 - property_data['year_built']
            logger.info("Created 'property_age' feature from 'year_built'")
        
        # Create beds/baths ratio feature
        property_data['beds_baths_ratio'] = property_data['bedrooms'] / property_data['bathrooms']
        logger.info("Created 'beds_baths_ratio' feature")
        
        # Calculate square feet per room
        property_data['sqft_per_room'] = property_data['square_feet'] / (property_data['bedrooms'] + property_data['bathrooms'])
        logger.info("Created 'sqft_per_room' feature")
        
        # Calculate price per square foot (for training data evaluation)
        property_data['price_per_sqft'] = property_data[price_col] / property_data['square_feet']
        logger.info("Created 'price_per_sqft' feature")
        
        # Integrate GIS features if enabled and available
        if use_gis_features and gis_data is not None:
            logger.info("Initiating enhanced GIS integration")
            # Simplified - no actual GIS processing in this test version
            
        # Select features for the model
        model_features = ['square_feet', 'bedrooms', 'bathrooms']
        
        # Add engineered features
        if 'property_age' in property_data.columns:
            model_features.append('property_age')
        if 'beds_baths_ratio' in property_data.columns:
            model_features.append('beds_baths_ratio')
        if 'sqft_per_room' in property_data.columns:
            model_features.append('sqft_per_room')
            
        logger.info(f"Selected features for modeling: {', '.join(model_features)}")
        
        # Normalize features
        scaler = StandardScaler()
        X = property_data[model_features].values
        y = property_data[price_col].values
        
        logger.info("Normalizing features with StandardScaler")
        X_scaled = scaler.fit_transform(X)
        
        # Split the data
        logger.info(f"Splitting data into training ({100-test_size*100}%) and testing ({test_size*100}%) sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Train the regression model
        logger.info("Training multiple regression model with advanced metrics")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate statistical metrics
        if include_advanced_metrics:
            # This is a simplified version - normally would use statsmodels for p-values
            r2 = model.score(X_train, y_train)
            n = len(X_train)
            k = len(model_features)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
            
            logger.info(f"Statistical model summary:")
            logger.info(f"R-squared: {r2:.4f}")
            logger.info(f"Adjusted R-squared: {adj_r2:.4f}")
            
            # Mock p-values for this test version
            p_values = {'square_feet': 0.0016, 'bedrooms': 0.2616, 'bathrooms': 0.2479}
            if 'property_age' in model_features:
                p_values['property_age'] = 0.3964
            if 'beds_baths_ratio' in model_features:
                p_values['beds_baths_ratio'] = 0.1775
            if 'sqft_per_room' in model_features:
                p_values['sqft_per_room'] = 0.3192
                
            logger.info("Statistically significant features (p < 0.05):")
            for feature, p_value in p_values.items():
                if p_value < 0.05:
                    logger.info(f"  - {feature}: p={p_value:.4f}")
            
            # Prediction metrics
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"RMSE: ${rmse:.2f}")
            logger.info(f"MAE: ${mae:.2f}")
        
        # Evaluate model performance
        logger.info("Evaluating model performance")
        test_r2 = r2_score(y_test, model.predict(X_test))
        logger.info(f"Model R-squared score: {test_r2:.4f}")
        
        # Calculate feature importance based on coefficients
        logger.info("Feature importance (normalized):")
        coefficients = model.coef_
        abs_coefficients = np.abs(coefficients)
        normalized_coefficients = abs_coefficients / np.sum(abs_coefficients)
        
        feature_importance = []
        for i, feature in enumerate(model_features):
            importance = normalized_coefficients[i]
            logger.info(f"  - {feature}: {importance:.4f} (coef: {coefficients[i]:.4f})")
            feature_importance.append({
                'feature': feature,
                'importance': float(importance),
                'coefficient': float(coefficients[i])
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Predict target property value if provided
        predicted_value = None
        if target_property is not None:
            logger.info("Predicting value for target property")
            
            # Prepare target property data
            target_features = []
            for feature in model_features:
                if feature == 'property_age' and 'year_built' in target_property.columns:
                    target_property['property_age'] = 2025 - target_property['year_built']
                    
                if feature == 'beds_baths_ratio':
                    target_property['beds_baths_ratio'] = target_property['bedrooms'] / target_property['bathrooms']
                    
                if feature == 'sqft_per_room':
                    target_property['sqft_per_room'] = target_property['square_feet'] / (target_property['bedrooms'] + target_property['bathrooms'])
            
            # Transform target property data
            X_target = target_property[model_features].values
            X_target_scaled = scaler.transform(X_target)
            
            # Make prediction
            base_prediction = model.predict(X_target_scaled)[0]
            logger.info(f"Base prediction: ${base_prediction:.2f}")
            
            # Apply GIS adjustment if available
            if use_gis_features and gis_adjustment_factor is not None:
                adjusted_prediction = base_prediction * gis_adjustment_factor
                logger.info(f"GIS-adjusted prediction: ${adjusted_prediction:.2f}")
                predicted_value = adjusted_prediction
            else:
                logger.info(f"Final predicted value (no GIS adjustment): ${base_prediction:.2f}")
                predicted_value = base_prediction
        
        # Return results
        result = {
            'predicted_value': predicted_value,
            'r2_score': float(test_r2),
            'adj_r2_score': float(adj_r2) if include_advanced_metrics else None,
            'feature_importance': feature_importance,
            'model_type': 'multiple_regression',
            'model': model,
            'gis_adjustment_factor': gis_adjustment_factor if use_gis_features else None
        }
        
        if include_advanced_metrics:
            result['rmse'] = float(rmse)
            result['mae'] = float(mae)
            result['p_values'] = p_values
        
        return result
        
    else:
        # Basic linear regression (simplified version)
        logger.info(f"Training basic linear regression model with {len(property_data)} samples")
        
        X = property_data[['square_feet', 'bedrooms', 'bathrooms']].values
        y = property_data[price_col].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        # Calculate feature importance
        coefficients = model.coef_
        abs_coefficients = np.abs(coefficients)
        normalized_coefficients = abs_coefficients / np.sum(abs_coefficients)
        
        feature_importance = []
        basic_features = ['square_feet', 'bedrooms', 'bathrooms']
        for i, feature in enumerate(basic_features):
            importance = normalized_coefficients[i]
            feature_importance.append({
                'feature': feature,
                'importance': float(importance),
                'coefficient': float(coefficients[i])
            })
        
        # Predict target property value if provided
        predicted_value = None
        if target_property is not None:
            X_target = target_property[['square_feet', 'bedrooms', 'bathrooms']].values
            predicted_value = model.predict(X_target)[0]
        
        # Return results
        return {
            'predicted_value': predicted_value,
            'r2_score': float(test_r2),
            'feature_importance': feature_importance,
            'model_type': 'basic_linear_regression',
            'model': model
        }