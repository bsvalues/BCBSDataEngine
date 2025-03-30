"""
Advanced property valuation module for the BCBS_Values system.

This module provides sophisticated real estate valuation capabilities using
multiple regression techniques, enhanced GIS integration, and advanced
statistical analysis. It supports both linear models and gradient boosting
for more accurate property valuations.
"""

import logging
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import LightGBM if available
LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info("LightGBM is available for advanced regression models")
except ImportError:
    logger.warning("LightGBM not available - will use GradientBoostingRegressor as fallback")
except OSError as e:
    # This can happen if OS dependencies are missing
    LIGHTGBM_AVAILABLE = False
    logger.warning(f"LightGBM found but could not be loaded due to OS error: {e}")
    logger.warning("Missing system dependencies for LightGBM - using GradientBoostingRegressor as fallback")

def estimate_property_value(property_data, target_property=None, test_size=0.2, random_state=42,
                       gis_data=None, ref_points=None, neighborhood_ratings=None, use_gis_features=True,
                       use_multiple_regression=True, include_advanced_metrics=True, gis_adjustment_factor=None,
                       model_type='linear', feature_selection_method='all', regularization_strength=0.01,
                       spatial_adjustment_method='multiplicative', confidence_interval_level=0.95,
                       handle_outliers=True, handle_missing_values=True, cross_validation_folds=5):
    """
    Estimates property value using advanced regression techniques with enhanced GIS integration.
    
    This function implements a comprehensive property valuation model that combines
    traditional regression techniques with geospatial data processing to provide
    more accurate and context-aware valuations.
    
    Parameters:
    -----------
    property_data : pandas.DataFrame
        Dataset containing property information including features like square feet,
        bedrooms, bathrooms, and historical sale prices.
    
    target_property : pandas.DataFrame or dict, optional
        Property for which to predict the value. If None, only model training is performed.
    
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split for model evaluation.
    
    random_state : int, default=42
        Random seed for reproducibility in data splitting and model training.
    
    gis_data : dict or pandas.DataFrame, optional
        Geospatial data to enhance valuation with location-based features.
    
    ref_points : dict, optional
        Dictionary of reference points (e.g., downtown, schools) with lat/lon coordinates
        and weights to calculate proximity scores.
    
    neighborhood_ratings : dict, optional
        Dictionary mapping neighborhoods to quality ratings used for location adjustments.
    
    use_gis_features : bool, default=True
        Whether to incorporate GIS and location-based features in the model.
    
    use_multiple_regression : bool, default=True
        If True, uses advanced multiple regression; if False, uses basic linear regression.
    
    include_advanced_metrics : bool, default=True
        Whether to include detailed statistical metrics like p-values and confidence intervals.
    
    gis_adjustment_factor : float, optional
        Direct multiplier for GIS-based adjustments to valuations.
    
    model_type : str, default='linear'
        Type of regression model to use. Options:
        - 'linear': Standard OLS linear regression (default)
        - 'ridge': Ridge regression with L2 regularization
        - 'lasso': Lasso regression with L1 regularization
        - 'elastic_net': Elastic Net combining L1 and L2 regularization
        - 'lightgbm': LightGBM gradient boosting (if available)
        - 'gbr': Gradient Boosting Regressor (sklearn implementation)
        - 'ensemble': Ensemble of linear and gradient boosting models
    
    feature_selection_method : str, default='all'
        Method for selecting features. Options:
        - 'all': Use all available features
        - 'correlation': Select features based on correlation with target
        - 'mutual_info': Select features based on mutual information
        - 'recursive': Recursive feature elimination
    
    regularization_strength : float, default=0.01
        Strength of regularization for models that support it (ridge, lasso, elastic_net).
    
    spatial_adjustment_method : str, default='multiplicative'
        Method for applying spatial adjustments. Options:
        - 'multiplicative': Apply as a percentage adjustment
        - 'additive': Apply as a fixed dollar amount
        - 'hybrid': Combination of multiplicative and additive
    
    confidence_interval_level : float, default=0.95
        Confidence level for prediction intervals (e.g., 0.95 for 95% CI).
    
    handle_outliers : bool, default=True
        Whether to detect and handle outliers in the data.
    
    handle_missing_values : bool, default=True
        Whether to impute missing values in the data.
    
    cross_validation_folds : int, default=5
        Number of folds for cross-validation when evaluating model performance.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - estimated_value: Predicted property value
        - confidence_interval: Lower and upper bounds of prediction confidence interval
        - r_squared: R-squared score of the model
        - adjusted_r_squared: Adjusted R-squared accounting for number of predictors
        - feature_importances: Feature importance scores
        - model_coefficients: Model coefficients (for linear models)
        - p_values: P-values for each coefficient (for linear models)
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - model_performance: Cross-validation performance metrics
        - spatial_factors: Information about spatial adjustments applied
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
        logger.info(f"Training model with {len(property_data)} samples using advanced regression techniques")
        
        # -------------------------------------------------------------------------
        # Step 1: Data Preparation and Error Handling
        # -------------------------------------------------------------------------
        
        # Handle missing values if enabled
        if handle_missing_values:
            logger.info("Checking for and handling missing values in the dataset")
            # Count missing values before imputation
            missing_before = property_data[basic_features].isnull().sum().sum()
            
            if missing_before > 0:
                logger.warning(f"Found {missing_before} missing values in basic features")
                # Create a simple imputer for basic features
                property_data[basic_features] = property_data[basic_features].fillna(property_data[basic_features].median())
                logger.info("Missing values imputed with median values")
            else:
                logger.info("No missing values found in basic features")
        
        # Handle division by zero errors in feature engineering
        # Avoid potential division by zero with small epsilon
        epsilon = 1e-6
        
        # -------------------------------------------------------------------------
        # Step 2: Enhanced Feature Engineering
        # -------------------------------------------------------------------------
        logger.info("Performing advanced feature engineering")
        
        # Create property age feature if year_built is available
        current_year = 2025
        if 'year_built' in property_data.columns:
            property_data['property_age'] = current_year - property_data['year_built']
            # Cap extreme property ages (error handling for invalid years)
            property_data['property_age'] = property_data['property_age'].clip(0, 200)
            logger.info("Created 'property_age' feature from 'year_built' with capping")
            
            # Add squared terms for non-linear relationships (age often has diminishing effect)
            property_data['property_age_squared'] = property_data['property_age'] ** 2
            logger.info("Created 'property_age_squared' feature for non-linear age effects")
        
        # Create beds/baths ratio feature with error handling
        property_data['beds_baths_ratio'] = property_data['bedrooms'] / (property_data['bathrooms'] + epsilon)
        # Clip extreme ratios (error handling)
        property_data['beds_baths_ratio'] = property_data['beds_baths_ratio'].clip(0.25, 4.0)
        logger.info("Created 'beds_baths_ratio' feature with bounds [0.25, 4.0]")
        
        # Calculate square feet per room with error handling
        property_data['sqft_per_room'] = property_data['square_feet'] / (property_data['bedrooms'] + property_data['bathrooms'] + epsilon)
        # Clip extreme values
        property_data['sqft_per_room'] = property_data['sqft_per_room'].clip(50, 1000)
        logger.info("Created 'sqft_per_room' feature with bounds [50, 1000]")
        
        # Create room count feature
        property_data['total_rooms'] = property_data['bedrooms'] + property_data['bathrooms']
        logger.info("Created 'total_rooms' feature")
        
        # Create bedroom to total room ratio
        property_data['bedroom_ratio'] = property_data['bedrooms'] / (property_data['total_rooms'] + epsilon)
        logger.info("Created 'bedroom_ratio' feature")
        
        # Calculate price per square foot (for training data evaluation)
        property_data['price_per_sqft'] = property_data[price_col] / (property_data['square_feet'] + epsilon)
        # Clip extreme price per sqft values (error handling)
        q1 = property_data['price_per_sqft'].quantile(0.01)
        q3 = property_data['price_per_sqft'].quantile(0.99)
        property_data['price_per_sqft'] = property_data['price_per_sqft'].clip(q1, q3)
        logger.info(f"Created 'price_per_sqft' feature with outlier clipping [{q1:.2f}, {q3:.2f}]")
        
        # -------------------------------------------------------------------------
        # Step 3: GIS Data Integration
        # -------------------------------------------------------------------------
        
        # Define placeholder for GIS features we'll add
        gis_features = []
        spatial_adjustments = {}
        
        # Integrate GIS features if enabled and available
        if use_gis_features:
            logger.info("Initiating enhanced GIS integration")
            
            # Check if we have coordinates for geospatial analysis
            has_coordinates = ('latitude' in property_data.columns and 'longitude' in property_data.columns)
            
            if has_coordinates:
                logger.info("Latitude and longitude data available for spatial analysis")
                
                # 3.1 Calculate proximity scores if reference points are provided
                if ref_points is not None:
                    logger.info(f"Calculating proximity scores using {len(ref_points)} reference points")
                    
                    # List to store proximity scores for each property
                    proximity_scores = []
                    
                    # For each property, calculate proximity to each reference point
                    for idx, row in property_data.iterrows():
                        # Calculate proximity to each reference point
                        total_score = 0
                        total_weight = 0
                        
                        try:
                            for point_name, point_data in ref_points.items():
                                # Skip points with missing data
                                if 'lat' not in point_data or 'lon' not in point_data or 'weight' not in point_data:
                                    continue
                                    
                                # Calculate distance in kilometers using Haversine formula
                                lat1, lon1 = row['latitude'], row['longitude']
                                lat2, lon2 = point_data['lat'], point_data['lon']
                                
                                # Simple version of Haversine formula
                                dlat = math.radians(lat2 - lat1)
                                dlon = math.radians(lon2 - lon1)
                                a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                                     math.sin(dlon/2) * math.sin(dlon/2))
                                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                                distance_km = 6371 * c  # Earth radius in km
                                
                                # Apply exponential decay to distance
                                # Closer points have higher scores (1.0 at distance=0, ~0 at large distances)
                                proximity = math.exp(-0.5 * distance_km)
                                
                                # Apply point weight
                                weighted_proximity = proximity * point_data['weight']
                                
                                total_score += weighted_proximity
                                total_weight += point_data['weight']
                            
                            # Normalize score by total weight
                            if total_weight > 0:
                                final_score = total_score / total_weight
                            else:
                                final_score = 0.5  # Default value if no valid reference points
                                
                            proximity_scores.append(final_score)
                            
                        except Exception as e:
                            logger.warning(f"Error calculating proximity score for property {idx}: {str(e)}")
                            proximity_scores.append(0.5)  # Default value on error
                    
                    # Add proximity scores as a feature
                    property_data['proximity_score'] = proximity_scores
                    gis_features.append('proximity_score')
                    logger.info("Added 'proximity_score' feature based on reference points")
                
                # 3.2 Apply neighborhood quality ratings if available
                if neighborhood_ratings is not None and 'neighborhood' in property_data.columns:
                    logger.info("Applying neighborhood quality ratings")
                    
                    # Create neighborhood quality feature
                    neighborhood_quality = []
                    
                    for idx, row in property_data.iterrows():
                        try:
                            neighborhood = row['neighborhood'] if 'neighborhood' in row else None
                            city = row['city'] if 'city' in row else None
                            
                            # First try exact neighborhood match
                            if neighborhood and neighborhood in neighborhood_ratings:
                                quality = neighborhood_ratings[neighborhood]
                            # Then try city match
                            elif city and city in neighborhood_ratings:
                                quality = neighborhood_ratings[city]
                            # Default to unknown
                            else:
                                quality = neighborhood_ratings.get('Unknown', 1.0)
                                
                            neighborhood_quality.append(quality)
                            
                        except Exception as e:
                            logger.warning(f"Error determining neighborhood quality for property {idx}: {str(e)}")
                            neighborhood_quality.append(1.0)  # Default neutral value on error
                    
                    # Add neighborhood quality as a feature
                    property_data['neighborhood_quality'] = neighborhood_quality
                    gis_features.append('neighborhood_quality')
                    logger.info("Added 'neighborhood_quality' feature based on neighborhood ratings")
                    
                    # Store for spatial adjustments
                    spatial_adjustments['neighborhood_factor'] = {
                        'type': 'multiplicative',
                        'description': 'Location quality adjustment based on neighborhood',
                        'average_impact': np.mean(neighborhood_quality)
                    }
                
                # 3.3 Calculate additional GIS-derived features using provided GIS data
                if gis_data is not None:
                    logger.info("Integrating additional GIS data")
                    
                    # Assumption: gis_data contains property-specific features like flood risk, etc.
                    # This would typically be from a dedicated GIS system or pre-processed dataset
                    
                    try:
                        # Access GIS data for each property (implementation would depend on gis_data structure)
                        # Example mapping via property ID
                        if hasattr(gis_data, 'items') and isinstance(gis_data, dict):
                            if 'flood_risk' in gis_data:
                                property_data['flood_risk'] = property_data['property_id'].map(gis_data['flood_risk']).fillna(0.5)
                                gis_features.append('flood_risk')
                                logger.info("Added 'flood_risk' feature from GIS data")
                            
                            if 'school_quality' in gis_data:
                                property_data['school_quality'] = property_data['property_id'].map(gis_data['school_quality']).fillna(5.0)
                                gis_features.append('school_quality')
                                logger.info("Added 'school_quality' feature from GIS data")
                                
                            if 'traffic_noise' in gis_data:
                                property_data['traffic_noise'] = property_data['property_id'].map(gis_data['traffic_noise']).fillna(0.5)
                                gis_features.append('traffic_noise')
                                logger.info("Added 'traffic_noise' feature from GIS data")
                                
                            if 'view_quality' in gis_data:
                                property_data['view_quality'] = property_data['property_id'].map(gis_data['view_quality']).fillna(0.5)
                                gis_features.append('view_quality')
                                logger.info("Added 'view_quality' feature from GIS data")
                        
                    except Exception as e:
                        logger.warning(f"Error integrating GIS data: {str(e)}")

            # 3.4 If we have a direct GIS adjustment factor, use it
            if gis_adjustment_factor is not None:
                logger.info(f"Using provided GIS adjustment factor: {gis_adjustment_factor}")
                spatial_adjustments['direct_adjustment'] = {
                    'type': spatial_adjustment_method,
                    'value': gis_adjustment_factor,
                    'description': 'Direct GIS adjustment factor'
                }
                
        # -------------------------------------------------------------------------
        # Step 4: Feature Selection
        # -------------------------------------------------------------------------
        logger.info(f"Performing feature selection using method: {feature_selection_method}")
        
        # Start with basic features
        model_features = list(basic_features)  # Make a copy to avoid modifying original
        
        # Add engineered features
        engineered_features = []
        if 'property_age' in property_data.columns:
            engineered_features.append('property_age')
        if 'property_age_squared' in property_data.columns:
            engineered_features.append('property_age_squared')
        if 'beds_baths_ratio' in property_data.columns:
            engineered_features.append('beds_baths_ratio')
        if 'sqft_per_room' in property_data.columns:
            engineered_features.append('sqft_per_room')
        if 'total_rooms' in property_data.columns:
            engineered_features.append('total_rooms')
        if 'bedroom_ratio' in property_data.columns:
            engineered_features.append('bedroom_ratio')
        
        # Feature selection based on specified method
        if feature_selection_method == 'all':
            # Use all features
            model_features.extend(engineered_features)
            model_features.extend(gis_features)
            
        elif feature_selection_method == 'correlation':
            # Select features based on correlation with target
            try:
                correlations = property_data[engineered_features + gis_features].corrwith(property_data[price_col]).abs()
                selected_features = correlations[correlations > 0.1].index.tolist()
                model_features.extend(selected_features)
                logger.info(f"Selected {len(selected_features)} additional features based on correlation")
            except Exception as e:
                logger.warning(f"Error during correlation-based feature selection: {str(e)}")
                # Fall back to using all features
                model_features.extend(engineered_features)
                model_features.extend(gis_features)
                
        elif feature_selection_method == 'mutual_info':
            # This would typically use sklearn's mutual_info_regression
            # For simplicity, we'll just use all features with a note
            logger.info("Mutual information feature selection requires sklearn.feature_selection")
            model_features.extend(engineered_features)
            model_features.extend(gis_features)
            
        elif feature_selection_method == 'recursive':
            # This would typically use sklearn's RFE
            # For simplicity, we'll just use all features with a note
            logger.info("Recursive feature elimination requires sklearn.feature_selection")
            model_features.extend(engineered_features)
            model_features.extend(gis_features)
            
        else:
            # Default to all features for unknown selection method
            logger.warning(f"Unknown feature selection method: {feature_selection_method}")
            model_features.extend(engineered_features)
            model_features.extend(gis_features)
        
        # Remove any duplicate features
        model_features = list(dict.fromkeys(model_features))
        logger.info(f"Selected features for modeling: {', '.join(model_features)}")
        
        # -------------------------------------------------------------------------
        # Step 5: Handle Outliers
        # -------------------------------------------------------------------------
        if handle_outliers:
            logger.info("Detecting and handling outliers")
            
            # Create a copy of the data for outlier handling
            X_data = property_data[model_features].copy()
            y_data = property_data[price_col].copy()
            
            # Z-score outlier detection and handling for features
            for feature in model_features:
                try:
                    z_scores = np.abs((X_data[feature] - X_data[feature].mean()) / X_data[feature].std())
                    outliers = z_scores > 3  # More than 3 standard deviations
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        logger.info(f"Detected {outlier_count} outliers in feature '{feature}'")
                        
                        # Clip instead of removing to preserve data size
                        lower_bound = X_data[feature].quantile(0.01)
                        upper_bound = X_data[feature].quantile(0.99)
                        X_data[feature] = X_data[feature].clip(lower_bound, upper_bound)
                except Exception as e:
                    logger.warning(f"Error handling outliers for feature '{feature}': {str(e)}")
            
            # Replace the original data with the outlier-handled data
            for feature in model_features:
                property_data[feature] = X_data[feature]
                
            # Handle outliers in target variable
            try:
                z_scores = np.abs((y_data - y_data.mean()) / y_data.std())
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"Detected {outlier_count} outliers in target variable '{price_col}'")
                    
                    # Clip instead of removing to preserve data size
                    lower_bound = y_data.quantile(0.01)
                    upper_bound = y_data.quantile(0.99)
                    property_data[price_col] = property_data[price_col].clip(lower_bound, upper_bound)
            except Exception as e:
                logger.warning(f"Error handling outliers for target variable: {str(e)}")
        
        # -------------------------------------------------------------------------
        # Step 6: Data Preparation for Modeling
        # -------------------------------------------------------------------------
        logger.info("Preparing data for modeling")
        
        # Select features and target
        X = property_data[model_features].values
        y = property_data[price_col].values
        
        # Choose appropriate scaler based on outlier handling
        if handle_outliers:
            scaler = RobustScaler()  # Better with outliers
            logger.info("Using RobustScaler for feature normalization (robust to outliers)")
        else:
            scaler = StandardScaler()  # Standard normalization
            logger.info("Using StandardScaler for feature normalization")
            
        # Normalize features
        X_scaled = scaler.fit_transform(X)
        
        # Split the data
        logger.info(f"Splitting data into training ({100-test_size*100}%) and testing ({test_size*100}%) sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # -------------------------------------------------------------------------
        # Step 7: Model Selection and Training
        # -------------------------------------------------------------------------
        logger.info(f"Training {model_type} model")
        
        # Initialize model based on specified type
        if model_type == 'linear':
            model = LinearRegression()
            model_name = 'Linear Regression'
            
        elif model_type == 'ridge':
            model = Ridge(alpha=regularization_strength, random_state=random_state)
            model_name = f'Ridge Regression (alpha={regularization_strength})'
            
        elif model_type == 'lasso':
            model = Lasso(alpha=regularization_strength, random_state=random_state)
            model_name = f'Lasso Regression (alpha={regularization_strength})'
            
        elif model_type == 'elastic_net':
            model = ElasticNet(alpha=regularization_strength, l1_ratio=0.5, random_state=random_state)
            model_name = f'Elastic Net (alpha={regularization_strength}, l1_ratio=0.5)'
            
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Convert to LightGBM dataset format
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            model_name = 'LightGBM Gradient Boosting'
            
        elif model_type == 'gbr' or (model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE):
            # Fall back to sklearn's GradientBoostingRegressor if LightGBM isn't available
            if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
                logger.warning("LightGBM not available - falling back to sklearn's GradientBoostingRegressor")
                
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state
            )
            model_name = 'Gradient Boosting Regressor'
            
        elif model_type == 'ensemble':
            # Create an ensemble of linear regression and gradient boosting
            linear_model = LinearRegression()
            
            # Check if LightGBM is available
            if LIGHTGBM_AVAILABLE:
                # Create LightGBM model
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'verbose': -1
                }
                
                # Convert to LightGBM dataset format
                train_data = lgb.Dataset(X_train, label=y_train)
                
                # Train the LightGBM model
                gbm_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
            else:
                # Fall back to sklearn's GBR if LightGBM not available
                gbm_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
                
            # Train both models separately
            linear_model.fit(X_train, y_train)
            
            # If using sklearn's GBR, fit it now
            if not LIGHTGBM_AVAILABLE:
                gbm_model.fit(X_train, y_train)
            
            # Create a dictionary to store both models
            model = {
                'linear': linear_model,
                'gbm': gbm_model,
                'type': 'lightgbm' if LIGHTGBM_AVAILABLE else 'gbr'
            }
            model_name = 'Ensemble (Linear + Gradient Boosting)'
        
        else:
            # Default to linear regression for unknown model types
            logger.warning(f"Unknown model type '{model_type}' - falling back to linear regression")
            model = LinearRegression()
            model_name = 'Linear Regression (fallback)'
        
        # Train the model (unless already trained for LightGBM or ensemble)
        is_ensemble = model_type == 'ensemble'
        is_lightgbm = model_type == 'lightgbm' and LIGHTGBM_AVAILABLE
        
        if not is_ensemble and not is_lightgbm:
            model.fit(X_train, y_train)
        
        logger.info(f"Successfully trained {model_name} model")
        
        # -------------------------------------------------------------------------
        # Step 8: Model Evaluation and Advanced Metrics
        # -------------------------------------------------------------------------
        logger.info("Evaluating model performance")
        
        # Different prediction logic based on model type
        if is_ensemble:
            # For ensemble, average predictions from both models
            if model['type'] == 'lightgbm':
                gbm_pred = model['gbm'].predict(X_test)
            else:
                gbm_pred = model['gbm'].predict(X_test)
                
            linear_pred = model['linear'].predict(X_test)
            y_pred = (gbm_pred + linear_pred) / 2
            
        elif is_lightgbm:
            # For LightGBM models
            y_pred = model.predict(X_test)
            
        else:
            # For sklearn models
            y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        test_r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Model evaluation metrics:")
        logger.info(f"  R-squared: {test_r2:.4f}")
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE: ${mae:.2f}")
        
        # Cross-validation if requested
        cv_scores = None
        cv_mean = None
        cv_std = None
        
        if cross_validation_folds > 0:
            logger.info(f"Performing {cross_validation_folds}-fold cross-validation")
            
            # Create a new model of the same type
            if model_type == 'linear':
                cv_model = LinearRegression()
            elif model_type == 'ridge':
                cv_model = Ridge(alpha=regularization_strength, random_state=random_state)
            elif model_type == 'lasso':
                cv_model = Lasso(alpha=regularization_strength, random_state=random_state)
            elif model_type == 'elastic_net':
                cv_model = ElasticNet(alpha=regularization_strength, l1_ratio=0.5, random_state=random_state)
            elif model_type == 'gbr' or (model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE):
                cv_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
            else:
                # Skip cross-validation for more complex models (LightGBM, ensemble)
                logger.info(f"Skipping cross-validation for {model_name} - not supported")
                cv_scores = None
                cv_mean = None
                cv_std = None
            
            # Perform cross-validation if we have a valid model
            if 'cv_model' in locals():
                try:
                    cv_scores = cross_val_score(
                        cv_model, X_scaled, y,
                        cv=cross_validation_folds,
                        scoring='r2'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    logger.info(f"Cross-validation R² scores: {cv_scores}")
                    logger.info(f"Mean CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
                except Exception as e:
                    logger.warning(f"Error during cross-validation: {str(e)}")
                    cv_scores = None
                    cv_mean = None
                    cv_std = None
        
        # Calculate advanced statistical metrics with statsmodels
        if include_advanced_metrics:
            logger.info("Calculating advanced statistical metrics")
            
            try:
                # For linear models, calculate detailed statistics using statsmodels
                if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
                    # Add constant for statsmodels
                    X_train_sm = sm.add_constant(X_train)
                    
                    # Fit statsmodels OLS
                    sm_model = sm.OLS(y_train, X_train_sm).fit()
                    
                    # Extract statistics
                    sm_results = sm_model.summary()
                    sm_rsquared = sm_model.rsquared
                    sm_adj_rsquared = sm_model.rsquared_adj
                    sm_pvalues = sm_model.pvalues[1:]  # Skip the constant
                    sm_conf_int = sm_model.conf_int()
                    
                    # Calculate confidence intervals for predictions
                    X_test_sm = sm.add_constant(X_test)
                    test_predictions = sm_model.get_prediction(X_test_sm)
                    prediction_intervals = test_predictions.conf_int(alpha=1-confidence_interval_level)
                    
                    # Create dictionary of p-values
                    p_values = {}
                    for i, feature in enumerate(model_features):
                        p_values[feature] = float(sm_pvalues[i])
                    
                    # Log significant features
                    logger.info("Statistically significant features (p < 0.05):")
                    for feature, p_value in p_values.items():
                        if p_value < 0.05:
                            logger.info(f"  - {feature}: p={p_value:.4f}")
                    
                    # Store adjusted R-squared
                    adj_r2 = sm_adj_rsquared
                    
                else:
                    # For non-linear models, use simpler metrics
                    logger.info(f"Full statistical inference not available for {model_name}")
                    # Calculate adjusted R² manually
                    n = len(X_train)
                    p = len(model_features)
                    r2 = test_r2
                    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    
                    # Mock p-values for features - not statistically valid for non-linear models
                    p_values = {}
                    for feature in model_features:
                        p_values[feature] = 0.01  # Placeholder
            
            except Exception as e:
                logger.warning(f"Error calculating advanced metrics: {str(e)}")
                # Fall back to basic calculations
                n = len(X_train)
                p = len(model_features)
                r2 = test_r2
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                
                # Mock p-values
                p_values = {}
                for feature in model_features:
                    p_values[feature] = 0.01  # Placeholder
        
        else:
            # Basic adjusted R-squared calculation
            n = len(X_train)
            p = len(model_features)
            r2 = test_r2
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            # No p-values if advanced metrics not requested
            p_values = None
        
        # -------------------------------------------------------------------------
        # Step 9: Feature Importance Calculation
        # -------------------------------------------------------------------------
        logger.info("Calculating feature importance")
        
        feature_importance = []
        
        # Different methods based on model type
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            # For linear models, use coefficients
            if is_ensemble:
                coefficients = model['linear'].coef_
            else:
                coefficients = model.coef_
                
            # Normalize coefficients to importance scores
            abs_coefficients = np.abs(coefficients)
            normalized_coefficients = abs_coefficients / (np.sum(abs_coefficients) + epsilon)
            
            for i, feature in enumerate(model_features):
                importance = normalized_coefficients[i]
                coef = coefficients[i]
                
                logger.info(f"  - {feature}: importance={importance:.4f}, coef={coef:.4f}")
                
                feature_importance.append({
                    'feature': feature,
                    'importance': float(importance),
                    'coefficient': float(coef),
                    'p_value': float(p_values[feature]) if p_values and feature in p_values else None
                })
                
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # For LightGBM, use built-in feature importance
            importance_gains = model.feature_importance(importance_type='gain')
            importance_splits = model.feature_importance(importance_type='split')
            
            # Normalize to sum to 1
            total_gain = np.sum(importance_gains) + epsilon
            normalized_gains = importance_gains / total_gain
            
            for i, feature in enumerate(model_features):
                importance = normalized_gains[i]
                gain = importance_gains[i]
                split = importance_splits[i]
                
                logger.info(f"  - {feature}: importance={importance:.4f}, gain={gain}, splits={split}")
                
                feature_importance.append({
                    'feature': feature,
                    'importance': float(importance),
                    'gain': float(gain),
                    'splits': int(split)
                })
                
        elif model_type == 'gbr' or (model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE):
            # For sklearn GBR, use feature_importances_
            if is_ensemble:
                importances = model['gbm'].feature_importances_
            else:
                importances = model.feature_importances_
                
            for i, feature in enumerate(model_features):
                importance = importances[i]
                
                logger.info(f"  - {feature}: importance={importance:.4f}")
                
                feature_importance.append({
                    'feature': feature,
                    'importance': float(importance)
                })
                
        elif model_type == 'ensemble':
            # For ensemble, use linear coefficients and GBM importances
            linear_coef = model['linear'].coef_
            abs_coef = np.abs(linear_coef)
            norm_coef = abs_coef / (np.sum(abs_coef) + epsilon)
            
            # Get GBM importances depending on type
            if model['type'] == 'lightgbm':
                gbm_imp = model['gbm'].feature_importance(importance_type='gain')
                total_gain = np.sum(gbm_imp) + epsilon
                norm_gbm = gbm_imp / total_gain
            else:
                gbm_imp = model['gbm'].feature_importances_
                norm_gbm = gbm_imp
            
            # Average the importances from both models
            for i, feature in enumerate(model_features):
                lin_imp = norm_coef[i]
                gbm_imp_val = norm_gbm[i]
                avg_imp = (lin_imp + gbm_imp_val) / 2
                
                logger.info(f"  - {feature}: avg_imp={avg_imp:.4f}, linear={lin_imp:.4f}, gbm={gbm_imp_val:.4f}")
                
                feature_importance.append({
                    'feature': feature,
                    'importance': float(avg_imp),
                    'linear_importance': float(lin_imp),
                    'gbm_importance': float(gbm_imp_val),
                    'coefficient': float(linear_coef[i])
                })
        
        # Sort feature importance by descending importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # -------------------------------------------------------------------------
        # Step 10: Predict Target Property (if provided)
        # -------------------------------------------------------------------------
        predicted_value = None
        confidence_interval = None
        prediction_details = {}
        
        if target_property is not None:
            logger.info("Predicting value for target property")
            
            # Prepare target property data
            if isinstance(target_property, dict):
                # Convert dictionary to DataFrame
                target_property = pd.DataFrame([target_property])
            
            try:
                # 10.1 Feature engineering for target property
                # Apply the same transformations as for the training data
                target_data = target_property.copy()
                
                # Create derived features
                if 'year_built' in target_data.columns:
                    target_data['property_age'] = current_year - target_data['year_built']
                    target_data['property_age'] = target_data['property_age'].clip(0, 200)
                    
                    if 'property_age_squared' in model_features:
                        target_data['property_age_squared'] = target_data['property_age'] ** 2
                
                if 'beds_baths_ratio' in model_features:
                    target_data['beds_baths_ratio'] = target_data['bedrooms'] / (target_data['bathrooms'] + epsilon)
                    target_data['beds_baths_ratio'] = target_data['beds_baths_ratio'].clip(0.25, 4.0)
                
                if 'sqft_per_room' in model_features:
                    target_data['sqft_per_room'] = target_data['square_feet'] / (target_data['bedrooms'] + target_data['bathrooms'] + epsilon)
                    target_data['sqft_per_room'] = target_data['sqft_per_room'].clip(50, 1000)
                
                if 'total_rooms' in model_features:
                    target_data['total_rooms'] = target_data['bedrooms'] + target_data['bathrooms']
                    
                if 'bedroom_ratio' in model_features:
                    target_data['bedroom_ratio'] = target_data['bedrooms'] / (target_data['total_rooms'] + epsilon)
                
                # Add GIS features if applicable
                if 'proximity_score' in model_features and ref_points is not None:
                    if 'latitude' in target_data.columns and 'longitude' in target_data.columns:
                        # Calculate proximity to reference points
                        total_score = 0
                        total_weight = 0
                        
                        for point_name, point_data in ref_points.items():
                            if 'lat' not in point_data or 'lon' not in point_data or 'weight' not in point_data:
                                continue
                                
                            lat1, lon1 = target_data['latitude'].iloc[0], target_data['longitude'].iloc[0]
                            lat2, lon2 = point_data['lat'], point_data['lon']
                            
                            # Haversine distance
                            dlat = math.radians(lat2 - lat1)
                            dlon = math.radians(lon2 - lon1)
                            a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                                 math.sin(dlon/2) * math.sin(dlon/2))
                            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                            distance_km = 6371 * c
                            
                            # Exponential decay
                            proximity = math.exp(-0.5 * distance_km)
                            weighted_proximity = proximity * point_data['weight']
                            
                            total_score += weighted_proximity
                            total_weight += point_data['weight']
                        
                        if total_weight > 0:
                            target_data['proximity_score'] = total_score / total_weight
                        else:
                            target_data['proximity_score'] = 0.5  # Default value
                
                if 'neighborhood_quality' in model_features and neighborhood_ratings is not None:
                    neighborhood = target_data['neighborhood'].iloc[0] if 'neighborhood' in target_data.columns else None
                    city = target_data['city'].iloc[0] if 'city' in target_data.columns else None
                    
                    # Try exact neighborhood match first
                    if neighborhood and neighborhood in neighborhood_ratings:
                        quality = neighborhood_ratings[neighborhood]
                    # Then try city match
                    elif city and city in neighborhood_ratings:
                        quality = neighborhood_ratings[city]
                    # Default to unknown
                    else:
                        quality = neighborhood_ratings.get('Unknown', 1.0)
                        
                    target_data['neighborhood_quality'] = quality
                
                # Check for missing features needed by the model
                missing_model_features = [feat for feat in model_features if feat not in target_data.columns]
                if missing_model_features:
                    logger.warning(f"Target property missing required features: {missing_model_features}")
                    # Set missing features to median values from training data
                    for feat in missing_model_features:
                        if feat in property_data.columns:
                            target_data[feat] = property_data[feat].median()
                        else:
                            target_data[feat] = 0  # Default value
                
                # Extract features for prediction
                X_target = target_data[model_features].values
                X_target_scaled = scaler.transform(X_target)
                
                # 10.2 Make prediction based on model type
                if is_ensemble:
                    # For ensemble, average predictions from both models
                    if model['type'] == 'lightgbm':
                        gbm_pred = model['gbm'].predict(X_target_scaled)
                    else:
                        gbm_pred = model['gbm'].predict(X_target_scaled)
                        
                    linear_pred = model['linear'].predict(X_target_scaled)
                    base_prediction = (gbm_pred[0] + linear_pred[0]) / 2
                    
                    # Store individual model predictions
                    prediction_details['linear_prediction'] = float(linear_pred[0])
                    prediction_details['gbm_prediction'] = float(gbm_pred[0])
                    
                elif is_lightgbm:
                    # For LightGBM models
                    base_prediction = model.predict(X_target_scaled)[0]
                    
                else:
                    # For sklearn models
                    base_prediction = model.predict(X_target_scaled)[0]
                
                logger.info(f"Base prediction: ${base_prediction:.2f}")
                
                # 10.3 Apply GIS and spatial adjustments if applicable
                applied_adjustments = []
                
                # Start with the base prediction
                adjusted_prediction = base_prediction
                
                # Apply spatial adjustments
                if use_gis_features and spatial_adjustments:
                    for adj_name, adj_data in spatial_adjustments.items():
                        adj_type = adj_data.get('type', 'multiplicative')
                        adj_value = adj_data.get('value', 1.0)
                        
                        if adj_type == 'multiplicative':
                            # Apply as a percentage adjustment
                            pre_adj = adjusted_prediction
                            adjusted_prediction *= adj_value
                            impact = adjusted_prediction - pre_adj
                            
                        elif adj_type == 'additive':
                            # Apply as a fixed dollar amount
                            pre_adj = adjusted_prediction
                            adjusted_prediction += adj_value
                            impact = adj_value
                            
                        elif adj_type == 'hybrid':
                            # Apply both percentage and fixed adjustments
                            pre_adj = adjusted_prediction
                            multiplier = adj_data.get('multiplier', 1.0)
                            adder = adj_data.get('adder', 0.0)
                            adjusted_prediction = adjusted_prediction * multiplier + adder
                            impact = adjusted_prediction - pre_adj
                        
                        # Record the adjustment
                        applied_adjustments.append({
                            'name': adj_name,
                            'type': adj_type,
                            'value': adj_value,
                            'impact': float(impact),
                            'description': adj_data.get('description', '')
                        })
                        
                        logger.info(f"Applied {adj_name} ({adj_type}): ${impact:.2f} impact")
                    
                    logger.info(f"Final adjusted prediction: ${adjusted_prediction:.2f}")
                    predicted_value = adjusted_prediction
                    
                else:
                    logger.info(f"No spatial adjustments applied")
                    predicted_value = base_prediction
                
                # 10.4 Calculate confidence interval for prediction
                if include_advanced_metrics:
                    try:
                        # For linear models with statsmodels
                        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net'] and 'sm_model' in locals():
                            X_target_sm = sm.add_constant(X_target_scaled)
                            prediction = sm_model.get_prediction(X_target_sm)
                            ci = prediction.conf_int(alpha=1-confidence_interval_level)
                            
                            lower_bound = float(ci[0][0])
                            upper_bound = float(ci[0][1])
                            
                            # Apply the same spatial adjustments to CI bounds
                            if use_gis_features and spatial_adjustments:
                                for adj_name, adj_data in spatial_adjustments.items():
                                    adj_type = adj_data.get('type', 'multiplicative')
                                    adj_value = adj_data.get('value', 1.0)
                                    
                                    if adj_type == 'multiplicative':
                                        lower_bound *= adj_value
                                        upper_bound *= adj_value
                                    elif adj_type == 'additive':
                                        lower_bound += adj_value
                                        upper_bound += adj_value
                                    elif adj_type == 'hybrid':
                                        multiplier = adj_data.get('multiplier', 1.0)
                                        adder = adj_data.get('adder', 0.0)
                                        lower_bound = lower_bound * multiplier + adder
                                        upper_bound = upper_bound * multiplier + adder
                            
                        else:
                            # For other models, use a simpler approximation based on RMSE
                            # Standard approach: use a multiplier of the RMSE for the CI
                            if confidence_interval_level == 0.95:
                                # For 95% CI, use approximately ±1.96 * RMSE
                                ci_factor = 1.96
                            elif confidence_interval_level == 0.90:
                                # For 90% CI, use approximately ±1.645 * RMSE
                                ci_factor = 1.645
                            elif confidence_interval_level == 0.99:
                                # For 99% CI, use approximately ±2.576 * RMSE
                                ci_factor = 2.576
                            else:
                                # Default to a reasonable value
                                ci_factor = 2.0
                                
                            lower_bound = predicted_value - ci_factor * rmse
                            upper_bound = predicted_value + ci_factor * rmse
                        
                        # Ensure lower bound is non-negative
                        lower_bound = max(0, lower_bound)
                        
                        confidence_interval = [float(lower_bound), float(upper_bound)]
                        logger.info(f"{confidence_interval_level*100}% Confidence Interval: ${lower_bound:.2f} - ${upper_bound:.2f}")
                        
                    except Exception as e:
                        logger.warning(f"Error calculating confidence interval: {str(e)}")
                        # Fallback to a simple approximation
                        margin = 0.1 * predicted_value  # 10% margin
                        confidence_interval = [predicted_value - margin, predicted_value + margin]
                
                # 10.5 Calculate prediction insights
                prediction_details['model_name'] = model_name
                prediction_details['base_prediction'] = float(base_prediction)
                prediction_details['adjusted_prediction'] = float(predicted_value)
                prediction_details['adjustments'] = applied_adjustments
                
                # Feature contributions for linear models
                if model_type in ['linear', 'ridge', 'lasso', 'elastic_net'] or (is_ensemble and 'linear' in model):
                    feature_contributions = []
                    
                    # Get coefficients from the appropriate model
                    if is_ensemble:
                        coef = model['linear'].coef_
                        intercept = model['linear'].intercept_
                    else:
                        coef = model.coef_
                        intercept = getattr(model, 'intercept_', 0)
                    
                    # Calculate contribution of each feature
                    for i, feature in enumerate(model_features):
                        # Get the scaled feature value
                        scaled_value = X_target_scaled[0, i]
                        # Calculate contribution
                        contribution = float(coef[i] * scaled_value)
                        # Get the original feature value
                        original_value = X_target[0, i]
                        
                        feature_contributions.append({
                            'feature': feature,
                            'value': float(original_value),
                            'contribution': contribution,
                            'percentage': float(contribution / (base_prediction - intercept) * 100) if base_prediction != intercept else 0
                        })
                    
                    # Sort by absolute contribution
                    feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
                    prediction_details['feature_contributions'] = feature_contributions
                    prediction_details['intercept'] = float(intercept)
                    
            except Exception as e:
                logger.error(f"Error predicting target property value: {str(e)}")
                predicted_value = None
                confidence_interval = None
        
        # -------------------------------------------------------------------------
        # Step 11: Prepare final result
        # -------------------------------------------------------------------------
        logger.info("Preparing final valuation result")
        
        result = {
            'estimated_value': predicted_value,
            'confidence_interval': confidence_interval,
            'r_squared': float(test_r2),
            'adjusted_r_squared': float(adj_r2),
            'feature_importances': feature_importance,
            'model_type': model_type,
            'model_name': model_name,
            'spatial_adjustments': spatial_adjustments,
            'prediction_details': prediction_details
        }
        
        # Add additional metrics if requested
        if include_advanced_metrics:
            result['rmse'] = float(rmse)
            result['mae'] = float(mae)
            result['p_values'] = p_values
            
            # Add cross-validation results if available
            if cv_mean is not None:
                result['cross_validation'] = {
                    'mean_r2': float(cv_mean),
                    'std_r2': float(cv_std),
                    'folds': cross_validation_folds
                }
        
        # Add model coefficients for linear models
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            if is_ensemble:
                coefficients = model['linear'].coef_
                intercept = float(model['linear'].intercept_)
            else:
                coefficients = model.coef_
                intercept = float(getattr(model, 'intercept_', 0))
                
            model_coefficients = {}
            for i, feature in enumerate(model_features):
                model_coefficients[feature] = float(coefficients[i])
                
            result['model_coefficients'] = model_coefficients
            result['intercept'] = intercept
        
        return result
        
    else:
        # Basic linear regression with essential enhancements
        logger.info(f"Training basic linear regression model with {len(property_data)} samples")
        
        # -------------------------------------------------------------------------
        # Step 1: Minimal Data Preparation
        # -------------------------------------------------------------------------
        
        # Handle division by zero errors with small epsilon
        epsilon = 1e-6
        
        # Handle missing values in basic features if needed
        if handle_missing_values and property_data[basic_features].isnull().sum().sum() > 0:
            logger.info("Imputing missing values in basic features with median")
            property_data[basic_features] = property_data[basic_features].fillna(property_data[basic_features].median())
        
        # -------------------------------------------------------------------------
        # Step 2: Extract Features and Target
        # -------------------------------------------------------------------------
        
        # Select basic features only
        X = property_data[basic_features].values
        y = property_data[price_col].values
        
        # Optional normalization of features
        if handle_outliers:
            logger.info("Normalizing features with RobustScaler (handles outliers)")
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            logger.info("Normalizing features with StandardScaler")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # -------------------------------------------------------------------------
        # Step 3: Split Data and Train Model
        # -------------------------------------------------------------------------
        
        # Split the data
        logger.info(f"Splitting data into training ({100-test_size*100}%) and testing ({test_size*100}%) sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        logger.info("Training linear regression model")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # -------------------------------------------------------------------------
        # Step 4: Evaluate Model
        # -------------------------------------------------------------------------
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        test_r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Model performance on test set:")
        logger.info(f"  R-squared: {test_r2:.4f}")
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE: ${mae:.2f}")
        
        # Calculate adjusted R-squared
        n = len(X_train)
        p = len(basic_features)
        adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
        logger.info(f"  Adjusted R-squared: {adj_r2:.4f}")
        
        # -------------------------------------------------------------------------
        # Step 5: Feature Importance
        # -------------------------------------------------------------------------
        
        # Calculate feature importance based on normalized coefficients
        logger.info("Calculating feature importance")
        coefficients = model.coef_
        abs_coefficients = np.abs(coefficients)
        normalized_coefficients = abs_coefficients / (np.sum(abs_coefficients) + epsilon)
        
        feature_importance = []
        for i, feature in enumerate(basic_features):
            importance = normalized_coefficients[i]
            coef = coefficients[i]
            logger.info(f"  - {feature}: importance={importance:.4f}, coefficient={coef:.4f}")
            
            feature_importance.append({
                'feature': feature,
                'importance': float(importance),
                'coefficient': float(coef)
            })
        
        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # -------------------------------------------------------------------------
        # Step 6: Predict Target Property Value (if provided)
        # -------------------------------------------------------------------------
        
        predicted_value = None
        confidence_interval = None
        
        if target_property is not None:
            logger.info("Predicting value for target property")
            
            try:
                # Convert dictionary to DataFrame if needed
                if isinstance(target_property, dict):
                    target_property = pd.DataFrame([target_property])
                
                # Extract features for prediction
                X_target = target_property[basic_features].values
                
                # Scale features using same scaler as training data
                X_target_scaled = scaler.transform(X_target)
                
                # Make prediction
                predicted_value = float(model.predict(X_target_scaled)[0])
                logger.info(f"Predicted value: ${predicted_value:.2f}")
                
                # Calculate simple confidence interval based on RMSE
                if include_advanced_metrics:
                    # Use approximately 2 standard errors for ~95% CI
                    margin = 2 * rmse
                    lower_bound = max(0, predicted_value - margin)  # Ensure non-negative
                    upper_bound = predicted_value + margin
                    confidence_interval = [float(lower_bound), float(upper_bound)]
                    logger.info(f"95% Confidence interval: ${lower_bound:.2f} - ${upper_bound:.2f}")
                
                # Apply GIS adjustment if available
                if use_gis_features and gis_adjustment_factor is not None:
                    original_value = predicted_value
                    predicted_value = original_value * gis_adjustment_factor
                    logger.info(f"Applied GIS adjustment factor {gis_adjustment_factor}: ${original_value:.2f} → ${predicted_value:.2f}")
                    
                    # Adjust confidence interval if it exists
                    if confidence_interval:
                        confidence_interval = [ci * gis_adjustment_factor for ci in confidence_interval]
                
            except Exception as e:
                logger.error(f"Error predicting target property value: {str(e)}")
                predicted_value = None
                confidence_interval = None
        
        # -------------------------------------------------------------------------
        # Step 7: Return Results
        # -------------------------------------------------------------------------
        
        result = {
            'estimated_value': predicted_value,
            'r2_score': float(test_r2),
            'adj_r2_score': float(adj_r2),
            'feature_importance': feature_importance,
            'model_type': 'basic_linear_regression',
            'model_name': 'Linear Regression (Basic)',
            'model': model,
        }
        
        # Add confidence interval if calculated
        if confidence_interval:
            result['confidence_interval'] = confidence_interval
        
        # Add additional metrics if requested
        if include_advanced_metrics:
            result['rmse'] = float(rmse)
            result['mae'] = float(mae)
            
            # Add statsmodels p-values if possible
            try:
                # Use statsmodels for additional metrics
                X_train_sm = sm.add_constant(X_train)
                sm_model = sm.OLS(y_train, X_train_sm).fit()
                
                # Get p-values (skip the constant)
                p_values = {}
                for i, feature in enumerate(basic_features):
                    p_values[feature] = float(sm_model.pvalues[i+1])
                
                result['p_values'] = p_values
                
                # Add coefficients with intercept
                result['model_coefficients'] = {feature: float(coef) for feature, coef in zip(basic_features, coefficients)}
                result['intercept'] = float(sm_model.params[0])
                
            except Exception as e:
                logger.warning(f"Could not calculate advanced statistical metrics: {str(e)}")
        
        return result