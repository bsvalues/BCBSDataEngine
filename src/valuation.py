"""
Property valuation module for the BCBS_Values system.
Implements valuation models for estimating property values based on features,
including advanced GIS data integration for spatial analysis and location-based valuation.

This module contains multiple valuation functions with different complexity levels:
- train_basic_valuation_model: Basic linear regression for property valuation
- train_multiple_regression_model: Enhanced multiple regression with feature selection and statistics
- estimate_property_value: Standard valuation with feature engineering
- advanced_property_valuation: Advanced modeling with multiple algorithms including LightGBM

The module now features enhanced GIS integration with sophisticated spatial analysis capabilities:
- Improved proximity scoring with exponential decay functions
- School district and quality integration
- Flood zone risk assessment
- Walkability and amenity scoring
- Traffic and noise impact assessment
- View quality estimation
- Housing density analysis
- Future development potential
"""
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression, RFECV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statsmodels.api as sm
from scipy import stats
# Try to import LightGBM for gradient boosting, but continue without it if not available
try:
    import lightgbm as lgb
    has_lightgbm = True
except (ImportError, OSError) as e:
    has_lightgbm = False
    lgb = None
    logging.warning(f"LightGBM not available: {str(e)}. LightGBM and ensemble models will be disabled.")
import warnings
import math
import time
from functools import partial

# Import enhanced GIS integration
try:
    from src.gis_integration import (
        enhance_property_with_gis, 
        process_properties_with_gis,
        calculate_proximity_score,
        calculate_combined_gis_multiplier
    )
    has_enhanced_gis = True
except ImportError:
    # Fall back to built-in GIS functions if enhanced module is not available
    has_enhanced_gis = False
    logging.warning("Enhanced GIS module not available, using basic GIS functions")

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_basic_valuation_model(properties_df):
    """
    Trains a basic linear regression model for property valuation.
    
    This function takes a Pandas DataFrame containing property data,
    cleans and normalizes it, then trains a basic linear regression model
    to predict property values. It returns the predicted values and model metrics.
    
    Args:
        properties_df (pd.DataFrame): DataFrame containing property data.
            Must include numerical features and a price column.
            
    Returns:
        dict: Dictionary containing:
            - 'predictions': DataFrame with original data and predicted values
            - 'model': Trained linear regression model
            - 'r2_score': R-squared score of the model
            - 'feature_importance': Dictionary mapping feature names to importance scores
    
    Example:
        >>> properties = pd.DataFrame({
        ...     'property_id': ['P001', 'P002', 'P003', 'P004'],
        ...     'square_feet': [1500, 2000, 1800, 2200],
        ...     'bedrooms': [3, 4, 3, 4],
        ...     'bathrooms': [2, 2.5, 2, 3],
        ...     'year_built': [1990, 2005, 2000, 2010],
        ...     'list_price': [300000, 400000, 350000, 450000]
        ... })
        >>> result = train_basic_valuation_model(properties)
        >>> print(f"R² Score: {result['r2_score']:.4f}")
        >>> print("Top features by importance:")
        >>> for feature, importance in sorted(result['feature_importance'].items(), 
        ...                                   key=lambda x: x[1], reverse=True)[:3]:
        ...     print(f"  {feature}: {importance:.4f}")
    """
    logger.info("Starting basic property valuation model training")
    
    # Step 1: Make a copy of the input data to avoid modifying the original
    df = properties_df.copy()
    
    # Step 2: Data cleaning and validation
    logger.info("Cleaning and validating input data")
    
    # Identify the price column to use as target
    price_columns = ['list_price', 'estimated_value', 'last_sale_price', 'total_value']
    target_column = None
    
    for col in price_columns:
        if col in df.columns:
            target_column = col
            logger.info(f"Using '{target_column}' as the target price column")
            break
    
    if target_column is None:
        raise ValueError("No price column found in data. Need one of: list_price, estimated_value, last_sale_price, or total_value")
    
    # Step 3: Select and prepare features for modeling
    logger.info("Preparing features for modeling")
    
    # Filter out non-numeric columns except property_id
    id_column = None
    if 'property_id' in df.columns:
        id_column = 'property_id'
    elif 'id' in df.columns:
        id_column = 'id'
    
    # Keep only numeric columns for modeling
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the target column from features
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Check if we have any features left
    if not numeric_cols:
        raise ValueError("No numeric feature columns found in the data")
    
    logger.info(f"Selected {len(numeric_cols)} numeric features for modeling")
    
    # Step 4: Handle missing values
    logger.info("Handling missing values")
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_column])
    
    # For feature columns, fill missing values with median
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {df[col].isna().sum()} missing values in '{col}' with median: {median_val}")
    
    # Step 5: Normalize the features
    logger.info("Normalizing features")
    
    # Create a scaler and fit it to the features
    scaler = StandardScaler()
    X = df[numeric_cols].values
    X_scaled = scaler.fit_transform(X)
    
    # Get the target values
    y = df[target_column].values
    
    # Step 6: Train the linear regression model
    logger.info("Training linear regression model")
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Make predictions on the training data
    y_pred = model.predict(X_scaled)
    
    # Step 7: Calculate model performance metrics
    logger.info("Calculating model performance metrics")
    
    # Calculate R-squared score
    r2 = r2_score(y, y_pred)
    logger.info(f"Model R² score: {r2:.4f}")
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    logger.info(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Step 8: Calculate feature importance
    logger.info("Calculating feature importance")
    
    # Calculate feature importance from model coefficients
    feature_importance = {}
    
    # Standardize coefficients to get relative importance
    coef_abs = np.abs(model.coef_)
    coef_sum = np.sum(coef_abs)
    
    if coef_sum > 0:  # Avoid division by zero
        for i, col in enumerate(numeric_cols):
            feature_importance[col] = coef_abs[i] / coef_sum
    
    # Step 9: Add predictions to the original DataFrame
    df['predicted_value'] = y_pred
    
    # Calculate prediction error and error percentage
    df['prediction_error'] = df[target_column] - df['predicted_value']
    df['error_percentage'] = (df['prediction_error'] / df[target_column]) * 100
    
    # Step 10: Prepare the result
    result = {
        'predictions': df,
        'model': model,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'feature_importance': feature_importance,
        'feature_coefficients': dict(zip(numeric_cols, model.coef_)),
        'scaler': scaler,
        'features_used': numeric_cols
    }
    
    logger.info("Basic property valuation model training completed")
    return result


def train_multiple_regression_model(properties_df, feature_selection=True, max_features=10, test_size=0.2, random_state=42):
    """
    Trains an enhanced multiple regression model for property valuation.
    
    This function expands on the basic valuation model by incorporating:
    1. Advanced feature engineering
    2. Feature selection to identify most predictive attributes
    3. Multiple regression analysis with statistical significance reporting
    4. Cross-validation for more robust performance evaluation
    
    Args:
        properties_df (pd.DataFrame): DataFrame containing property data.
            Must include numerical features and a price column.
        feature_selection (bool, optional): Whether to perform feature selection
            to identify most important predictors. Default is True.
        max_features (int, optional): Maximum number of features to select
            if feature_selection is True. Default is 10.
        test_size (float, optional): Proportion of data to use for testing. Default is 0.2.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
            
    Returns:
        dict: Dictionary containing:
            - 'predictions': DataFrame with original data and predicted values
            - 'model': Trained multiple regression model
            - 'statsmodel': StatsModels regression result with statistical metrics
            - 'r2_score': R-squared score of the model
            - 'adj_r2_score': Adjusted R-squared score of the model
            - 'feature_importance': Dictionary mapping feature names to importance scores
            - 'p_values': Dictionary mapping feature names to p-values
            - 'cross_val_scores': Cross-validation scores
    
    Example:
        >>> properties = pd.DataFrame({
        ...     'property_id': ['P001', 'P002', 'P003', 'P004'],
        ...     'square_feet': [1500, 2000, 1800, 2200],
        ...     'bedrooms': [3, 4, 3, 4],
        ...     'bathrooms': [2, 2.5, 2, 3],
        ...     'year_built': [1990, 2005, 2000, 2010],
        ...     'list_price': [300000, 400000, 350000, 450000]
        ... })
        >>> result = train_multiple_regression_model(properties)
        >>> print(f"R² Score: {result['r2_score']:.4f}")
        >>> print(f"Adjusted R² Score: {result['adj_r2_score']:.4f}")
        >>> # Display statistically significant features (p < 0.05)
        >>> for feature, p_value in result['p_values'].items():
        ...     if p_value < 0.05:
        ...         print(f"  {feature}: p={p_value:.4f}")
    """
    logger.info("Starting enhanced multiple regression model training")
    
    # Step 1: Make a copy of the input data to avoid modifying the original
    df = properties_df.copy()
    
    # Step 2: Data cleaning and validation
    logger.info("Cleaning and validating input data")
    
    # Identify the price column to use as target
    price_columns = ['list_price', 'estimated_value', 'last_sale_price', 'total_value']
    target_column = None
    
    for col in price_columns:
        if col in df.columns:
            target_column = col
            logger.info(f"Using '{target_column}' as the target price column")
            break
    
    if target_column is None:
        raise ValueError("No price column found in data. Need one of: list_price, estimated_value, last_sale_price, or total_value")
    
    # Step 3: Enhanced feature engineering
    logger.info("Performing enhanced feature engineering")
    
    # Keep track of the property ID column if it exists
    id_column = None
    if 'property_id' in df.columns:
        id_column = 'property_id'
    elif 'id' in df.columns:
        id_column = 'id'
    
    # 3.1 Calculate property age from year_built
    if 'year_built' in df.columns:
        current_year = pd.Timestamp.now().year
        df['property_age'] = current_year - df['year_built']
        logger.info("Created 'property_age' feature from 'year_built'")
    
    # 3.2 Create interaction terms (important for capturing combined effects)
    if 'square_feet' in df.columns and 'bedrooms' in df.columns:
        # Square footage per bedroom - indicator of room size
        df['sqft_per_bedroom'] = df['square_feet'] / df['bedrooms'].clip(lower=1)
        logger.info("Created 'sqft_per_bedroom' interaction feature")
    
    if 'square_feet' in df.columns and 'bathrooms' in df.columns:
        # Square footage per bathroom - indicator of bathroom size/quality
        df['sqft_per_bathroom'] = df['square_feet'] / df['bathrooms'].clip(lower=0.5)
        logger.info("Created 'sqft_per_bathroom' interaction feature")
    
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        # Ratio of bedrooms to bathrooms - indicator of property balance
        df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].clip(lower=0.5)
        logger.info("Created 'bed_bath_ratio' interaction feature")
    
    if 'square_feet' in df.columns and 'lot_size' in df.columns:
        # Ratio of house to lot - indicator of property density
        df['house_lot_ratio'] = df['square_feet'] / df['lot_size'].clip(lower=1)
        logger.info("Created 'house_lot_ratio' interaction feature")
    
    # 3.3 Create nonlinear transformations for key numeric features
    numeric_features_to_transform = [
        'square_feet', 'bedrooms', 'bathrooms', 'lot_size', 
        'property_age', 'garage_spaces'
    ]
    
    for feature in numeric_features_to_transform:
        if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
            # Log transformation - helps linearize relationships with diminishing returns
            if (df[feature] > 0).all():
                df[f'{feature}_log'] = np.log(df[feature])
                logger.info(f"Created log transformation for '{feature}'")
            
            # Squared transformation - captures increasing returns
            df[f'{feature}_squared'] = df[feature] ** 2
            logger.info(f"Created squared transformation for '{feature}'")
    
    # 3.4 Create location-based features if lat/long are available
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Calculate distance from the center of the data (proxy for centrality)
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Use Euclidean distance as a simplified proxy (actual haversine distance would be better)
        df['distance_from_center'] = np.sqrt(
            (df['latitude'] - center_lat)**2 + 
            (df['longitude'] - center_lon)**2
        )
        logger.info("Created 'distance_from_center' feature from latitude/longitude")
    
    # 3.5 One-hot encode categorical variables
    categorical_features = ['property_type', 'city', 'neighborhood', 'school_district']
    
    for feature in categorical_features:
        if feature in df.columns:
            # For categorical variables with many categories, limit to top N most frequent
            if df[feature].nunique() > 10:
                top_categories = df[feature].value_counts().nlargest(10).index
                df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'Other')
            
            # Create dummy variables (one-hot encoding)
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            logger.info(f"One-hot encoded '{feature}' into {dummies.shape[1]} dummy variables")
    
    # Step 4: Select and prepare features for modeling
    logger.info("Preparing features for modeling")
    
    # Keep only numeric columns for modeling
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the target column from features
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Remove ID columns if they exist
    if id_column and id_column in numeric_cols:
        numeric_cols.remove(id_column)
    
    # Check if we have any features left
    if not numeric_cols:
        raise ValueError("No numeric feature columns found in the data")
    
    logger.info(f"Selected {len(numeric_cols)} potential features for modeling")
    
    # Step 5: Handle missing values
    logger.info("Handling missing values")
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_column])
    
    # For feature columns, fill missing values with median
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {df[col].isna().sum()} missing values in '{col}' with median: {median_val}")
    
    # Step 6: Split data into training and testing sets
    logger.info("Splitting data into training and testing sets")
    
    X = df[numeric_cols]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Step 7: Normalize features
    logger.info("Normalizing features")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create feature names for the scaled data
    feature_names = X.columns.tolist()
    
    # Step 8: Feature selection (if enabled)
    selector = None  # Initialize selector to avoid "possibly unbound" error
    if feature_selection and len(numeric_cols) > max_features:
        logger.info(f"Performing feature selection to select top {max_features} features")
        
        # Use SelectKBest with f_regression to select features with strongest correlation to target
        selector = SelectKBest(f_regression, k=max_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Get selected feature indices and names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected features: {', '.join(selected_features)}")
        
        # Use the selected features
        X_train_final = X_train_selected
        X_test_final = X_test_selected
        final_features = selected_features
    else:
        # Use all features
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
        final_features = feature_names
    
    # Step 9: Train the multiple regression model using scikit-learn
    logger.info("Training multiple regression model")
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train_final, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_final)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info(f"Test set R² score: {r2:.4f}")
    logger.info(f"Test set MAE: {mae:.2f}")
    logger.info(f"Test set RMSE: {rmse:.2f}")
    
    # Step 10: Perform cross-validation for more robust evaluation
    logger.info("Performing cross-validation")
    
    # Combine feature scaling and model training in a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # If feature selection is enabled, add it to the pipeline
    if feature_selection and len(numeric_cols) > max_features:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_regression, k=max_features)),
            ('model', LinearRegression())
        ])
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    
    logger.info(f"Cross-validation R² scores: {cv_scores}")
    logger.info(f"Mean cross-validation R² score: {cv_scores.mean():.4f}")
    
    # Step 11: Fit a StatsModels OLS model for detailed statistical metrics
    logger.info("Calculating detailed statistical metrics using StatsModels")
    
    # Prepare data for StatsModels (add constant term for intercept)
    if feature_selection and len(numeric_cols) > max_features:
        # Use only selected features for StatsModels
        X_sm = sm.add_constant(X[final_features])
    else:
        # Use all features for StatsModels
        X_sm = sm.add_constant(X)
    
    # Fit StatsModels OLS model
    sm_model = sm.OLS(y, X_sm).fit()
    
    # Extract key statistical metrics
    adj_r2 = sm_model.rsquared_adj
    p_values = sm_model.pvalues.drop('const').to_dict()  # Drop the constant term
    
    logger.info(f"Adjusted R² score: {adj_r2:.4f}")
    
    # Display significant features (p < 0.05)
    significant_features = {k: v for k, v in p_values.items() if v < 0.05}
    if significant_features:
        logger.info("Statistically significant features (p < 0.05):")
        for feature, p_value in sorted(significant_features.items(), key=lambda x: x[1]):
            logger.info(f"  {feature}: p={p_value:.4f}")
    
    # Step 12: Calculate feature importance
    logger.info("Calculating feature importance")
    
    # Calculate feature importance from model coefficients
    feature_importance = {}
    
    # Standardize coefficients to get relative importance
    if feature_selection and len(numeric_cols) > max_features:
        # Use only selected features for importance calculation
        coef_abs = np.abs(model.coef_)
        coef_sum = np.sum(coef_abs)
        
        if coef_sum > 0:  # Avoid division by zero
            for i, feature in enumerate(final_features):
                feature_importance[feature] = coef_abs[i] / coef_sum
    else:
        # Use all features for importance calculation
        coef_abs = np.abs(model.coef_)
        coef_sum = np.sum(coef_abs)
        
        if coef_sum > 0:  # Avoid division by zero
            for i, feature in enumerate(feature_names):
                feature_importance[feature] = coef_abs[i] / coef_sum
    
    # Display top features by importance
    logger.info("Top features by importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Step 13: Make predictions on the full dataset
    logger.info("Making predictions on the full dataset")
    
    # Transform all data using the same pipeline
    if feature_selection and len(numeric_cols) > max_features:
        # Scale all data
        X_all_scaled = scaler.transform(X)
        # Apply feature selection
        X_all_selected = selector.transform(X_all_scaled)
        # Make predictions
        all_predictions = model.predict(X_all_selected)
    else:
        # Scale all data
        X_all_scaled = scaler.transform(X)
        # Make predictions
        all_predictions = model.predict(X_all_scaled)
    
    # Add predictions to the original DataFrame
    df['predicted_value'] = all_predictions
    
    # Calculate prediction error and error percentage
    df['prediction_error'] = df[target_column] - df['predicted_value']
    df['error_percentage'] = (df['prediction_error'] / df[target_column]) * 100
    
    # Step 14: Prepare the result
    result = {
        'predictions': df,
        'model': model,
        'statsmodel': sm_model,
        'r2_score': r2,
        'adj_r2_score': adj_r2,
        'mae': mae,
        'rmse': rmse,
        'feature_importance': feature_importance,
        'p_values': p_values,
        'cross_val_scores': cv_scores,
        'scaler': scaler,
        'features_used': final_features if feature_selection and len(numeric_cols) > max_features else feature_names,
        'summary_text': sm_model.summary().as_text()
    }
    
    logger.info("Enhanced multiple regression model training completed")
    return result

def calculate_gis_features(properties_df, gis_data=None, ref_points=None, neighborhood_ratings=None):
    """
    Calculate GIS-related features for property valuation.
    
    This function integrates GIS spatial data with property information to create
    location-based features that influence property values. These features include
    distance to key reference points, neighborhood quality scores, and proximity
    to amenities.
    
    The function now uses an enhanced GIS integration module when available for more
    sophisticated spatial analysis, including:
    - Improved proximity scoring with exponential decay functions
    - School district and quality integration
    - Flood zone risk assessment
    - Walkability and amenity scoring
    - Traffic and noise impact assessment
    - View quality estimation
    - Housing density analysis
    - Future development potential
    
    Args:
        properties_df (pd.DataFrame): DataFrame containing property data with latitude/longitude
        gis_data (pd.DataFrame, optional): DataFrame containing GIS-specific data like
            flood zones, school districts, etc.
        ref_points (dict, optional): Dictionary of reference points with latitude/longitude
            and importance weights. Example: {'city_center': {'lat': 46.2804, 'lon': -119.2752, 'weight': 1.0}}
        neighborhood_ratings (dict, optional): Dictionary mapping neighborhood names/IDs to
            quality ratings. Example: {'West Richland': 0.85, 'Kennewick': 0.78}
            
    Returns:
        pd.DataFrame: Original DataFrame with added GIS-based features
        
    Example:
        >>> ref_pts = {'downtown': {'lat': 46.2804, 'lon': -119.2752, 'weight': 1.0},
        ...           'school': {'lat': 46.2698, 'lon': -119.2720, 'weight': 0.7}}
        >>> neighborhood_ratings = {'West Richland': 0.9, 'Kennewick': 0.8, 'Richland': 0.85}
        >>> df = calculate_gis_features(properties, ref_points=ref_pts, 
        ...                             neighborhood_ratings=neighborhood_ratings)
    """
    try:
        # Check if enhanced GIS module is available
        if has_enhanced_gis:
            logger.info("Using enhanced GIS integration module")
            
            # Create GIS datasets dictionary for enhanced processing
            gis_datasets = {
                'ref_points': ref_points,
                'neighborhood_ratings': neighborhood_ratings
            }
            
            # Add additional GIS data if provided
            if gis_data is not None:
                if isinstance(gis_data, dict):
                    # If gis_data is a dictionary, merge it with gis_datasets
                    gis_datasets.update(gis_data)
                else:
                    # If it's a DataFrame, add it as 'generic_gis_data'
                    gis_datasets['generic_gis_data'] = gis_data
            
            # Use enhanced processing function
            enhanced_df = process_properties_with_gis(properties_df, gis_datasets)
            
            # Check if enhanced_df has the gis_value_multiplier, if not calculate it
            if 'gis_value_multiplier' not in enhanced_df.columns:
                logger.info("Adding combined GIS value multiplier")
                
                try:
                    # Collect all multiplier factors from various GIS features
                    gis_factors = {}
                    
                    # Location quality (basic)
                    if 'gis_location_quality' in enhanced_df.columns:
                        # Convert 0-1 score to 0.9-1.1 multiplier
                        gis_factors['location_quality'] = 0.9 + 0.2 * enhanced_df['gis_location_quality']
                    
                    # School quality
                    if 'school_quality_score' in enhanced_df.columns:
                        # Convert 0-10 score to 0.95-1.15 multiplier
                        school_impact = 0.95 + 0.02 * enhanced_df['school_quality_score']
                        gis_factors['school_quality'] = school_impact
                    
                    # Flood zone risk (negative impact)
                    if 'flood_zone_risk' in enhanced_df.columns:
                        # Higher risk = lower value (0-5 risk to 0.95-0.75 multiplier)
                        flood_impact = 0.95 - 0.04 * enhanced_df['flood_zone_risk']
                        gis_factors['flood_risk'] = flood_impact
                    
                    # Amenity score
                    if 'amenity_score' in enhanced_df.columns:
                        # 0-1 score to 0.95-1.1 multiplier
                        amenity_impact = 0.95 + 0.15 * enhanced_df['amenity_score']
                        gis_factors['amenities'] = amenity_impact
                    
                    # View quality
                    if 'view_score' in enhanced_df.columns:
                        # 0-10 score to 1.0-1.2 multiplier for good views
                        view_impact = 1.0 + 0.02 * enhanced_df['view_score']
                        gis_factors['view'] = view_impact
                    
                    # Traffic and noise (negative impact)
                    if 'traffic_noise_level' in enhanced_df.columns:
                        # 0-10 score to 1.0-0.9 multiplier (higher noise = lower value)
                        traffic_impact = 1.0 - 0.01 * enhanced_df['traffic_noise_level']
                        gis_factors['traffic'] = traffic_impact
                    
                    # Proximity to reference points 
                    if 'proximity_score' in enhanced_df.columns:
                        # 0-1 score to 0.95-1.15 multiplier
                        proximity_impact = 0.95 + 0.2 * enhanced_df['proximity_score']
                        gis_factors['proximity'] = proximity_impact
                    
                    # Future growth potential
                    if 'growth_potential' in enhanced_df.columns:
                        # 0-5 score to 1.0-1.15 multiplier
                        growth_impact = 1.0 + 0.03 * enhanced_df['growth_potential']
                        gis_factors['growth'] = growth_impact
                    
                    # Combine all factors
                    if len(gis_factors) > 0:
                        # Start with neutral multiplier
                        enhanced_df['gis_value_multiplier'] = 1.0
                        
                        # Apply each factor
                        for factor, impact in gis_factors.items():
                            enhanced_df['gis_value_multiplier'] *= impact
                        
                        # Limit to reasonable range (0.7 to 1.5)
                        enhanced_df['gis_value_multiplier'] = enhanced_df['gis_value_multiplier'].clip(0.7, 1.5)
                        
                        # Add a GIS summary for reporting
                        enhanced_df['gis_summary'] = enhanced_df.apply(
                            lambda row: {
                                factor: float(impact.loc[row.name]) if hasattr(impact, 'loc') else float(impact)
                                for factor, impact in gis_factors.items()
                            },
                            axis=1
                        )
                        
                        logger.info(f"Created combined GIS value multiplier with {len(gis_factors)} factors")
                except Exception as e:
                    logger.warning(f"Error calculating GIS value multiplier: {str(e)}")
                    
            # Return the enhanced DataFrame
            logger.info("Completed enhanced GIS feature calculation")
            return enhanced_df
            
        # Fall back to original implementation if enhanced module is not available
        logger.info("Using standard GIS feature calculation (enhanced module not available)")
        
        # Make a copy to avoid modifying the original DataFrame
        df = properties_df.copy()
        
        # ======== 1. Basic coordinate validation ========
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.warning("GIS feature calculation requires latitude and longitude coordinates")
            return df
            
        # Ensure coordinates are numeric and within reasonable ranges
        # (valid latitude: -90 to 90, valid longitude: -180 to 180)
        valid_coords = (
            df['latitude'].between(-90, 90, inclusive='both') & 
            df['longitude'].between(-180, 180, inclusive='both')
        )
        
        if not valid_coords.any():
            logger.warning("No valid coordinates found in data. GIS features not calculated.")
            return df
            
        # Filter to only valid coordinates for GIS calculations
        coord_df = df[valid_coords].copy()
        logger.info(f"Found {len(coord_df)}/{len(df)} properties with valid coordinates")

        # ======== 2. Calculate basic location features ========
        
        # 2.1 Calculate centrality score (distance from center of the dataset)
        # This indicates how central a property is relative to other properties
        center_lat, center_lon = coord_df['latitude'].mean(), coord_df['longitude'].mean()
        
        # Haversine formula for accurate Earth distance calculation
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the great circle distance between two points on Earth."""
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Earth radius in kilometers
            return c * r
            
        # Apply to all valid coordinates and normalize
        df.loc[valid_coords, 'centrality_km'] = df.loc[valid_coords].apply(
            lambda row: haversine_distance(row['latitude'], row['longitude'], center_lat, center_lon),
            axis=1
        )
        
        # 2.2 Calculate median property value by grid cell
        # Divide area into grid cells and calculate median property value in each cell
        if 'list_price' in df.columns or 'estimated_value' in df.columns or 'last_sale_price' in df.columns:
            # Determine which price column to use
            if 'list_price' in df.columns:
                price_col = 'list_price'
            elif 'estimated_value' in df.columns:
                price_col = 'estimated_value'
            else:
                price_col = 'last_sale_price'
                
            # Create grid cells (0.01 degree is roughly 1km)
            df.loc[valid_coords, 'lat_grid'] = np.floor(df.loc[valid_coords, 'latitude'] * 100) / 100
            df.loc[valid_coords, 'lon_grid'] = np.floor(df.loc[valid_coords, 'longitude'] * 100) / 100
            
            # Get median prices by grid cell
            grid_prices = df.loc[valid_coords].groupby(['lat_grid', 'lon_grid'])[price_col].median().to_dict()
            
            # Map back to properties
            def get_grid_price(row):
                if pd.isna(row['lat_grid']) or pd.isna(row['lon_grid']):
                    return np.nan
                grid_key = (row['lat_grid'], row['lon_grid'])
                return grid_prices.get(grid_key, np.nan)
                
            df['location_price_index'] = df.apply(get_grid_price, axis=1)
            
            # Calculate price ratio relative to location
            df.loc[df['location_price_index'].notna() & df[price_col].notna(), 'location_price_ratio'] = (
                df.loc[df['location_price_index'].notna() & df[price_col].notna(), price_col] / 
                df.loc[df['location_price_index'].notna() & df[price_col].notna(), 'location_price_index']
            )
            
            logger.info("Created location-based price index features")
        
        # ======== 3. Process reference points if provided ========
        if ref_points and isinstance(ref_points, dict):
            # Create distance columns for each reference point
            for point_name, point_data in ref_points.items():
                if 'lat' in point_data and 'lon' in point_data:
                    # Calculate distance to this reference point
                    col_name = f"dist_to_{point_name}"
                    df.loc[valid_coords, col_name] = df.loc[valid_coords].apply(
                        lambda row: haversine_distance(
                            row['latitude'], row['longitude'], 
                            point_data['lat'], point_data['lon']
                        ),
                        axis=1
                    )
                    
                    # Apply importance weight if specified
                    weight = point_data.get('weight', 1.0)
                    if weight != 1.0:
                        df[f"{col_name}_weighted"] = df[col_name] * weight
                        
                    logger.info(f"Created distance feature to {point_name}")
            
            # Create combined proximity score (inverse of distance) - higher is better
            dist_cols = [col for col in df.columns if col.startswith('dist_to_')]
            if dist_cols:
                # Standardize distances
                for col in dist_cols:
                    mean_dist = df[col].mean()
                    std_dist = df[col].std()
                    if std_dist > 0:
                        df[f"{col}_std"] = (df[col] - mean_dist) / std_dist
                
                # Combined proximity score (inverse of standardized distances)
                std_dist_cols = [f"{col}_std" for col in dist_cols if f"{col}_std" in df.columns]
                if std_dist_cols:
                    df['proximity_score'] = -df[std_dist_cols].mean(axis=1)
                    logger.info("Created combined proximity score from reference points")
        
        # ======== 4. Apply neighborhood ratings if provided ========
        if neighborhood_ratings and isinstance(neighborhood_ratings, dict):
            if 'city' in df.columns:
                # Map neighborhood ratings to properties
                df['neighborhood_rating'] = df['city'].map(neighborhood_ratings)
                logger.info("Added neighborhood rating based on city field")
            elif 'zip_code' in df.columns:
                # Try to use zip code mapping if available
                zip_ratings = {zip_code: rating for zip_code, rating in neighborhood_ratings.items() 
                              if isinstance(zip_code, (str, int)) and str(zip_code).isdigit()}
                if zip_ratings:
                    df['neighborhood_rating'] = df['zip_code'].astype(str).map(zip_ratings)
                    logger.info("Added neighborhood rating based on zip code")
            
            # Fill missing neighborhood ratings with median value
            if 'neighborhood_rating' in df.columns:
                median_rating = df['neighborhood_rating'].median()
                missing_ratings = df['neighborhood_rating'].isna().sum()
                if missing_ratings > 0 and not np.isnan(median_rating):
                    df['neighborhood_rating'].fillna(median_rating, inplace=True)
                    logger.info(f"Filled {missing_ratings} missing neighborhood ratings with median: {median_rating:.2f}")
        
        # ======== 5. Create aggregate location quality score ========
        # Combine various GIS factors into a single quality score
        location_factors = []
        
        if 'proximity_score' in df.columns:
            location_factors.append('proximity_score')
            
        if 'neighborhood_rating' in df.columns:
            location_factors.append('neighborhood_rating')
            
        if 'location_price_ratio' in df.columns:
            location_factors.append('location_price_ratio')
        
        if location_factors:
            # Normalize each factor to 0-1 scale
            for factor in location_factors:
                min_val = df[factor].min()
                max_val = df[factor].max()
                if max_val > min_val:
                    df[f"{factor}_norm"] = (df[factor] - min_val) / (max_val - min_val)
                else:
                    df[f"{factor}_norm"] = 0.5  # Default if no variation
            
            # Calculate combined location quality score
            norm_factors = [f"{factor}_norm" for factor in location_factors]
            df['gis_location_quality'] = df[norm_factors].mean(axis=1)
            
            # Create a price adjustment multiplier based on location quality
            # Typical adjustment range: 0.8 (poor location) to 1.2 (excellent location)
            df['gis_price_multiplier'] = 0.8 + 0.4 * df['gis_location_quality']
            
            logger.info(f"Created GIS location quality score from {len(location_factors)} factors")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in GIS feature calculation: {str(e)}", exc_info=True)
        return properties_df  # Return original DataFrame on error

def estimate_property_value(property_data, target_property=None, test_size=0.2, random_state=42,
                       gis_data=None, ref_points=None, neighborhood_ratings=None, use_gis_features=True):
    """
    Estimates property value using linear regression with integrated GIS attributes.
    
    This function trains a linear regression model on property data to predict property values
    based on key features like square footage, bedrooms, bathrooms, property age, and now
    including GIS/spatial data for enhanced location-based valuation.
    
    Args:
        property_data (pd.DataFrame): DataFrame containing property data with features and prices
        target_property (pd.DataFrame, optional): Single property to predict value for.
            If None, returns model performance metrics only.
        test_size (float, optional): Proportion of data to use for testing (default: 0.2)
        random_state (int, optional): Random seed for reproducibility (default: 42)
        gis_data (pd.DataFrame, optional): GIS data with spatial attributes like flood zones, 
            school districts, etc.
        ref_points (dict, optional): Dictionary of reference points (downtown, schools, etc.) 
            with latitude/longitude and importance weights
        neighborhood_ratings (dict, optional): Dictionary mapping neighborhoods to quality ratings
        use_gis_features (bool, optional): Whether to incorporate GIS features in the model (default: True)
    
    Returns:
        dict: Dictionary containing predicted value (if target_property provided),
              model performance metrics, feature importance, and GIS adjustment factors
    
    Example:
        >>> df = pd.DataFrame({
        ...     'square_feet': [1500, 2000, 1800, 2200],
        ...     'bedrooms': [3, 4, 3, 4],
        ...     'bathrooms': [2, 2.5, 2, 3],
        ...     'year_built': [1990, 2005, 2000, 2010],
        ...     'list_price': [300000, 400000, 350000, 450000],
        ...     'latitude': [46.2543, 46.2671, 46.2812, 46.2953],
        ...     'longitude': [-119.2681, -119.2344, -119.2876, -119.3012]
        ... })
        >>> target = pd.DataFrame({
        ...     'square_feet': [1750],
        ...     'bedrooms': [3],
        ...     'bathrooms': [2],
        ...     'year_built': [1995],
        ...     'latitude': [46.2712],
        ...     'longitude': [-119.2543]
        ... })
        >>> ref_pts = {'downtown': {'lat': 46.2804, 'lon': -119.2752, 'weight': 1.0}}
        >>> neighborhood_ratings = {'West Richland': 0.9, 'Kennewick': 0.8, 'Richland': 0.85}
        >>> estimate_property_value(df, target, ref_points=ref_pts, 
        ...                        neighborhood_ratings=neighborhood_ratings)
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
        
        # ******** GIS INTEGRATION ********
        # Step 2b: Add GIS/spatial features if enabled
        gis_features_available = False
        has_enhanced_gis = False
        available_gis_features = []  # List to track available GIS features
        if use_gis_features:
            logger.info("Checking for GIS features")
            
            # Check if basic GIS coordinates are available
            has_coordinates = 'latitude' in df.columns and 'longitude' in df.columns
            
            if has_coordinates:
                logger.info("Coordinates found in dataset, calculating GIS features")
                
                # Calculate GIS features using the helper function
                df = calculate_gis_features(df, gis_data, ref_points, neighborhood_ratings)
                
                # Check which GIS features were successfully calculated
                gis_columns = [
                    # Basic GIS features from original implementation
                    'centrality_km',          # Distance to center of dataset
                    'location_price_index',   # Price index based on grid location
                    'proximity_score',        # Proximity to reference points
                    'neighborhood_rating',    # Quality rating of the neighborhood
                    'gis_location_quality',   # Overall location quality score
                    'gis_price_multiplier',   # Price adjustment multiplier based on location
                    
                    # Enhanced GIS features from new implementation
                    'gis_value_multiplier',   # Combined GIS value multiplier with multiple factors
                    'amenity_score',          # Score based on proximity to amenities
                    'flood_zone_risk',        # Flood zone risk assessment
                    'school_quality_score',   # School quality impact on value
                    'view_score',             # View quality score
                    'traffic_noise_level',    # Traffic and noise impact
                    'housing_density',        # Housing density analysis
                    'growth_potential'        # Future development potential
                ]
                
                available_gis_features = [col for col in gis_columns if col in df.columns]
                
                if available_gis_features:
                    gis_features_available = True
                    
                    # Check if we have enhanced GIS features available (from new GIS integration)
                    enhanced_gis_features = ['gis_value_multiplier', 'amenity_score', 'flood_zone_risk',
                                           'school_quality_score', 'view_score', 'traffic_noise_level',
                                           'housing_density', 'growth_potential']
                    if any(feature in available_gis_features for feature in enhanced_gis_features):
                        has_enhanced_gis = True
                        logger.info("Enhanced GIS features detected")
                    
                    # Add available GIS features to the model features
                    for feature in available_gis_features:
                        # For any GIS feature, handle missing values by filling with median
                        if df[feature].isna().any():
                            median_value = df[feature].median()
                            logger.info(f"Filling missing values in {feature} with median: {median_value:.4f}")
                            df[feature].fillna(median_value, inplace=True)
                    
                    # Add core GIS features to the model if available
                    if 'centrality_km' in available_gis_features:
                        features.append('centrality_km')  # Distance from center
                    
                    if 'proximity_score' in available_gis_features:
                        features.append('proximity_score')  # Proximity to reference points
                    
                    if 'neighborhood_rating' in available_gis_features:
                        features.append('neighborhood_rating')  # Neighborhood quality
                    
                    # Note: We don't include gis_price_multiplier as a feature because
                    # we'll use it as a post-prediction adjustment
                    
                    logger.info(f"Added {len(features) - 6} GIS features to the model")
                else:
                    logger.warning("No GIS features were successfully calculated")
            else:
                logger.warning("No latitude/longitude coordinates found for GIS feature calculation")
        
        # Prepare feature matrix X and target vector y
        X = df[features]
        y = df[target_column]
        
        # Handle any remaining NaN values in features
        if X.isna().any().any():
            logger.warning("Detected NaN values in features, filling with median values")
            for col in X.columns:
                if X[col].isna().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):  # If median is also NaN, use 0
                        logger.warning(f"Median for {col} is NaN, using 0 instead")
                        median_val = 0
                    missing_count = X[col].isna().sum()
                    logger.info(f"Filling {missing_count} missing values in {col} with {median_val:.4f}")
                    X[col] = X[col].fillna(median_val)
        
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
        gis_adjustment_factor = None
        if target_property is not None:
            logger.info("Predicting value for target property")
            
            # Prepare target property features
            target_df = target_property.copy()
            
            # Apply same feature engineering to target property
            target_df['property_age'] = current_year - target_df['year_built']
            target_df['beds_baths_ratio'] = target_df['bedrooms'] / target_df['bathrooms'].clip(lower=0.5)
            target_df['sqft_per_room'] = target_df['square_feet'] / (target_df['bedrooms'] + target_df['bathrooms']).clip(lower=1.0)
            
            # ******** GIS INTEGRATION FOR TARGET PROPERTY ********
            # If we used GIS features for modeling, we need them for prediction too
            if gis_features_available and use_gis_features:
                # Check if target property has coordinates
                has_target_coords = 'latitude' in target_df.columns and 'longitude' in target_df.columns
                
                if has_target_coords:
                    logger.info("Calculating GIS features for target property")
                    
                    # Calculate GIS features for the target property
                    target_df = calculate_gis_features(target_df, gis_data, ref_points, neighborhood_ratings)
                    
                    # Extract the location quality multiplier if available
                    # First check for enhanced GIS multiplier, otherwise fall back to basic multiplier
                    if has_enhanced_gis and 'gis_value_multiplier' in target_df.columns:
                        gis_adjustment_factor = float(target_df['gis_value_multiplier'].iloc[0])
                        logger.info(f"Enhanced GIS value multiplier: {gis_adjustment_factor:.4f}")
                    elif 'gis_price_multiplier' in target_df.columns:
                        gis_adjustment_factor = float(target_df['gis_price_multiplier'].iloc[0])
                        logger.info(f"Basic GIS location adjustment factor: {gis_adjustment_factor:.4f}")
                    
                    # For missing GIS features in target property, fill with median from training data
                    for feature in [f for f in features if f not in target_df.columns]:
                        if feature in df.columns:
                            median_value = df[feature].median()
                            logger.info(f"Target property missing {feature}, using median: {median_value:.4f}")
                            target_df[feature] = median_value
                else:
                    logger.warning("Target property missing coordinates, GIS features will be estimated")
                    
                    # For missing GIS features in target, fill with median from training data
                    for feature in [f for f in features if f not in target_df.columns]:
                        if feature in df.columns:
                            median_value = df[feature].median()
                            logger.info(f"Target property missing {feature}, using median: {median_value:.4f}")
                            target_df[feature] = median_value
            
            # Make sure all required features are present
            missing_features = [f for f in features if f not in target_df.columns]
            if missing_features:
                logger.warning(f"Target property missing features: {missing_features}")
                for feature in missing_features:
                    if feature in df.columns:
                        target_df[feature] = df[feature].median()
            
            # Extract features and scale them
            target_features = target_df[features]
            
            # Handle any NaN values in target features
            if target_features.isna().any().any():
                logger.warning("Detected NaN values in target property features, filling with median values")
                for col in target_features.columns:
                    if target_features[col].isna().any():
                        median_val = df[col].median()
                        if pd.isna(median_val):  # If median is also NaN, use 0
                            logger.warning(f"Median for {col} is NaN, using 0 instead")
                            median_val = 0
                        logger.info(f"Filling missing values in target property's {col} with {median_val:.4f}")
                        target_features[col] = target_features[col].fillna(median_val)
            
            target_scaled = scaler.transform(target_features)
            
            # Make base prediction
            base_prediction = float(model.predict(target_scaled)[0])
            
            # ******** APPLY GIS ADJUSTMENT ********
            # Apply GIS-based location adjustment if available
            if gis_adjustment_factor is not None:
                # Apply the location quality multiplier
                adjusted_prediction = base_prediction * gis_adjustment_factor
                
                # Log the adjustment effect
                adjustment_amount = adjusted_prediction - base_prediction
                adjustment_percent = (gis_adjustment_factor - 1.0) * 100
                
                logger.info(f"Base prediction: ${base_prediction:,.2f}")
                logger.info(f"GIS adjustment: {adjustment_percent:+.2f}% (${adjustment_amount:+,.2f})")
                logger.info(f"Adjusted prediction: ${adjusted_prediction:,.2f}")
                
                predicted_value = adjusted_prediction
            else:
                # If no GIS adjustment available, use base prediction
                predicted_value = base_prediction
                logger.info(f"Predicted property value: ${predicted_value:,.2f}")
        
        # Return results
        result = {
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
        
        # Add GIS-specific information to the results
        if gis_features_available and use_gis_features:
            # Get list of all GIS features used in the model
            gis_feature_list = [f for f in features if f in [
                'centrality_km', 'proximity_score', 'neighborhood_rating', 
                'gis_location_quality', 'amenity_score', 'flood_zone_risk',
                'school_quality_score', 'view_score', 'traffic_noise_level',
                'housing_density', 'growth_potential'
            ]]
            
            # Create more comprehensive GIS metrics reporting
            result['gis_metrics'] = {
                'features_used': gis_feature_list,
                'adjustment_factor': gis_adjustment_factor,
                'location_quality_calculated': 'gis_location_quality' in df.columns,
                'enhanced_gis_used': has_enhanced_gis and 'gis_value_multiplier' in df.columns,
                'gis_features_available': available_gis_features
            }
            
            # For target property, include detailed GIS summary if available
            if target_property is not None and has_enhanced_gis and 'target_df' in locals():
                if 'gis_summary' in target_df.columns:
                    result['gis_metrics']['target_summary'] = target_df['gis_summary'].iloc[0]
                elif 'gis_analysis' in target_df.columns:
                    result['gis_metrics']['target_analysis'] = target_df['gis_analysis'].iloc[0]
        
        return result
        
    except Exception as e:
        logger.error(f"Error in property valuation: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'predicted_value': None,
            'r2_score': None,
            'feature_importance': None
        }


def advanced_property_valuation(property_data, target_property=None, test_size=0.2, random_state=42,
                          feature_selection='auto', poly_degree=2, regularization=None, alpha=1.0,
                          gis_data=None, ref_points=None, neighborhood_ratings=None, use_gis_features=True,
                          model_type='linear', lightgbm_params=None, normalize_features=True,
                          feature_selection_method='f_regression', cv_folds=5):
    """
    Advanced property valuation using multiple regression analysis with comprehensive
    data processing, feature engineering, and model evaluation, now with GIS data integration
    and support for LightGBM models.
    
    This function performs sophisticated property valuation using multiple regression techniques,
    including polynomial features, regularization, feature selection, and statistical analysis
    of results with p-values and confidence intervals. It integrates spatial (GIS) data
    to account for location-based factors that significantly affect property values, and now
    supports advanced gradient boosting models through LightGBM.
    
    Args:
        property_data (pd.DataFrame): DataFrame containing property data with features and prices
        target_property (pd.DataFrame, optional): Single property to predict value for.
            If None, returns model performance metrics only.
        test_size (float, optional): Proportion of data to use for testing (default: 0.2)
        random_state (int, optional): Random seed for reproducibility (default: 42)
        feature_selection (str, optional): Method for feature selection ('auto', 'k-best', 'rfe', None)
        poly_degree (int, optional): Degree of polynomial features to generate (default: 2)
        regularization (str, optional): Type of regularization to use ('ridge', 'lasso', 'elastic', None)
        alpha (float, optional): Regularization strength parameter (default: 1.0)
        gis_data (pd.DataFrame, optional): Supplementary GIS data to enhance location analysis
        ref_points (dict, optional): Dictionary of reference points with lat/lon coordinates and importance weights
            Example: {'downtown': {'lat': 46.2804, 'lon': -119.2752, 'weight': 1.0}}
        neighborhood_ratings (dict, optional): Dictionary mapping neighborhoods or zip codes to quality ratings
            Example: {'West Richland': 0.9, 'Kennewick': 0.8, 'Richland': 0.85}
        use_gis_features (bool, optional): Whether to incorporate GIS features in the valuation model
        model_type (str, optional): The type of model to use ('linear', 'lightgbm', 'ensemble')
            - 'linear': Uses sklearn's linear regression with optional regularization
            - 'lightgbm': Uses LightGBM gradient boosting for potentially more accurate predictions
            - 'ensemble': Combines multiple models for more robust predictions (default: 'linear')
        lightgbm_params (dict, optional): Parameters for the LightGBM model if model_type='lightgbm'
            Default parameters will be applied if None is provided
        normalize_features (bool, optional): Whether to normalize features before training (default: True)
            This can significantly improve performance of linear models
        feature_selection_method (str, optional): Method for feature importance estimation
            Options: 'f_regression', 'mutual_info', 'model_specific' (default: 'f_regression')
        cv_folds (int, optional): Number of cross-validation folds for model evaluation (default: 5)
            
    Returns:
        dict: Dictionary containing predicted value, model performance metrics, 
              feature importance, statistical significance, and more.
    """
    # Initialize variables that might be referenced in the error case
    gis_multiplier = None
    gis_adjusted_value = None
    target_with_gis = None
    
    try:
        # Make a copy to avoid modifying original data
        df = property_data.copy()
        
        # Step 1: Data preparation and cleaning with enhanced error checks
        logger.info("Preparing property data for advanced valuation model")
        
        # Identify potential target columns and select the best available one
        price_columns = ['list_price', 'estimated_value', 'last_sale_price', 'total_value']
        available_price_cols = [col for col in price_columns if col in df.columns]
        
        if not available_price_cols:
            raise ValueError("No price column found in data. Required columns: list_price, "
                           "estimated_value, last_sale_price, or total_value")
            
        # Select the target column with the most non-null values
        target_column = max(available_price_cols, 
                           key=lambda col: df[col].notna().sum() if col in df.columns else 0)
        logger.info(f"Selected target column for valuation: {target_column}")
        
        # Step 2: Enhanced feature selection - identify all potentially useful features
        # A. Mandatory core features (required for valuation)
        core_features = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        
        # B. Important secondary features (use if available)
        secondary_features = [
            'lot_size', 'garage_spaces', 'stories', 
            'total_rooms', 'property_type'
        ]
        
        # C. Location features (for spatial considerations)
        location_features = [
            'city', 'zip_code', 'latitude', 'longitude'
        ]
        
        # D. Value-related features (may have correlation with target)
        value_features = [
            'land_value', 'improvement_value', 'assessment_year',
            'days_on_market', 'last_sale_date'
        ]
        
        # Check for required core features
        missing_core = [feat for feat in core_features if feat not in df.columns]
        if missing_core:
            raise ValueError(f"Missing required core features: {', '.join(missing_core)}")
        
        # Identify available features from each category
        available_secondary = [f for f in secondary_features if f in df.columns]
        available_location = [f for f in location_features if f in df.columns]
        available_value = [f for f in value_features if f in df.columns]
        
        logger.info(f"Available secondary features: {len(available_secondary)}/{len(secondary_features)}")
        logger.info(f"Available location features: {len(available_location)}/{len(location_features)}")
        logger.info(f"Available value features: {len(available_value)}/{len(value_features)}")
        
        # Create a list of all features to check for null values (excluding target column)
        all_potential_features = core_features + available_secondary + available_location + available_value
        
        # Step 3: Advanced data cleaning and preprocessing
        logger.info("Performing advanced data cleaning and preprocessing")
        
        # A. Drop rows with missing values in core features and target column
        required_columns = core_features + [target_column]
        df = df.dropna(subset=required_columns)
        
        if len(df) < 5:  # Minimum sample size check
            logger.warning(f"Insufficient data for reliable model training (< 5 samples, found {len(df)})")
            return {
                'error': 'Insufficient data for model training',
                'predicted_value': None,
                'r2_score': None,
                'feature_importance': None
            }
        
        logger.info(f"Training model with {len(df)} samples")
        
        # B. Impute missing values for non-core features if necessary
        # Fill numeric columns with median values
        numeric_features = df[all_potential_features].select_dtypes(include=['number']).columns
        for col in numeric_features:
            if df[col].isna().any():
                median_value = df[col].median()
                logger.info(f"Imputing {df[col].isna().sum()} missing values in '{col}' with median: {median_value}")
                df[col] = df[col].fillna(median_value)
        
        # C. Handle categorical features through encoding
        categorical_features = [f for f in all_potential_features if f in df.columns and 
                               df[f].dtype == 'object' and f != target_column]
        
        if categorical_features:
            logger.info(f"Processing categorical features: {categorical_features}")
            # One-hot encode categorical features with low cardinality
            for feature in categorical_features:
                if df[feature].nunique() <= 10:  # Only encode if fewer than 10 unique values
                    # Convert to categorical type first
                    df[feature] = df[feature].astype('category')
                    # Create dummy variables with proper prefixing
                    dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
                    # Add to dataframe
                    df = pd.concat([df, dummies], axis=1)
                    logger.info(f"One-hot encoded '{feature}' with {dummies.shape[1]} categories")
        
        # Step 4: Comprehensive feature engineering
        logger.info("Performing comprehensive feature engineering")
        
        # A. Create standard features as in the basic model
        current_year = pd.Timestamp.now().year
        df['property_age'] = current_year - df['year_built']
        df['beds_baths_ratio'] = df['bedrooms'] / df['bathrooms'].clip(lower=0.5)
        df['sqft_per_room'] = df['square_feet'] / (df['bedrooms'] + df['bathrooms']).clip(lower=1.0)
        
        # A2. Integrate GIS features if enabled
        if use_gis_features:
            logger.info("Integrating GIS features for spatial valuation factors")
            
            # Process through the GIS feature calculation function
            df_with_gis = calculate_gis_features(
                df, 
                gis_data=gis_data, 
                ref_points=ref_points, 
                neighborhood_ratings=neighborhood_ratings
            )
            
            # Merge back any new GIS features
            gis_columns = [col for col in df_with_gis.columns if col not in df.columns]
            if gis_columns:
                logger.info(f"Added {len(gis_columns)} GIS-based features: {', '.join(gis_columns)}")
                df = df_with_gis
            else:
                logger.warning("No GIS features were created - check if valid GIS data is available")
                
            # If we have a GIS price multiplier, log it for later use in prediction
            if 'gis_price_multiplier' in df.columns:
                logger.info(f"GIS price multiplier range: {df['gis_price_multiplier'].min():.2f} to {df['gis_price_multiplier'].max():.2f}")
                
            # If we have a location quality score, log it for transparency
            if 'gis_location_quality' in df.columns:
                logger.info(f"GIS location quality score range: {df['gis_location_quality'].min():.2f} to {df['gis_location_quality'].max():.2f}")
        
        # B. Create advanced engineered features
        
        # B1. Price per square foot ratio (if historical values available)
        if 'last_sale_price' in df.columns and df['last_sale_price'].notna().any():
            df['price_per_sqft_historical'] = df['last_sale_price'] / df['square_feet'].clip(lower=1.0)
            logger.info("Created price_per_sqft_historical feature")
        
        # B2. Age-related features
        df['is_new'] = (df['property_age'] <= 5).astype(int)  # New property flag
        df['age_category'] = pd.cut(df['property_age'], 
                                   bins=[0, 5, 15, 30, 50, 100, 1000],
                                   labels=['new', 'recent', 'established', 'mature', 'historic', 'antique'])
        # Convert age_category to dummy variables
        age_dummies = pd.get_dummies(df['age_category'], prefix='age', drop_first=True)
        df = pd.concat([df, age_dummies], axis=1)
        
        # B3. Size-related interaction terms
        df['bed_bath_product'] = df['bedrooms'] * df['bathrooms']  # Interaction term
        
        # Ensure numeric type before division
        if 'total_rooms' in df.columns:
            # Convert from categorical to numeric if needed
            if df['total_rooms'].dtype.name == 'category':
                df['total_rooms'] = pd.to_numeric(df['total_rooms'], errors='coerce')
            if not pd.api.types.is_numeric_dtype(df['total_rooms']):
                df['total_rooms'] = pd.to_numeric(df['total_rooms'], errors='coerce')
            
            # Only create the feature if we have valid numeric data
            if pd.api.types.is_numeric_dtype(df['total_rooms']):
                df['total_rooms_ratio'] = df['total_rooms'] / df['square_feet'].clip(lower=1.0) * 1000
            else:
                logger.warning("Could not create total_rooms_ratio - total_rooms column is not numeric")
        
        # B4. Location-based features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Ensure numeric type before calculations
            for col in ['latitude', 'longitude']:
                if df[col].dtype.name == 'category':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Only proceed if columns are numeric
            if pd.api.types.is_numeric_dtype(df['latitude']) and pd.api.types.is_numeric_dtype(df['longitude']):
                # Calculate distance from city center or reference point (simplified example)
                # Here you would normally use actual coordinates of city centers
                # For demonstration, we use the mean coordinates as a reference point
                ref_lat, ref_lon = df['latitude'].mean(), df['longitude'].mean()
                
                # Calculate distance using Haversine formula (simplified for small distances)
                df['location_score'] = np.sqrt(
                    (df['latitude'] - ref_lat)**2 + 
                    (df['longitude'] - ref_lon)**2
                ) * 111  # Approximate conversion to kilometers (111km per degree)
                logger.info("Created location_score feature based on coordinates")
            else:
                logger.warning("Could not create location_score feature - latitude or longitude are not numeric")
        
        # B5. Zip code analysis if available
        if 'zip_code' in df.columns:
            # Ensure we have enough zip codes for meaningful grouping
            try:
                # Convert zip_code to string if it's a category to avoid comparison issues
                if df['zip_code'].dtype.name == 'category':
                    df['zip_code_str'] = df['zip_code'].astype(str)
                    use_col = 'zip_code_str'
                else:
                    use_col = 'zip_code'
                
                unique_zips = df[use_col].nunique()
                if unique_zips >= 2:  # Only if we have at least 2 different zip codes
                    # Calculate median price by zip code
                    zip_price_medians = df.groupby(use_col)[target_column].median()
                    df['zip_price_index'] = df[use_col].map(zip_price_medians)
                    
                    # Ensure non-zero values before division
                    valid_indexes = df['zip_price_index'].notna() & (df['zip_price_index'] > 0)
                    if valid_indexes.any():
                        df.loc[valid_indexes, 'zip_price_ratio'] = (
                            df.loc[valid_indexes, target_column] / 
                            df.loc[valid_indexes, 'zip_price_index'].clip(lower=1.0)
                        )
                    logger.info(f"Created zip code price index features for {unique_zips} zip codes")
                else:
                    logger.info(f"Not enough unique zip codes ({unique_zips}) to create meaningful zip code features")
            except Exception as e:
                logger.warning(f"Could not create zip code price features: {str(e)}")
        
        # Step 5: Feature selection process
        logger.info("Performing feature selection")
        
        # First, identify all available features (excluding the target and non-numeric/dummy encoded categorical)
        # Start with numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Add dummy-encoded columns (usually start with prefix and contain '_')
        dummy_cols = [col for col in df.columns if '_' in col and col != target_column 
                     and col not in ['last_sale_date', 'listing_date', 'import_date']]
        
        # Combine all potential predictors
        all_features = list(set(numeric_cols + dummy_cols))
        
        # Remove target column and any date columns from feature list
        all_features = [f for f in all_features if f != target_column and 'date' not in f.lower()]
        
        # Also remove ID-like columns and other non-predictive features
        exclude_patterns = ['id', 'parcel', 'apn', 'listing_id', 'property_id', 'mls_id']
        all_features = [f for f in all_features if not any(pattern in f.lower() for pattern in exclude_patterns)]
        
        logger.info(f"Identified {len(all_features)} potential predictor features")
        
        # Prepare feature matrix X and target vector y
        X = df[all_features].copy()
        y = df[target_column]
        
        # Handle any remaining NaN values in features
        # Convert any categorical columns to numeric or string types
        for col in X.columns:
            if X[col].dtype.name == 'category':
                # Try to convert to numeric, fallback to string
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X[col] = X[col].astype(str)
        
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            
        # Fill non-numeric columns with most frequent value
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if not non_numeric_cols.empty:
            for col in non_numeric_cols:
                # For non-numeric columns, try to convert to numeric first, otherwise use string
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    # If successfully converted to numeric, fill NaNs with median
                    X[col] = X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0)
                except:
                    # Otherwise, keep as string/categorical and fill with most frequent value
                    if not X[col].mode().empty:
                        X[col] = X[col].fillna(X[col].mode().iloc[0])
                    else:
                        # If no mode exists, use a neutral numeric value
                        X[col] = 0
        
        # Apply feature selection if specified
        if feature_selection == 'k-best':
            # Select k best features based on F-statistic
            k = min(15, len(all_features))  # Select up to 15 features or all if fewer
            
            # Handle any remaining NaN values before feature selection
            X_clean = X.fillna(0)  # Replace any remaining NaN with zeros
            y_clean = y.fillna(y.median() if not pd.isna(y.median()) else 0)  # Use median for target
            
            selector = SelectKBest(f_regression, k=k)
            X_selected = selector.fit_transform(X_clean, y_clean)
            
            # Get names of selected features
            selected_indices = selector.get_support(indices=True)
            selected_features = [all_features[i] for i in selected_indices]
            
            logger.info(f"K-best feature selection: {len(selected_features)} features selected")
            
            # Update X with selected features only
            X = df[selected_features].copy()
            
        elif feature_selection == 'rfe':
            # Recursive feature elimination
            estimator = LinearRegression()
            min_features = min(10, len(all_features))  # Select at least 5 features
            
            # Handle any remaining NaN values before feature selection
            X_clean = X.fillna(0)  # Replace any remaining NaN with zeros
            y_clean = y.fillna(y.median() if not pd.isna(y.median()) else 0)  # Use median for target
            
            selector = RFE(estimator, n_features_to_select=min_features, step=1)
            X_selected = selector.fit_transform(X_clean, y_clean)
            
            # Get names of selected features
            selected_features = [all_features[i] for i, selected in enumerate(selector.support_) if selected]
            
            logger.info(f"RFE feature selection: {len(selected_features)} features selected")
            
            # Update X with selected features only
            X = df[selected_features].copy()
            
        elif feature_selection == 'auto':
            # Automatic feature selection based on correlation with target
            # First ensure there are no NaN values that would affect correlation
            X_clean = X.fillna(0)
            y_clean = y.fillna(y.median() if not pd.isna(y.median()) else 0)
            
            # Calculate correlations, handling potential errors
            try:
                correlations = X_clean.corrwith(y_clean).abs().sort_values(ascending=False)
                # Select features with correlation > 0.1 and limit to top 15
                corr_threshold = 0.1
                selected_features = correlations[correlations > corr_threshold].index.tolist()[:15]
            except Exception as e:
                logger.warning(f"Error in correlation calculation: {str(e)}")
                # Fallback: use top 10 features
                # Store empty correlations dataframe to avoid referencing errors later
                selected_features = X.columns.tolist()[:10]
                correlations = pd.Series(index=selected_features)
            
            # Ensure we have at least 5 features
            if len(selected_features) < 5 and len(all_features) >= 5:
                # Use the columns we already have if correlations failed
                try:
                    if 'correlations' in locals() and hasattr(correlations, 'index'):
                        selected_features = correlations.index.tolist()[:5]
                    else:
                        selected_features = X.columns.tolist()[:5]
                except Exception:
                    # Fallback: just use first 5 columns
                    selected_features = X.columns.tolist()[:5]
                
            logger.info(f"Auto feature selection: {len(selected_features)} features selected")
            
            # Update X with selected features only
            X = df[selected_features].copy()
        else:
            # No feature selection (feature_selection is None, 'none' or any other value)
            # Use all features
            selected_features = all_features
            logger.info(f"Using all {len(selected_features)} features without selection")
        
        # Step 6: Advanced data preprocessing
        logger.info("Applying advanced data preprocessing")
        
        # A. Apply robust scaling to handle outliers better
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # B. Generate polynomial features if requested
        poly = None  # Initialize poly variable
        if poly_degree > 1:
            logger.info(f"Generating polynomial features of degree {poly_degree}")
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_scaled)
            
            # Update feature names for polynomial terms
            if hasattr(poly, 'get_feature_names_out'):
                poly_features = poly.get_feature_names_out([f"feature_{i}" for i in range(X_scaled.shape[1])])
            else:
                # Fallback for older scikit-learn versions
                poly_features = [f"poly_{i}" for i in range(X_poly.shape[1])]
                
            logger.info(f"Generated {X_poly.shape[1]} polynomial features")
            
            # Use polynomial features for modeling
            X_processed = X_poly
        else:
            # Just use scaled features
            X_processed = X_scaled
            poly_features = selected_features
        
        # Step 7: Split data into training and testing sets
        logger.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        
        # Step 8: Feature normalization if requested
        if normalize_features and model_type != 'lightgbm':  # LightGBM can handle non-normalized features well
            logger.info("Normalizing features for linear models")
            # Use MinMaxScaler for interpretability of coefficients (esp. for LightGBM)
            # or StandardScaler for better numerical stability in linear models
            if model_type == 'linear':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
                
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # If we have a target property, normalize it too
            if target_property is not None:
                target_features = target_property[selected_features].values
                target_features = scaler.transform(target_features)
        else:
            scaler = None
            
        # Validate model_type parameter based on available models
        valid_model_types = ['linear']
        if has_lightgbm:
            valid_model_types.extend(['lightgbm', 'ensemble'])
            
        if model_type not in valid_model_types:
            if model_type in ['lightgbm', 'ensemble'] and not has_lightgbm:
                logger.warning(f"Model type '{model_type}' requested but LightGBM is not available. Falling back to 'linear' model.")
                model_type = 'linear'
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_model_types}")
        
        # Step 9: Train the model based on selected model type
        logger.info(f"Training model with type: {model_type}")
        
        # Define default parameters for LightGBM if none provided and LightGBM is available
        if has_lightgbm and model_type == 'lightgbm' and lightgbm_params is None:
            lightgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 100
            }
            logger.info("Using default LightGBM parameters")
            
        # Create and train the model based on model_type
        if model_type == 'lightgbm' and has_lightgbm:
            # Convert data to LightGBM dataset format for faster training
            logger.info("Training LightGBM model for gradient boosting regression")
            lgb_train = lgb.Dataset(X_train, y_train, feature_name=selected_features)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
            
            # Train LightGBM model with early stopping
            model = lgb.train(
                lightgbm_params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_eval],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            # Store feature importance for LightGBM
            lgb_importance = model.feature_importance(importance_type='gain')
            feature_importance = dict(zip(selected_features, lgb_importance))
            
        elif model_type == 'ensemble' and has_lightgbm:
            # For ensemble, we train both a linear model and LightGBM model
            # Then average their predictions (can be weighted)
            logger.info("Training ensemble of linear and LightGBM models")
            
            # Train linear model
            if regularization == 'ridge':
                linear_model = Ridge(alpha=alpha, random_state=random_state)
            elif regularization == 'lasso':
                linear_model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
            elif regularization == 'elastic':
                linear_model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=random_state, max_iter=2000)
            else:
                linear_model = LinearRegression()
                
            linear_model.fit(X_train, y_train)
            
            # Train LightGBM model
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_model = lgb.train(lightgbm_params or {
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'n_estimators': 100
            }, lgb_train)
            
            # Create a composite model (simple class to wrap both models)
            class EnsembleModel:
                def __init__(self, linear_model, lgb_model, linear_weight=0.4):
                    self.linear_model = linear_model
                    self.lgb_model = lgb_model
                    self.linear_weight = linear_weight
                    self.lgb_weight = 1.0 - linear_weight
                    
                def predict(self, X):
                    linear_pred = self.linear_model.predict(X)
                    lgb_pred = self.lgb_model.predict(X)
                    return (linear_pred * self.linear_weight) + (lgb_pred * self.lgb_weight)
                    
                def feature_importances(self, feature_names):
                    # Combine feature importances from both models
                    linear_importance = np.abs(self.linear_model.coef_)
                    linear_importance = linear_importance / np.sum(linear_importance)
                    
                    lgb_importance = self.lgb_model.feature_importance(importance_type='gain')
                    # Normalize LightGBM importance
                    if np.sum(lgb_importance) > 0:
                        lgb_importance = lgb_importance / np.sum(lgb_importance)
                    
                    # Weighted average of feature importances
                    combined_importance = (linear_importance * self.linear_weight + 
                                         lgb_importance * self.lgb_weight)
                    
                    return dict(zip(feature_names, combined_importance))
            
            # Create the ensemble model
            model = EnsembleModel(linear_model, lgb_model)
            feature_importance = model.feature_importances(selected_features)
            
        else:  # Default to linear models
            if regularization == 'ridge':
                logger.info(f"Using Ridge regression with alpha={alpha}")
                model = Ridge(alpha=alpha, random_state=random_state)
            elif regularization == 'lasso':
                logger.info(f"Using Lasso regression with alpha={alpha}")
                model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
            elif regularization == 'elastic':
                logger.info(f"Using ElasticNet regression with alpha={alpha}")
                model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=random_state, max_iter=2000)
            else:
                logger.info("Using standard LinearRegression")
                model = LinearRegression()
            
            # Fit the linear model
            model.fit(X_train, y_train)
            
            # Calculate feature importance for linear models
            if hasattr(model, 'coef_'):
                # Standardize coefficients to get relative importance
                coef_abs = np.abs(model.coef_)
                if np.sum(coef_abs) > 0:  # Avoid division by zero
                    feature_importance = dict(zip(selected_features, coef_abs / np.sum(coef_abs)))
                else:
                    feature_importance = dict(zip(selected_features, np.zeros_like(coef_abs)))
            else:
                feature_importance = {}
        
        # Step 9: Comprehensive model evaluation
        logger.info("Performing comprehensive model evaluation")
        
        # A. Basic metrics
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate various metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        logger.info(f"R² (train): {r2_train:.4f}, R² (test): {r2_test:.4f}")
        logger.info(f"RMSE (train): ${rmse_train:.2f}, RMSE (test): ${rmse_test:.2f}")
        logger.info(f"MAE (train): ${mae_train:.2f}, MAE (test): ${mae_test:.2f}")
        
        # B. Cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X_processed, y, cv=min(5, len(X)), scoring='r2')
        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean CV R²: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")
        
        # C. Statistical significance testing with statsmodels
        # Add constant for statsmodels
        X_sm = sm.add_constant(X_scaled)
        
        # Fit statsmodels OLS for detailed statistics
        sm_model = sm.OLS(y, X_sm).fit()
        logger.info(f"statsmodels Summary:\nR²: {sm_model.rsquared:.4f}, Adj. R²: {sm_model.rsquared_adj:.4f}")
        logger.info(f"F-statistic: {sm_model.fvalue:.4f}, p-value: {sm_model.f_pvalue:.6f}")
        
        # Extract p-values for coefficients
        p_values = sm_model.pvalues[1:]  # Skip the constant
        
        # D. Calculate feature importance
        # For linear models, importance is based on the absolute standardized coefficient magnitude
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                # Linear models
                coefs = model.coef_
            else:
                # Multi-output models
                coefs = model.coef_[0]
        else:
            # For models without direct coefficients
            coefs = np.zeros(len(selected_features))
            
        # Create feature importance DataFrame
        if poly_degree > 1:
            # For polynomial features, we need to summarize importance
            # This is simplified; a more detailed approach would map poly features to original features
            importance = np.abs(coefs)
            feature_importance = pd.DataFrame({
                'feature': [f"poly_feature_{i}" for i in range(len(importance))],
                'importance': importance,
                'coefficient': coefs
            })
        else:
            # For linear features, directly map coefficients to features
            importance = np.abs(coefs)
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': importance,
                'coefficient': coefs,
                'p_value': p_values if len(p_values) == len(selected_features) else [np.nan] * len(selected_features)
            })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Add statistical significance markers
        if 'p_value' in feature_importance.columns:
            feature_importance['significant'] = feature_importance['p_value'] < 0.05
            
        logger.info("Feature importance with statistical significance:")
        for _, row in feature_importance.iterrows():
            significance = ""
            if 'p_value' in row and 'significant' in row:
                if row['significant']:
                    significance = f"(p={row['p_value']:.4f}) *"
                else:
                    significance = f"(p={row['p_value']:.4f})"
            
            logger.info(f"  - {row['feature']}: {row['importance']:.4f} (coef: {row['coefficient']:.4f}) {significance}")
        
        # Step 10: Predict value for target property if provided
        predicted_value = None
        prediction_interval = None
        
        if target_property is not None:
            logger.info("Predicting value for target property")
            
            # Prepare target property features
            target_df = target_property.copy()
            
            # Apply same feature engineering as training data
            # Assuming core features are present in target_property
            # If target_property is missing features, we need to handle accordingly
            
            # A. Standard feature engineering as in training with NaN handling
            
            # Ensure all the necessary core features are present and numeric
            for feature in core_features:
                if feature not in target_df.columns:
                    logger.warning(f"Missing core feature '{feature}' in target property. Using median from training data.")
                    target_df[feature] = df[feature].median()
                elif pd.isna(target_df[feature].iloc[0]):
                    logger.warning(f"NaN value in core feature '{feature}' in target property. Using median from training data.")
                    target_df[feature] = df[feature].median()
                elif not pd.api.types.is_numeric_dtype(target_df[feature]):
                    logger.warning(f"Non-numeric value in core feature '{feature}' in target property. Converting to numeric.")
                    target_df[feature] = pd.to_numeric(target_df[feature], errors='coerce')
                    if pd.isna(target_df[feature].iloc[0]):
                        target_df[feature] = df[feature].median()
            
            # Now calculate the engineered features safely
            target_df['property_age'] = current_year - target_df['year_built']
            target_df['beds_baths_ratio'] = target_df['bedrooms'] / target_df['bathrooms'].clip(lower=0.5)
            target_df['sqft_per_room'] = target_df['square_feet'] / (target_df['bedrooms'] + target_df['bathrooms']).clip(lower=1.0)
            
            # Fill any NaN values that might have been created during the calculations
            target_df['property_age'] = target_df['property_age'].fillna(df['property_age'].median())
            target_df['beds_baths_ratio'] = target_df['beds_baths_ratio'].fillna(df['beds_baths_ratio'].median())
            target_df['sqft_per_room'] = target_df['sqft_per_room'].fillna(df['sqft_per_room'].median())
            
            # B. Additional features based on what's available
            # Check if target has required features & create matching structure
            target_features = []
            for feature in selected_features:
                # Get the feature value and handle NaN values
                if feature in target_df.columns:
                    value = target_df[feature].iloc[0]
                elif feature == 'property_age':
                    value = target_df['property_age'].iloc[0]
                elif feature == 'beds_baths_ratio':
                    value = target_df['beds_baths_ratio'].iloc[0]
                elif feature == 'sqft_per_room':
                    value = target_df['sqft_per_room'].iloc[0]
                else:
                    # Feature not available, use median from training data
                    if feature in df.columns:
                        try:
                            # Check if the feature is categorical
                            if df[feature].dtype.name == 'category':
                                # Get the most common value
                                value = df[feature].value_counts().index[0]
                            else:
                                # For numeric features use median
                                value = df[feature].median()
                        except Exception as e:
                            logger.warning(f"Error getting replacement value for {feature}: {str(e)}")
                            value = 0
                    else:
                        value = 0
                
                # Check for NaN value and replace if needed
                if pd.isna(value):
                    logger.warning(f"NaN value detected for feature '{feature}' in target property. Replacing with 0.")
                    value = 0
                
                target_features.append(value)
            
            # Convert to numpy array for scaling
            target_features = np.array(target_features).reshape(1, -1)
            
            # Double-check for any remaining NaN values
            if np.isnan(target_features).any():
                logger.warning("NaN values detected in target_features array. Replacing with zeros.")
                target_features = np.nan_to_num(target_features, nan=0.0)
            
            logger.info(f"Prepared {len(selected_features)} features for prediction")
            
            # Scale the features - use the already cleaned target_features
            target_scaled = scaler.transform(target_features)
            
            # Generate polynomial features if used in training
            if poly_degree > 1 and poly is not None:
                target_processed = poly.transform(target_scaled)
            else:
                target_processed = target_scaled
            
            # Make prediction
            predicted_value = float(model.predict(target_processed)[0])
            
            # Apply GIS location adjustment if we have the quality multiplier
            gis_adjusted_value = predicted_value
            gis_multiplier = None
            
            # Check if we have GIS data for the target property
            if use_gis_features and target_property is not None:
                target_with_gis = calculate_gis_features(
                    target_property, 
                    gis_data=gis_data,
                    ref_points=ref_points,
                    neighborhood_ratings=neighborhood_ratings
                )
                
                # Apply GIS price multiplier if available
                if 'gis_price_multiplier' in target_with_gis.columns:
                    gis_multiplier = float(target_with_gis['gis_price_multiplier'].iloc[0])
                    gis_adjusted_value = predicted_value * gis_multiplier
                    logger.info(f"Applying GIS location multiplier: {gis_multiplier:.4f}")
                    logger.info(f"GIS adjusted value: ${gis_adjusted_value:,.2f} " +
                               f"({'increased' if gis_multiplier > 1 else 'decreased'} by " +
                               f"${abs(gis_adjusted_value - predicted_value):,.2f})")
                    
                    # Update the predicted value to use the GIS-adjusted value
                    predicted_value = gis_adjusted_value
                else:
                    logger.warning("No GIS price multiplier available for target property")
            
            # Calculate prediction interval using statsmodels
            # We use the standard error of the prediction for a rough confidence interval
            # This is a simplified approach; a more rigorous method would use the full prediction variance
            
            if hasattr(sm_model, 'get_prediction'):
                # For statsmodels, we need to match the format including the constant term
                target_sm = np.concatenate([np.ones(1).reshape(1, -1), target_scaled], axis=1)
                sm_pred = sm_model.get_prediction(target_sm)
                prediction_std_error = float(sm_pred.se_mean)
                
                # Calculate 95% confidence interval
                # Using t-distribution for small samples, z-distribution for large
                if len(X) < 30:
                    t_value = stats.t.ppf(0.975, df=len(X)-len(selected_features)-1)
                else:
                    t_value = stats.norm.ppf(0.975)  # 1.96 for 95% CI
                
                margin_of_error = t_value * prediction_std_error
                prediction_interval = [predicted_value - margin_of_error, predicted_value + margin_of_error]
                
                logger.info(f"Predicted property value: ${predicted_value:,.2f}")
                logger.info(f"95% Prediction interval: ${prediction_interval[0]:,.2f} to ${prediction_interval[1]:,.2f}")
            else:
                # Simple fallback if full statsmodels prediction isn't available
                # Use RMSE as a rough approximation of prediction error
                prediction_interval = [predicted_value - rmse_test, predicted_value + rmse_test]
                logger.info(f"Predicted property value: ${predicted_value:,.2f}")
                logger.info(f"Approximate 68% prediction interval (±RMSE): ${prediction_interval[0]:,.2f} to ${prediction_interval[1]:,.2f}")
        
        # Step 11: Return comprehensive results
        return {
            'predicted_value': predicted_value,
            'prediction_interval': prediction_interval,
            'r2_score': float(r2_test),
            'r2_train': float(r2_train),
            'rmse': float(rmse_test),
            'mae': float(mae_test),
            'cross_val_r2': float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
            'cross_val_std': float(np.std(cv_scores)) if len(cv_scores) > 0 else None,
            'feature_importance': feature_importance.to_dict('records'),
            'model_statistics': {
                'f_statistic': float(sm_model.fvalue) if hasattr(sm_model, 'fvalue') else None,
                'f_pvalue': float(sm_model.f_pvalue) if hasattr(sm_model, 'f_pvalue') else None,
                'adj_r2': float(sm_model.rsquared_adj) if hasattr(sm_model, 'rsquared_adj') else None,
            },
            'model_params': {
                'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None,
                'feature_count': len(selected_features),
                'selected_features': selected_features,
                'regularization': regularization
            },
            'data_metrics': {
                'sample_size': len(df),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'mean_price': float(df[target_column].mean()),
                'median_price': float(df[target_column].median()),
                'price_range': [float(df[target_column].min()), float(df[target_column].max())]
            },
            'model_config': {
                'feature_selection': feature_selection,
                'poly_degree': poly_degree,
                'regularization': regularization,
                'alpha': alpha
            },
            'gis_factors': {
                'used_gis_features': use_gis_features,
                'location_multiplier': gis_multiplier,
                'base_prediction': (gis_adjusted_value / gis_multiplier if gis_multiplier and gis_multiplier != 0 and gis_adjusted_value is not None 
                                   else predicted_value),
                'location_quality': (target_with_gis['gis_location_quality'].iloc[0] 
                                    if use_gis_features and 
                                       target_property is not None and 
                                       target_with_gis is not None and
                                       isinstance(target_with_gis, pd.DataFrame) and
                                       'gis_location_quality' in target_with_gis.columns and
                                       not target_with_gis.empty
                                    else None),
                'reference_points': ref_points
            }
        }
        
    except Exception as e:
        logger.error(f"Error in advanced property valuation: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'predicted_value': None,
            'r2_score': None,
            'feature_importance': None,
            'gis_factors': {
                'used_gis_features': use_gis_features,
                'location_multiplier': None,
                'base_prediction': None,
                'location_quality': None,
                'reference_points': ref_points,
                'error': f"Failed to calculate GIS features: {str(e)}"
            }
        }