"""
Property valuation module for the BCBS_Values system.
Implements valuation models for estimating property values based on features.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from scipy import stats
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


def advanced_property_valuation(property_data, target_property=None, test_size=0.2, random_state=42,
                          feature_selection='auto', poly_degree=2, regularization=None, alpha=1.0):
    """
    Advanced property valuation using multiple regression analysis with comprehensive
    data processing, feature engineering, and model evaluation.
    
    This function performs sophisticated property valuation using multiple regression techniques,
    including polynomial features, regularization, feature selection, and statistical analysis
    of results with p-values and confidence intervals.
    
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
        
    Returns:
        dict: Dictionary containing predicted value, model performance metrics, 
              feature importance, statistical significance, and more.
    """
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
        
        # Step 8: Train the regression model with appropriate regularization
        logger.info("Training regression model with selected configuration")
        
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
        
        # Fit the model
        model.fit(X_train, y_train)
        
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
            }
        }
        
    except Exception as e:
        logger.error(f"Error in advanced property valuation: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'predicted_value': None,
            'r2_score': None,
            'feature_importance': None
        }