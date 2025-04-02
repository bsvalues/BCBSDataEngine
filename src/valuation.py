"""
Valuation Engine for the BCBS Values application.

This module provides classes and functions for property valuation using various
regression techniques and model selection.
"""
import json
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Set up logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
LIGHTGBM_AVAILABLE = True
XGBOOST_AVAILABLE = True

try:
    import lightgbm as lgb
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available, LightGBM model will be skipped")

try:
    import xgboost as xgb
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, XGBoost model will be skipped")


class ValuationEngine:
    """Base class for property valuation engines."""
    
    def __init__(self):
        """Initialize the valuation engine."""
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'last_sale_price'
        self.model_metrics = {}
        self.best_model_name = None
    
    def prepare_data(self, data):
        """Prepare data for model training and prediction.
        
        Args:
            data: DataFrame containing property data
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        # Make sure all required columns are present
        required_columns = ['bedrooms', 'bathrooms', 'square_feet', 'year_built']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")
        
        # Select feature columns
        self.feature_columns = [
            'bedrooms', 'bathrooms', 'square_feet', 'year_built',
            'lot_size'
        ]
        
        # Add optional columns if they exist
        optional_columns = ['latitude', 'longitude']
        for col in optional_columns:
            if col in data.columns:
                self.feature_columns.append(col)
        
        # Create dummy variables for categorical columns
        categorical_columns = ['property_type', 'neighborhood']
        for col in categorical_columns:
            if col in data.columns:
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                self.feature_columns.extend(dummies.columns.tolist())
        
        # Create feature matrix and target
        X = data[self.feature_columns].copy()
        
        # Check if target column exists
        if self.target_column not in data.columns:
            raise ValueError(f"Target column {self.target_column} not found in data")
        
        y = data[self.target_column].copy()
        
        return X, y
    
    def train_model(self, data):
        """Train regression models and select the best one.
        
        Args:
            data: DataFrame containing property data
            
        Returns:
            dict: Training results with model metrics
        """
        X, y = self.prepare_data(data)
        
        # Create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store the scaler for later use
        self.scalers['standard'] = scaler
        
        # Train linear regression model
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        self.models['linear_regression'] = lr
        
        # Calculate metrics
        y_pred = lr.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate feature coefficients and p-values using statsmodels
        X_train_sm = sm.add_constant(X_train_scaled)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        p_values = model_sm.pvalues[1:].tolist()  # Skip the constant
        
        self.model_metrics['linear_regression'] = {
            'r2': r2,
            'adj_r2': 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1),
            'rmse': rmse,
            'mae': mae,
            'feature_coefficients': lr.coef_.tolist(),
            'p_values': p_values,
            'feature_names': self.feature_columns
        }
        
        # Set the best model to linear regression initially
        self.best_model_name = 'linear_regression'
        
        return {
            'best_model': self.best_model_name,
            'metrics': self.model_metrics
        }
    
    def predict(self, property_data, model_name=None):
        """Predict property value using trained models.
        
        Args:
            property_data: DataFrame or dict containing property features
            model_name: Name of the model to use for prediction (optional)
                        If None, the best model will be used
        
        Returns:
            dict: Prediction results with estimated value and model metrics
        """
        if not self.models:
            raise ValueError("No trained models available. Call train_model first.")
        
        # Use the best model if model_name is not specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Convert dict to DataFrame if needed
        if isinstance(property_data, dict):
            property_data = pd.DataFrame([property_data])
        
        # Prepare the data
        X, _ = self.prepare_data(property_data)
        
        # Scale the features
        X_scaled = self.scalers['standard'].transform(X)
        
        # Predict with the selected model
        model = self.models[model_name]
        estimated_value = model.predict(X_scaled)[0]
        
        # Get model metrics
        metrics = self.model_metrics.get(model_name, {})
        
        # Prepare the result
        result = {
            'estimated_value': float(estimated_value),
            'valuation_method': model_name,
            'confidence_score': float(metrics.get('r2', 0.0)),
            'model_metrics': metrics
        }
        
        return result


class AdvancedValuationEngine(ValuationEngine):
    """Advanced valuation engine with multiple regression techniques and model selection."""
    
    def train_model(self, data):
        """Train multiple regression models and select the best one.
        
        Args:
            data: DataFrame containing property data
            
        Returns:
            dict: Training results with model metrics
        """
        X, y = self.prepare_data(data)
        
        # Create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store the scaler for later use
        self.scalers['standard'] = scaler
        
        # Define models to train
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                objective='regression',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100
            )
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=100
            )
        
        # Train models and calculate metrics
        best_r2 = -float('inf')
        self.best_model_name = None
        
        for name, model in models.items():
            logger.info(f"Training model: {name}")
            
            try:
                # Train the model
                model.fit(X_train_scaled, y_train)
                self.models[name] = model
                
                # Predict on test set
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
                
                # Calculate feature importance for tree-based models
                feature_importance = None
                if name in ['lightgbm', 'xgboost']:
                    feature_importance = model.feature_importances_.tolist()
                
                # Calculate feature coefficients for linear models
                feature_coefficients = None
                p_values = None
                if name in ['linear_regression', 'ridge_regression', 'lasso_regression', 'elastic_net']:
                    feature_coefficients = model.coef_.tolist()
                    
                    # Calculate p-values using statsmodels for linear models
                    if name == 'linear_regression':
                        X_train_sm = sm.add_constant(X_train_scaled)
                        model_sm = sm.OLS(y_train, X_train_sm).fit()
                        p_values = model_sm.pvalues[1:].tolist()  # Skip the constant
                
                # Calculate adjusted R-squared
                adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
                
                # Store metrics
                self.model_metrics[name] = {
                    'r2': r2,
                    'adj_r2': adj_r2,
                    'rmse': rmse,
                    'mae': mae,
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean_r2': cv_scores.mean(),
                    'feature_importance': feature_importance,
                    'feature_coefficients': feature_coefficients,
                    'p_values': p_values,
                    'feature_names': self.feature_columns
                }
                
                # Update best model if this one is better
                if adj_r2 > best_r2:
                    best_r2 = adj_r2
                    self.best_model_name = name
                    logger.info(f"New best model: {name} with adj_r2={adj_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training model {name}: {str(e)}")
        
        if not self.best_model_name:
            logger.warning("No models were successfully trained. Using linear_regression as fallback.")
            self.best_model_name = 'linear_regression'
        
        return {
            'best_model': self.best_model_name,
            'metrics': self.model_metrics
        }


class EnhancedGISValuationEngine(AdvancedValuationEngine):
    """Enhanced valuation engine that incorporates GIS features for improved accuracy."""
    
    def __init__(self):
        """Initialize the enhanced GIS valuation engine."""
        super().__init__()
        self.gis_features = None
    
    def set_gis_features(self, gis_features):
        """Set GIS features to use for valuation adjustments.
        
        Args:
            gis_features: dict mapping feature names to values
        """
        self.gis_features = gis_features
    
    def prepare_data(self, data):
        """Prepare data for model training and prediction, incorporating GIS features.
        
        Args:
            data: DataFrame containing property data
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        # Get base features
        X, y = super().prepare_data(data)
        
        # Add GIS features if available
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # Calculate distance-based features
            # This is a placeholder for actual GIS feature calculation
            X['near_water'] = 0  # Placeholder
            X['near_park'] = 0   # Placeholder
            X['school_quality'] = 0  # Placeholder
            
            if self.gis_features:
                # Apply GIS features from configuration
                for feature, value in self.gis_features.items():
                    if feature not in X.columns:
                        X[feature] = value
        
        return X, y
    
    def predict(self, property_data, model_name=None):
        """Predict property value using trained models with GIS adjustments.
        
        Args:
            property_data: DataFrame or dict containing property features
            model_name: Name of the model to use for prediction (optional)
                        If None, the best model will be used
        
        Returns:
            dict: Prediction results with estimated value and model metrics
        """
        # Get base prediction
        result = super().predict(property_data, model_name)
        
        # Apply GIS adjustments if available
        gis_adjustments = {}
        if isinstance(property_data, dict):
            if 'latitude' in property_data and 'longitude' in property_data:
                # Calculate neighborhood quality adjustment (placeholder)
                neighborhood_adj = 0.0
                
                # Calculate proximity adjustments (placeholder)
                proximity_adj = 0.0
                
                # Apply adjustments
                base_value = result['estimated_value']
                result['estimated_value'] = base_value * (1 + neighborhood_adj + proximity_adj)
                
                # Store adjustments
                gis_adjustments = {
                    'neighborhood_quality': neighborhood_adj,
                    'proximity_factors': proximity_adj,
                    'base_value': base_value,
                    'adjusted_value': result['estimated_value']
                }
                
                result['gis_adjustments'] = gis_adjustments
        
        return result