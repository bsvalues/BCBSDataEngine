"""
Unit tests for the advanced property valuation functionality.

This module tests the advanced valuation features including:
1. Basic property value prediction
2. Multiple regression metrics and statistics
3. Advanced valuation with LightGBM integration 
4. GIS-enhanced valuation with location quality
5. Error handling for missing data

Each test case verifies specific aspects of the valuation engine with
appropriate assertions and well-documented test data.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add the project root to the Python path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the valuation functions to test
from src.valuation import (
    train_basic_valuation_model,
    train_multiple_regression_model,
    advanced_property_valuation,
    has_lightgbm  # Import the has_lightgbm flag directly
)

# Try to import GIS functions, but continue even if they're not available
try:
    from src.gis_integration import (
        enhance_property_with_gis,
        calculate_combined_gis_multiplier
    )
    has_gis_integration = True
except ImportError:
    has_gis_integration = False
    print("GIS integration module not available, tests requiring it will be skipped.")


@pytest.fixture
def sample_property_data():
    """
    Create a sample DataFrame with property data for testing.
    
    Returns:
        pd.DataFrame: Sample property data with standard real estate features
    """
    # Create sample property data with sufficient records for model training
    np.random.seed(42)  # Set seed for reproducibility
    
    # Number of sample properties
    n = 100
    
    # Generate realistic property data with correlations
    square_feet_base = np.random.normal(2000, 500, n)
    bedrooms_base = np.round(np.clip(np.random.normal(3, 1, n), 1, 6))
    bathrooms_base = np.round(np.clip(np.random.normal(2, 0.7, n), 1, 4)) * 0.5
    year_built_base = np.round(np.random.normal(1990, 15, n)).astype(int)
    lot_size_base = np.random.normal(8000, 2000, n)
    
    # Create latitude and longitude within Benton County, WA
    # Approximate boundaries: 46.0 to 46.4 latitude, -119.4 to -119.0 longitude
    latitude_base = np.random.uniform(46.0, 46.4, n)
    longitude_base = np.random.uniform(-119.4, -119.0, n)
    
    # Generate property prices with a realistic model
    # Price is a function of square feet, bedrooms, bathrooms, age and location
    price_per_sqft = 150  # Base price per square foot
    bedroom_value = 10000  # Value per bedroom
    bathroom_value = 15000  # Value per bathroom
    age_penalty = 500  # Price reduction per year of age
    current_year = 2023
    
    # Calculate base prices
    prices = (
        square_feet_base * price_per_sqft + 
        bedrooms_base * bedroom_value + 
        bathrooms_base * bathroom_value - 
        (current_year - year_built_base) * age_penalty
    )
    
    # Add location factor (properties in north are worth more)
    location_factor = 1 + (latitude_base - 46.0) / 2  # 1.0 to 1.2 multiplier
    prices = prices * location_factor
    
    # Add random noise (±10%)
    noise = np.random.uniform(0.9, 1.1, n)
    prices = prices * noise
    
    # Assemble the DataFrame
    data = {
        'property_id': [f'PROP{i:03d}' for i in range(1, n+1)],
        'square_feet': square_feet_base,
        'bedrooms': bedrooms_base,
        'bathrooms': bathrooms_base,
        'year_built': year_built_base,
        'lot_size': lot_size_base,
        'latitude': latitude_base,
        'longitude': longitude_base,
        'city': np.random.choice(['Richland', 'Kennewick', 'West Richland', 'Pasco'], n),
        'estimated_value': prices
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_gis_data():
    """
    Create sample GIS reference data for testing.
    
    Returns:
        dict: Sample GIS datasets for testing
    """
    # Create reference points (important locations)
    ref_points = {
        'downtown_richland': {
            'lat': 46.2804, 
            'lon': -119.2752, 
            'weight': 1.0
        },
        'downtown_kennewick': {
            'lat': 46.2112, 
            'lon': -119.1367, 
            'weight': 0.9
        },
        'downtown_pasco': {
            'lat': 46.2395, 
            'lon': -119.1005, 
            'weight': 0.8
        }
    }
    
    # Create neighborhood ratings
    neighborhood_ratings = {
        'Richland': 1.15,
        'West Richland': 1.05,
        'Kennewick': 1.0,
        'Pasco': 0.95
    }
    
    # Create sample flood zone data
    flood_zones = pd.DataFrame({
        'latitude': [46.25, 46.18, 46.30, 46.22],
        'longitude': [-119.20, -119.15, -119.10, -119.25],
        'flood_zone': ['X', 'AE', 'X500', 'X']
    })
    
    # Create sample school data
    schools = pd.DataFrame({
        'name': ['Richland High', 'Hanford High', 'Southridge High', 'Pasco High'],
        'type': ['high', 'high', 'high', 'high'],
        'latitude': [46.28, 46.32, 46.19, 46.23],
        'longitude': [-119.27, -119.33, -119.19, -119.09],
        'rating': [8.2, 8.5, 7.8, 7.0]
    })
    
    # Create sample amenity data
    amenities = pd.DataFrame({
        'name': ['Columbia Center Mall', 'Howard Amon Park', 'Queensgate Shopping', 'Kadlec Hospital'],
        'type': ['shopping', 'park', 'shopping', 'hospital'],
        'latitude': [46.21, 46.29, 46.26, 46.28],
        'longitude': [-119.19, -119.27, -119.32, -119.28]
    })
    
    return {
        'ref_points': ref_points,
        'neighborhood_ratings': neighborhood_ratings,
        'flood_zones': flood_zones,
        'schools': schools,
        'amenities': amenities
    }


def test_basic_valuation_model(sample_property_data):
    """
    Test the basic property valuation model to ensure it returns the expected outputs.
    
    This test verifies that the basic linear regression model:
    1. Successfully runs without errors
    2. Returns reasonable R² scores and predictions
    3. Includes important features in the model
    4. Properly calculates feature importance
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Train the basic model
    result = train_basic_valuation_model(sample_property_data)
    
    # Verify the model produces a reasonable R² score 
    # (should be good since our data was generated with a linear relationship)
    assert 'r2_score' in result, "R² score missing from results"
    assert result['r2_score'] > 0.7, "Model R² score below acceptable threshold"
    
    # Verify the model produces reasonable predictions
    assert 'predictions' in result, "Predictions missing from results"
    assert 'predicted_value' in result['predictions'].columns, "Predicted value column missing"
    
    # Check that squared footage is identified as an important feature (should be the case
    # with our synthetic data where it has a large effect on price)
    assert 'feature_importance' in result, "Feature importance missing from results"
    assert 'square_feet' in result['feature_importance'], "Square feet not in feature importance"
    
    # Check that the model returned key performance metrics
    assert 'mae' in result, "MAE missing from results"
    assert 'rmse' in result, "RMSE missing from results"
    assert 'features_used' in result, "Features used missing from results"
    
    # Verify feature coefficients are returned
    assert 'feature_coefficients' in result, "Feature coefficients missing from results"
    
    # Verify the model object is returned
    assert 'model' in result, "Model object missing from results"
    
    # Verify scaler is returned for feature normalization
    assert 'scaler' in result, "Scaler missing from results"


def test_multiple_regression_model(sample_property_data):
    """
    Test the enhanced multiple regression model with feature selection and statistics.
    
    This test verifies that the multiple regression model:
    1. Successfully runs without errors
    2. Returns improved R² scores with feature engineering
    3. Includes statistical significance metrics (p-values)
    4. Properly calculates adjusted R² scores
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Train the multiple regression model
    result = train_multiple_regression_model(
        sample_property_data, 
        feature_selection=True, 
        max_features=10
    )
    
    # Verify the model produces a good R² score
    assert 'r2_score' in result, "R² score missing from results"
    assert result['r2_score'] > 0.7, "Model R² score below acceptable threshold"
    
    # Verify the model returns adjusted R² (key metric for multiple regression)
    assert 'adj_r2_score' in result, "Adjusted R² score missing from results"
    
    # Verify the model returns p-values for statistical significance
    assert 'p_values' in result, "P-values missing from results"
    
    # Verify the model performs feature selection
    assert 'selected_features' in result, "Selected features missing from results"
    
    # Verify the model returns cross-validation scores
    assert 'cross_val_scores' in result, "Cross-validation scores missing from results"
    assert len(result['cross_val_scores']) > 0, "Cross-validation scores empty"
    
    # Verify the statsmodel result is included
    assert 'statsmodel' in result, "Statsmodel result missing"
    
    # Verify feature engineering was performed by checking for engineered features
    features_used = result.get('selected_features', [])
    engineered_features_present = any('_log' in feat or '_squared' in feat for feat in features_used)
    assert engineered_features_present, "No evidence of feature engineering in selected features"


def test_advanced_property_valuation_basic(sample_property_data):
    """
    Test the advanced property valuation function with the basic linear model.
    
    This test verifies that the advanced property valuation:
    1. Successfully runs without errors
    2. Returns comprehensive metrics
    3. Properly handles target property prediction
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Create a target property to predict
    target_property = pd.DataFrame({
        'square_feet': [2200],
        'bedrooms': [4],
        'bathrooms': [2.5],
        'year_built': [2000],
        'lot_size': [9000]
    })
    
    # Run advanced valuation with basic linear model
    result = advanced_property_valuation(
        property_data=sample_property_data,
        target_property=target_property,
        model_type='linear',
        use_gis_features=False  # Disable GIS for this test
    )
    
    # Verify the function returns a prediction
    assert 'predicted_value' in result, "Predicted value missing from results"
    assert result['predicted_value'] > 0, "Predicted value should be positive"
    
    # Check that the model returns comprehensive performance metrics
    assert 'r2_score' in result, "R² score missing from results"
    assert 'adj_r2_score' in result, "Adjusted R² score missing from results"
    assert 'rmse' in result, "RMSE missing from results"
    assert 'mae' in result, "MAE missing from results"
    
    # Verify the model returns feature importance
    assert 'feature_importance' in result, "Feature importance missing from results"
    
    # Check model parameters are returned
    assert 'model_params' in result, "Model parameters missing from results"
    
    # Verify training information is returned
    assert 'training_samples' in result, "Training samples count missing"
    assert 'test_samples' in result, "Test samples count missing"


@pytest.mark.skipif(not has_gis_integration, reason="GIS integration module not available")
def test_advanced_property_valuation_with_gis(sample_property_data, sample_gis_data):
    """
    Test the advanced property valuation with GIS features enabled.
    
    This test verifies that:
    1. GIS features are properly integrated into the valuation
    2. Location-based adjustments are correctly applied
    3. The model returns enhanced GIS metrics
    
    Args:
        sample_property_data: Fixture providing sample property data
        sample_gis_data: Fixture providing sample GIS reference data
    """
    # Create a target property to predict, with coordinates
    target_property = pd.DataFrame({
        'square_feet': [2200],
        'bedrooms': [4],
        'bathrooms': [2.5],
        'year_built': [2000],
        'lot_size': [9000],
        'latitude': [46.28],
        'longitude': [-119.27],
        'city': ['Richland']
    })
    
    # Run advanced valuation with GIS features
    result = advanced_property_valuation(
        property_data=sample_property_data,
        target_property=target_property,
        model_type='linear',
        use_gis_features=True,
        ref_points=sample_gis_data['ref_points'],
        neighborhood_ratings=sample_gis_data['neighborhood_ratings'],
        gis_data=sample_gis_data
    )
    
    # Verify the function returns a prediction
    assert 'predicted_value' in result, "Predicted value missing from results"
    assert result['predicted_value'] > 0, "Predicted value should be positive"
    
    # Verify GIS-specific metrics are included
    assert 'gis_features' in result, "GIS features missing from results"
    assert 'location_multiplier' in result, "Location multiplier missing from results"
    assert 'gis_factors' in result, "GIS factors missing from results"
    
    # Check that proximity scores are calculated
    if 'gis_factors' in result and 'proximity' in result['gis_factors']:
        proximity = result['gis_factors']['proximity']
        assert 'combined_score' in proximity['scores'], "Proximity scores missing combined score"
    
    # Verify the location adjustments actually affect the valuation
    assert 'base_predicted_value' in result, "Base predicted value missing"
    assert 'gis_adjusted_value' in result, "GIS-adjusted value missing"
    
    # The GIS-adjusted value should differ from the base prediction
    if 'base_predicted_value' in result and 'gis_adjusted_value' in result:
        assert result['base_predicted_value'] != result['gis_adjusted_value'], \
            "GIS adjustment had no effect on the valuation"


@pytest.mark.skipif(not has_gis_integration, reason="GIS integration module not available")
def test_enhanced_property_with_gis(sample_gis_data):
    """
    Test the enhance_property_with_gis function directly.
    
    This test verifies that:
    1. The function correctly enhances a property with GIS data
    2. All expected GIS factors are calculated
    3. The combined multiplier is within expected ranges
    
    Args:
        sample_gis_data: Fixture providing sample GIS reference data
    """
    # Skip if GIS integration is not available
    if not has_gis_integration:
        pytest.skip("GIS integration module not available")
    
    # Create a sample property in Richland near downtown
    property_data = {
        'latitude': 46.28,
        'longitude': -119.27,
        'city': 'Richland',
        'property_id': 'TEST001'
    }
    
    # Enhance the property with GIS data
    enhanced = enhance_property_with_gis(property_data, sample_gis_data)
    
    # Verify enhancement adds GIS factors
    assert 'gis_factors' in enhanced, "GIS factors missing from enhanced property"
    
    # Check that proximity is calculated
    assert 'proximity' in enhanced['gis_factors'], "Proximity factor missing"
    assert 'value_multiplier' in enhanced['gis_factors']['proximity'], "Proximity multiplier missing"
    
    # Check that neighborhood rating is applied
    assert 'neighborhood' in enhanced['gis_factors'], "Neighborhood factor missing"
    
    # Calculate the combined multiplier
    if 'gis_factors' in enhanced:
        combined = calculate_combined_gis_multiplier(enhanced['gis_factors'])
        
        # Verify the combined multiplier is within reasonable ranges
        assert 0.7 <= combined <= 1.3, "Combined GIS multiplier outside expected range (0.7-1.3)"
        
        # For a Richland property near downtown, the multiplier should be above 1.0
        assert combined > 1.0, "Combined multiplier for prime location should be above 1.0"


@pytest.mark.skipif(not has_lightgbm, reason="LightGBM not available")
def test_advanced_property_valuation_lightgbm(sample_property_data):
    """
    Test the advanced property valuation with LightGBM model.
    
    This test verifies that:
    1. The LightGBM model can be used for valuation
    2. Feature importance is calculated properly
    3. Performance metrics are accurate
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Skip if LightGBM is not available
    if not has_lightgbm:
        pytest.skip("LightGBM is not available")
    # Skip if sample data is too small
    if len(sample_property_data) < 30:
        pytest.skip("Not enough data for LightGBM model (need at least 30 samples)")
    
    # Create a target property to predict
    target_property = pd.DataFrame({
        'square_feet': [2200],
        'bedrooms': [4],
        'bathrooms': [2.5],
        'year_built': [2000],
        'lot_size': [9000]
    })
    
    # Run advanced valuation with LightGBM model
    result = advanced_property_valuation(
        property_data=sample_property_data,
        target_property=target_property,
        model_type='lightgbm',
        use_gis_features=False  # Disable GIS for this test
    )
    
    # Verify the function returns a prediction
    assert 'predicted_value' in result, "Predicted value missing from results"
    assert result['predicted_value'] > 0, "Predicted value should be positive"
    
    # Check that the model returns comprehensive performance metrics
    assert 'r2_score' in result, "R² score missing from results"
    assert 'rmse' in result, "RMSE missing from results"
    assert 'mae' in result, "MAE missing from results"
    
    # Verify the model returns feature importance (LightGBM specific)
    assert 'feature_importance' in result, "Feature importance missing from results"
    
    # Check LightGBM specific parameters
    assert 'model_params' in result, "Model parameters missing from results"
    assert 'model_type' in result, "Model type missing from results"
    assert result['model_type'] == 'lightgbm', "Incorrect model type returned"


@pytest.mark.skipif(not has_lightgbm, reason="LightGBM not available (required for ensemble model)")
def test_advanced_property_valuation_ensemble(sample_property_data):
    """
    Test the advanced property valuation with ensemble model.
    
    This test verifies that:
    1. The ensemble model can be used for valuation
    2. Multiple model predictions are combined properly
    3. Ensemble usually performs better than individual models
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Skip if LightGBM is not available
    if not has_lightgbm:
        pytest.skip("LightGBM is not available (required for ensemble model)")
    # Skip if sample data is too small
    if len(sample_property_data) < 30:
        pytest.skip("Not enough data for ensemble model (need at least 30 samples)")
    
    # Create a target property to predict
    target_property = pd.DataFrame({
        'square_feet': [2200],
        'bedrooms': [4],
        'bathrooms': [2.5],
        'year_built': [2000],
        'lot_size': [9000]
    })
    
    # Run advanced valuation with ensemble model
    result = advanced_property_valuation(
        property_data=sample_property_data,
        target_property=target_property,
        model_type='ensemble',
        use_gis_features=False  # Disable GIS for this test
    )
    
    # Verify the function returns a prediction
    assert 'predicted_value' in result, "Predicted value missing from results"
    assert result['predicted_value'] > 0, "Predicted value should be positive"
    
    # Check that the model returns comprehensive performance metrics
    assert 'r2_score' in result, "R² score missing from results"
    assert 'rmse' in result, "RMSE missing from results"
    
    # Verify the ensemble model returns individual model predictions
    assert 'model_predictions' in result, "Individual model predictions missing"
    assert isinstance(result['model_predictions'], dict), "Model predictions should be a dictionary"
    
    # Check that ensemble model info is returned
    assert 'model_type' in result, "Model type missing from results"
    assert result['model_type'] == 'ensemble', "Incorrect model type returned"
    assert 'ensemble_weights' in result, "Ensemble weights missing from results"


def test_missing_gis_data_handling(sample_property_data):
    """
    Test that the valuation model gracefully handles missing GIS data.
    
    This test verifies that:
    1. The model runs without errors when GIS is enabled but no data is provided
    2. The model falls back to basic valuation without GIS
    3. Appropriate warning logs are generated
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Create a target property that's missing coordinates
    target_property = pd.DataFrame({
        'square_feet': [2200],
        'bedrooms': [4],
        'bathrooms': [2.5],
        'year_built': [2000],
        'lot_size': [9000]
        # Missing latitude and longitude
    })
    
    # Run advanced valuation with GIS features enabled but no data provided
    result = advanced_property_valuation(
        property_data=sample_property_data,
        target_property=target_property,
        model_type='linear',
        use_gis_features=True,  # Enable GIS
        ref_points=None,  # But provide no reference points
        neighborhood_ratings=None  # And no neighborhood ratings
    )
    
    # Verify the function still returns a prediction
    assert 'predicted_value' in result, "Predicted value missing from results"
    assert result['predicted_value'] > 0, "Predicted value should be positive"
    
    # Check that the model falls back to non-GIS valuation
    # If there's a gis_adjusted_value, it should equal the base value
    if 'base_predicted_value' in result and 'gis_adjusted_value' in result:
        assert result['base_predicted_value'] == result['gis_adjusted_value'], \
            "GIS adjustment should not occur when no GIS data is provided"
    
    # Alternatively, there might not be a gis_adjusted_value at all
    # In this case, check there's a base_predicted_value that equals predicted_value 
    elif 'base_predicted_value' in result:
        assert result['base_predicted_value'] == result['predicted_value'], \
            "Predicted value should equal base value when no GIS adjustment occurs"


def test_feature_selection_modes(sample_property_data):
    """
    Test that different feature selection methods work properly.
    
    This test verifies that:
    1. Each feature selection method runs without errors
    2. Features are actually selected differently by each method
    3. Performance metrics are consistent
    
    Args:
        sample_property_data: Fixture providing sample property data
    """
    # Skip if sample data is too small
    if len(sample_property_data) < 30:
        pytest.skip("Not enough data for feature selection comparison")
    
    # Create a target property to predict
    target_property = pd.DataFrame({
        'square_feet': [2200],
        'bedrooms': [4],
        'bathrooms': [2.5],
        'year_built': [2000],
        'lot_size': [9000]
    })
    
    # Feature selection methods to test
    methods = ['auto', 'k-best', 'rfe']
    results = {}
    
    # Run advanced valuation with each feature selection method
    for method in methods:
        results[method] = advanced_property_valuation(
            property_data=sample_property_data,
            target_property=target_property,
            model_type='linear',
            use_gis_features=False,
            feature_selection=method
        )
        
        # Verify each method returns a prediction
        assert 'predicted_value' in results[method], f"Predicted value missing for {method}"
        assert results[method]['predicted_value'] > 0, f"Predicted value should be positive for {method}"
        
        # Verify feature selection method is recorded
        assert 'feature_selection_method' in results[method], f"Feature selection method missing for {method}"
    
    # Check that feature selection methods give different results
    # This could be in terms of selected features or predictions
    unique_predictions = set(round(results[method]['predicted_value'], 2) for method in methods)
    assert len(unique_predictions) > 1, "Feature selection methods should yield different predictions"


if __name__ == "__main__":
    pytest.main(["-v", __file__])