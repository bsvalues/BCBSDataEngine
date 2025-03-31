"""
BCBS Values - Property Valuation Module

This module contains the core valuation logic for calculating property values
using various methods including regression models, machine learning algorithms,
and statistical analysis.
"""

import logging
import random
from datetime import datetime, timedelta
import json
import math
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logger = logging.getLogger(__name__)

# Constants for valuation calculations
BASE_PRICE_PER_SQFT = {
    'single_family': 250,
    'condo': 325,
    'townhouse': 275,
    'multi_family': 200,
    'land': 50
}

NEIGHBORHOOD_MULTIPLIERS = {
    # These would ideally come from a database or API
    'downtown': 1.5,
    'capitol hill': 1.3,
    'ballard': 1.2,
    'fremont': 1.25,
    'queen anne': 1.35,
    'west seattle': 1.1,
    'beacon hill': 0.9,
    'rainier valley': 0.85,
    'university district': 1.15,
    'northgate': 0.95
}

PROPERTY_AGE_FACTORS = {
    # New construction premium
    (0, 5): 1.1,
    # Newer homes
    (6, 10): 1.05,
    # Established homes
    (11, 20): 1.0,
    # Older homes
    (21, 40): 0.95,
    # Much older homes
    (41, 75): 0.9,
    # Historic homes (may have premium again)
    (76, 100): 0.95,
    # Very old homes
    (101, 1000): 0.85
}

def perform_valuation(property_obj, valuation_method='enhanced_regression'):
    """
    Calculate property valuation based on property characteristics and chosen method.
    
    Args:
        property_obj: A Property object containing details about the property
        valuation_method: The valuation method to use
        
    Returns:
        dict: A dictionary containing valuation results and supporting data
    """
    logger.info(f"Performing {valuation_method} valuation for property: {property_obj.address}")
    
    # Set default values
    estimated_value = 0
    confidence_score = 0
    
    try:
        # Different valuation methods
        if valuation_method == 'linear_regression':
            estimated_value, confidence_score = _linear_regression_valuation(property_obj)
        elif valuation_method == 'ridge_regression':
            estimated_value, confidence_score = _ridge_regression_valuation(property_obj)
        elif valuation_method == 'lasso_regression':
            estimated_value, confidence_score = _lasso_regression_valuation(property_obj)
        elif valuation_method == 'elastic_net':
            estimated_value, confidence_score = _elastic_net_valuation(property_obj)
        elif valuation_method == 'lightgbm':
            estimated_value, confidence_score = _lightgbm_valuation(property_obj)
        elif valuation_method == 'xgboost':
            estimated_value, confidence_score = _xgboost_valuation(property_obj)
        else:  # default to enhanced_regression
            estimated_value, confidence_score = _enhanced_regression_valuation(property_obj)
        
        # Get additional information for the valuation
        comparable_properties = _get_comparable_properties(property_obj)
        market_trends = get_market_trends(property_obj.neighborhood)
        gis_features = _get_gis_features(property_obj)
        model_features = _get_model_features(property_obj)
        
        # Round the estimated value for a cleaner display
        estimated_value = _round_to_nearest(estimated_value, 1000)
        
        return {
            'estimated_value': estimated_value,
            'confidence_score': confidence_score,
            'valuation_method': valuation_method,
            'comparable_properties': comparable_properties,
            'market_trends': market_trends,
            'gis_features': gis_features,
            'model_features': model_features
        }
        
    except Exception as e:
        logger.error(f"Error during valuation: {e}")
        raise ValueError(f"Valuation calculation failed: {str(e)}")

def _enhanced_regression_valuation(property_obj):
    """
    Advanced valuation using a combination of methods and additional factors.
    
    This method combines multiple approaches and incorporates market trends,
    GIS data, and other factors for a more comprehensive valuation.
    """
    try:
        # Calculate base price based on property type and square footage
        price_per_sqft = BASE_PRICE_PER_SQFT.get(property_obj.property_type, 200)
        
        # Start with basic calculation
        base_value = price_per_sqft * (property_obj.square_feet or 2000)
        
        # Apply neighborhood factor
        neighborhood_lower = property_obj.neighborhood.lower() if property_obj.neighborhood else ''
        neighborhood_factor = NEIGHBORHOOD_MULTIPLIERS.get(neighborhood_lower, 1.0)
        
        # Apply property age factor
        current_year = datetime.now().year
        property_age = current_year - (property_obj.year_built or current_year - 20)
        age_factor = 1.0
        
        for age_range, factor in PROPERTY_AGE_FACTORS.items():
            if age_range[0] <= property_age <= age_range[1]:
                age_factor = factor
                break
        
        # Apply bedroom/bathroom factors
        bedroom_factor = 1.0 + (0.05 * (property_obj.bedrooms or 3 - 3)) if property_obj.bedrooms else 1.0
        bathroom_factor = 1.0 + (0.05 * (property_obj.bathrooms or 2 - 2)) if property_obj.bathrooms else 1.0
        
        # Apply lot size factor
        lot_size_factor = 1.0 + (0.1 * (property_obj.lot_size or 0.25)) if property_obj.lot_size else 1.0
        
        # Calculate the enhanced value
        enhanced_value = base_value * neighborhood_factor * age_factor * bedroom_factor * bathroom_factor * lot_size_factor
        
        # Apply a small random adjustment to simulate market variance (±3%)
        variance_factor = random.uniform(0.97, 1.03)
        final_value = enhanced_value * variance_factor
        
        # Higher confidence for this method
        confidence_score = random.uniform(0.85, 0.95)
        
        return final_value, confidence_score
        
    except Exception as e:
        logger.error(f"Error in enhanced regression valuation: {e}")
        # Fallback to a simpler calculation
        return _linear_regression_valuation(property_obj)

def _linear_regression_valuation(property_obj):
    """Simple linear regression based valuation."""
    try:
        # Simple approach based on square footage
        price_per_sqft = BASE_PRICE_PER_SQFT.get(property_obj.property_type, 200)
        base_value = price_per_sqft * (property_obj.square_feet or 2000)
        
        # Apply a simple adjustment based on bedrooms and bathrooms
        bedroom_adjustment = (property_obj.bedrooms or 3) * 5000
        bathroom_adjustment = (property_obj.bathrooms or 2) * 7500
        
        final_value = base_value + bedroom_adjustment + bathroom_adjustment
        
        # Apply a random factor for simulation
        variance_factor = random.uniform(0.93, 1.07)
        final_value *= variance_factor
        
        # Lower confidence for the simple method
        confidence_score = random.uniform(0.7, 0.8)
        
        return final_value, confidence_score
        
    except Exception as e:
        logger.error(f"Error in linear regression valuation: {e}")
        # Very basic fallback
        return (property_obj.square_feet or 2000) * 200, 0.5

def _ridge_regression_valuation(property_obj):
    """Ridge regression based valuation (L2 regularization)."""
    # Simulate a ridge regression result with a variation from linear
    linear_value, _ = _linear_regression_valuation(property_obj)
    ridge_factor = random.uniform(0.95, 1.05)
    confidence = random.uniform(0.75, 0.85)
    return linear_value * ridge_factor, confidence

def _lasso_regression_valuation(property_obj):
    """Lasso regression based valuation (L1 regularization)."""
    # Simulate a lasso regression result with a variation from linear
    linear_value, _ = _linear_regression_valuation(property_obj)
    lasso_factor = random.uniform(0.93, 1.07)
    confidence = random.uniform(0.72, 0.82)
    return linear_value * lasso_factor, confidence

def _elastic_net_valuation(property_obj):
    """Elastic Net regression based valuation (L1 + L2 regularization)."""
    # Simulate an elastic net result (between ridge and lasso)
    ridge_value, _ = _ridge_regression_valuation(property_obj)
    lasso_value, _ = _lasso_regression_valuation(property_obj)
    elastic_value = (ridge_value + lasso_value) / 2
    confidence = random.uniform(0.76, 0.86)
    return elastic_value, confidence

def _lightgbm_valuation(property_obj):
    """LightGBM gradient boosting based valuation."""
    # Simulate a LightGBM model with a more significant deviation
    enhanced_value, _ = _enhanced_regression_valuation(property_obj)
    lightgbm_factor = random.uniform(0.92, 1.08)
    confidence = random.uniform(0.8, 0.9)
    return enhanced_value * lightgbm_factor, confidence

def _xgboost_valuation(property_obj):
    """XGBoost gradient boosting based valuation."""
    # Simulate an XGBoost model with a more significant deviation
    enhanced_value, _ = _enhanced_regression_valuation(property_obj)
    xgboost_factor = random.uniform(0.91, 1.09)
    confidence = random.uniform(0.82, 0.92)
    return enhanced_value * xgboost_factor, confidence

def _get_comparable_properties(property_obj):
    """
    Get comparable properties for the valuation.
    
    In a real implementation, this would query a database for similar properties
    that have recently sold in the same area.
    """
    # Create simulated comparable properties
    base_price = BASE_PRICE_PER_SQFT.get(property_obj.property_type, 200) * (property_obj.square_feet or 2000)
    comps = []
    
    # Generate 4 comparable properties
    for i in range(4):
        # Randomize the characteristics slightly
        price_variation = random.uniform(0.9, 1.1)
        sqft_variation = random.uniform(0.9, 1.1)
        bed_variation = random.randint(-1, 1)
        bath_variation = random.choice([-0.5, 0, 0.5])
        
        # Create dates in the past 6 months
        days_ago = random.randint(30, 180)
        sale_date = (datetime.now() - timedelta(days=days_ago)).strftime('%B %d, %Y')
        
        # Distance in miles (simulated)
        distance = round(random.uniform(0.2, 2.5), 1)
        
        comp = {
            'address': f"{random.randint(100, 999)} {['Oak', 'Maple', 'Pine', 'Cedar'][i]} St",
            'sale_price': round(base_price * price_variation, -3),  # Round to nearest 1000
            'sale_date': sale_date,
            'square_feet': round((property_obj.square_feet or 2000) * sqft_variation),
            'bedrooms': max(1, (property_obj.bedrooms or 3) + bed_variation),
            'bathrooms': max(1, (property_obj.bathrooms or 2) + bath_variation),
            'distance': distance
        }
        comps.append(comp)
    
    return comps

def get_market_trends(neighborhood=None):
    """
    Get market trends data for the property's area.
    
    In a real implementation, this would query historical price data
    for the market and provide trend analysis.
    """
    # Create simulated market trend data
    today = datetime.now()
    
    # Determine market direction with some randomness
    trend_direction = random.choice([-1, -1, 1, 1, 1])  # Slightly more positive
    current_trend = round(random.uniform(1.5, 4.5) * trend_direction, 1)
    
    # Determine market health
    if current_trend > 3:
        market_health = "strong"
        trend_description = "The market is showing strong positive growth with increasing property values."
    elif current_trend > 0:
        market_health = "stable"
        trend_description = "The market is stable with moderate growth in property values."
    elif current_trend > -3:
        market_health = "cooling"
        trend_description = "The market is cooling with slight decreases in property values."
    else:
        market_health = "declining"
        trend_description = "The market is showing signs of decline with decreasing property values."
    
    # Generate 12 months of historical data
    monthly_trends = []
    baseline = 100
    
    # Add some randomness to simulate real market fluctuations
    for i in range(12, 0, -1):
        month_date = today - timedelta(days=i*30)
        # Create a gradual trend with some noise
        month_change = (current_trend / 12) * (12 - i) + random.uniform(-0.5, 0.5)
        baseline += month_change
        
        monthly_trends.append({
            'date': month_date.strftime('%b %Y'),
            'value': round(baseline, 1)
        })
    
    # Generate 6 months of forecast data
    forecast = []
    for i in range(1, 7):
        month_date = today + timedelta(days=i*30)
        # Forecast with increasing uncertainty
        uncertainty = i * 0.2
        month_change = (current_trend / 6) + random.uniform(-uncertainty, uncertainty)
        baseline += month_change
        
        forecast.append({
            'date': month_date.strftime('%b %Y'),
            'value': round(baseline, 1)
        })
    
    return {
        'current_trend': current_trend,
        'market_health': market_health,
        'trend_description': trend_description,
        'monthly_trends': monthly_trends,
        'forecast': forecast
    }

def _get_gis_features(property_obj):
    """
    Get GIS (Geographic Information System) features for the property.
    
    In a real implementation, this would query GIS data sources for
    information about the property's location and surroundings.
    """
    # Create simulated GIS feature data
    # These scores would normally come from actual GIS analysis
    
    gis_features = {
        'school_quality': round(random.uniform(5.0, 9.5), 1),
        'crime_risk': round(random.uniform(2.0, 7.0), 1),
        'flood_risk': round(random.uniform(1.0, 4.0), 1),
        'walkability': round(random.uniform(3.0, 9.5), 1),
        'transit_access': round(random.uniform(2.0, 9.0), 1),
        'noise_level': round(random.uniform(3.0, 8.0), 1),
        'air_quality': round(random.uniform(5.0, 9.5), 1),
        'proximity_to_amenities': round(random.uniform(4.0, 9.0), 1),
        'parks_nearby': round(random.uniform(3.0, 9.0), 1)
    }
    
    # If we have neighborhood information, adjust scores to be more consistent
    if property_obj.neighborhood:
        neighborhood_seed = sum(ord(c) for c in property_obj.neighborhood.lower())
        random.seed(neighborhood_seed)
        
        # Apply neighborhood-specific adjustments
        for key in gis_features:
            # Adjust by up to ±1.5 but keep within 1-10 range
            adjustment = random.uniform(-1.5, 1.5)
            gis_features[key] = max(1.0, min(10.0, gis_features[key] + adjustment))
    
    # Reset the random seed
    random.seed()
    
    return gis_features

def _get_model_features(property_obj):
    """
    Get the features that were used in the valuation model.
    
    This provides transparency into what factors were considered
    in the property valuation.
    """
    # Base features (actual property characteristics)
    base_features = {
        'square_feet': property_obj.square_feet,
        'bedrooms': property_obj.bedrooms,
        'bathrooms': property_obj.bathrooms,
        'year_built': property_obj.year_built,
        'lot_size': property_obj.lot_size,
        'property_type': property_obj.property_type
    }
    
    # Filter out None values
    base_features = {k: v for k, v in base_features.items() if v is not None}
    
    # Derived features (calculated values used in the model)
    current_year = datetime.now().year
    derived_features = {
        'property_age': current_year - property_obj.year_built if property_obj.year_built else None,
        'price_per_sqft': BASE_PRICE_PER_SQFT.get(property_obj.property_type, 200),
        'neighborhood_factor': NEIGHBORHOOD_MULTIPLIERS.get(property_obj.neighborhood.lower() if property_obj.neighborhood else '', 1.0),
        'bedroom_to_sqft_ratio': round(property_obj.square_feet / property_obj.bedrooms, 1) if property_obj.square_feet and property_obj.bedrooms else None,
        'bathroom_to_bedroom_ratio': round(property_obj.bathrooms / property_obj.bedrooms, 2) if property_obj.bathrooms and property_obj.bedrooms else None
    }
    
    # Filter out None values
    derived_features = {k: v for k, v in derived_features.items() if v is not None}
    
    return {
        'base_features': base_features,
        'derived_features': derived_features
    }

def perform_what_if_analysis(property_obj, params):
    """
    Perform a what-if analysis by adjusting property parameters.
    
    Args:
        property_obj: The original property object
        params: Dictionary of parameters to adjust
        
    Returns:
        dict: Updated valuation with original and adjusted values
    """
    # Get the original valuation as baseline
    original_valuation = perform_valuation(property_obj)
    
    # Create a copy of the property with adjusted parameters
    class PropertyCopy:
        pass
    
    adjusted_property = PropertyCopy()
    
    # Copy all attributes from the original property
    for attr_name in dir(property_obj):
        # Skip special methods and functions
        if not attr_name.startswith('_') and not callable(getattr(property_obj, attr_name)):
            setattr(adjusted_property, attr_name, getattr(property_obj, attr_name))
    
    # Apply the parameter adjustments
    for param_name, param_value in params.items():
        if hasattr(adjusted_property, param_name):
            setattr(adjusted_property, param_name, param_value)
    
    # Get the adjusted valuation
    adjusted_valuation = perform_valuation(adjusted_property)
    
    # Calculate impact of each parameter
    parameter_impacts = {}
    for param_name, param_value in params.items():
        # Create a property with only this parameter changed
        single_param_property = PropertyCopy()
        for attr_name in dir(property_obj):
            if not attr_name.startswith('_') and not callable(getattr(property_obj, attr_name)):
                setattr(single_param_property, attr_name, getattr(property_obj, attr_name))
        
        if hasattr(single_param_property, param_name):
            setattr(single_param_property, param_name, param_value)
            
        # Get valuation with just this parameter changed
        single_param_valuation = perform_valuation(single_param_property)
        
        # Calculate the impact
        impact = single_param_valuation['estimated_value'] - original_valuation['estimated_value']
        impact_percent = (impact / original_valuation['estimated_value']) * 100
        
        parameter_impacts[param_name] = {
            'original_value': getattr(property_obj, param_name),
            'adjusted_value': param_value,
            'impact_value': impact,
            'impact_percent': round(impact_percent, 2)
        }
    
    return {
        'original_valuation': original_valuation,
        'adjusted_valuation': adjusted_valuation,
        'parameter_impacts': parameter_impacts,
        'total_impact_value': adjusted_valuation['estimated_value'] - original_valuation['estimated_value'],
        'total_impact_percent': round(((adjusted_valuation['estimated_value'] - original_valuation['estimated_value']) / original_valuation['estimated_value']) * 100, 2)
    }

def _round_to_nearest(value, nearest=1000):
    """Round a value to the nearest specified amount."""
    if value is None:
        return None
        
    # Use Decimal for accurate rounding
    d_value = Decimal(str(value))
    d_nearest = Decimal(str(nearest))
    
    return float((d_value / d_nearest).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_nearest)