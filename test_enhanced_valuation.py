#!/usr/bin/env python3
"""
Test script for the enhanced property valuation module.

This script demonstrates the enhanced property valuation functionality
with the new advanced regression models, GIS integration, and model comparison.
"""

import logging
import json
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('valuation_test')

# Create a property class for testing
class Property:
    """Test property object that mimics the properties of the real Property model."""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.address = kwargs.get('address', '123 Test Street')
        self.city = kwargs.get('city', 'Seattle')
        self.state = kwargs.get('state', 'WA')
        self.zip_code = kwargs.get('zip_code', '98101')
        self.neighborhood = kwargs.get('neighborhood', 'downtown')
        self.property_type = kwargs.get('property_type', 'single_family')
        self.bedrooms = kwargs.get('bedrooms', 3)
        self.bathrooms = kwargs.get('bathrooms', 2.5)
        self.square_feet = kwargs.get('square_feet', 2000)
        self.year_built = kwargs.get('year_built', 1985)
        self.lot_size = kwargs.get('lot_size', 0.25)  # In acres
        self.latitude = kwargs.get('latitude', 47.6062)
        self.longitude = kwargs.get('longitude', -122.3321)
        self.description = kwargs.get('description', 'Test property for valuation module')


def test_valuation_methods():
    """Test all valuation methods and compare results."""
    logger.info("Testing all valuation methods...")
    
    # Import the valuation module
    from src.valuation import perform_valuation
    
    # Create a test property
    test_property = Property()
    
    # List of valuation methods to test
    valuation_methods = [
        'enhanced_regression',
        'linear_regression',
        'ridge_regression',
        'lasso_regression',
        'elastic_net',
        'lightgbm',
        'xgboost',
        'advanced_lightgbm',
        'advanced_linear',
        'auto'  # Automatic method comparison
    ]
    
    results = {}
    
    for method in valuation_methods:
        logger.info(f"Testing valuation method: {method}")
        
        result = perform_valuation(test_property, valuation_method=method)
        results[method] = {
            'estimated_value': result['estimated_value'],
            'confidence_score': result['confidence_score'],
            'valuation_method': result['valuation_method']
        }
        
        # Print the valuation and confidence
        logger.info(f"{method} valuation: ${result['estimated_value']:,.2f} " +
                   f"(confidence: {result['confidence_score']:.2f})")
        
        # For advanced methods, print the performance metrics
        if 'performance_metrics' in result and result['performance_metrics']:
            if 'r_squared' in result['performance_metrics']:
                logger.info(f"R-squared: {result['performance_metrics']['r_squared']:.3f}")
            
            # Print model comparison results for the 'auto' method
            if method == 'auto' and 'model_comparison' in result['performance_metrics']:
                model_comp = result['performance_metrics']['model_comparison']
                logger.info(f"Models evaluated: {', '.join(model_comp['models_evaluated'])}")
                logger.info(f"Selected model: {model_comp['selected_model']}")
                logger.info("Model values:")
                for model, value in model_comp['model_values'].items():
                    logger.info(f"  {model}: ${value:,.2f}")
    
    # Print a summary comparison
    logger.info("\nSUMMARY COMPARISON OF VALUATION METHODS")
    logger.info("=" * 60)
    logger.info(f"{'Method':<20} {'Estimated Value':<20} {'Confidence':<10}")
    logger.info("-" * 60)
    
    for method, result in results.items():
        logger.info(f"{method:<20} ${result['estimated_value']:,.2f} {result['confidence_score']:.2f}")
    
    # Return results dictionary
    return results


def test_gis_integration():
    """Test GIS integration for property valuation."""
    logger.info("\nTesting GIS integration for property valuation...")
    
    # Import the valuation and GIS modules
    from src.valuation import perform_valuation
    from src.gis_integration import get_location_score, get_school_district_info, get_flood_risk_assessment
    
    # Create properties with different locations
    properties = [
        Property(
            id=1,
            address='123 Downtown Street',
            neighborhood='downtown',
            latitude=47.6062,
            longitude=-122.3321
        ),
        Property(
            id=2,
            address='456 Queen Anne Ave',
            neighborhood='queen anne',
            latitude=47.6370,
            longitude=-122.3570
        ),
        Property(
            id=3,
            address='789 Beacon Hill Road',
            neighborhood='beacon hill',
            latitude=47.5867,
            longitude=-122.3138
        )
    ]
    
    for prop in properties:
        logger.info(f"\nProperty: {prop.address} ({prop.neighborhood})")
        
        # Get location score
        location_data = get_location_score(prop.latitude, prop.longitude)
        logger.info(f"Location score: {location_data.get('score')}")
        
        # Get school district info
        school_data = get_school_district_info(prop.latitude, prop.longitude)
        logger.info(f"School district: {school_data.get('district_name')}")
        logger.info(f"School rating: {school_data.get('overall_rating')}")
        
        # Get flood risk
        flood_data = get_flood_risk_assessment(prop.latitude, prop.longitude)
        logger.info(f"Flood risk: {flood_data.get('risk_level')} " +
                   f"({flood_data.get('risk_factor', 0):.1f})")
        
        # Perform valuation with GIS integration
        valuation = perform_valuation(prop, valuation_method='advanced_lightgbm')
        logger.info(f"Valuation with GIS: ${valuation['estimated_value']:,.2f}")
        
        # Check if spatial adjustments were applied
        if 'performance_metrics' in valuation and 'spatial_adjustment' in valuation['performance_metrics']:
            spatial = valuation['performance_metrics']['spatial_adjustment']
            if spatial.get('spatial_adjustment_applied', False):
                logger.info(f"Spatial adjustment factor: {spatial.get('spatial_adjustment_factor', 1.0)}")
                logger.info(f"Location impact: {spatial.get('percentage_impact', 0)}%")


def test_property_modifications():
    """Test how property modifications affect valuation."""
    logger.info("\nTesting property modifications and their impact on valuation...")
    
    # Import the valuation module
    from src.valuation import perform_valuation, perform_what_if_analysis
    
    # Create a base property
    base_property = Property(
        square_feet=2000,
        bedrooms=3,
        bathrooms=2,
        year_built=1985,
        lot_size=0.25,
        neighborhood='ballard'
    )
    
    # Get base valuation
    base_valuation = perform_valuation(base_property, valuation_method='advanced_lightgbm')
    base_value = base_valuation['estimated_value']
    logger.info(f"Base property valuation: ${base_value:,.2f}")
    
    # Test what-if analysis for different modifications
    modifications = [
        # Test adding square footage
        {'square_feet': 2500},
        # Test adding a bedroom
        {'bedrooms': 4},
        # Test adding a bathroom
        {'bathrooms': 3},
        # Test newer construction
        {'year_built': 2005},
        # Test larger lot
        {'lot_size': 0.5},
        # Test premium neighborhood
        {'neighborhood': 'queen anne'},
        # Test multiple upgrades
        {'square_feet': 2500, 'bathrooms': 3, 'year_built': 2000}
    ]
    
    for mod in modifications:
        # Create description of modifications
        mod_desc = ", ".join([f"{k}={v}" for k, v in mod.items()])
        logger.info(f"\nTesting modification: {mod_desc}")
        
        # Use what-if analysis to calculate the impact
        result = perform_what_if_analysis(base_property, mod)
        
        adjusted_value = result['adjusted_valuation']['estimated_value']
        value_change = result['total_impact_value']
        percent_change = result['total_impact_percent']
        
        logger.info(f"Adjusted value: ${adjusted_value:,.2f}")
        logger.info(f"Value change: ${value_change:,.2f} ({percent_change:+.1f}%)")
        
        # Print impact of individual parameters if multiple were changed
        if len(mod) > 1 and 'parameter_impacts' in result:
            logger.info("Individual parameter impacts:")
            for param, impact in result['parameter_impacts'].items():
                logger.info(f"  {param}: ${impact['impact_value']:,.2f} ({impact['impact_percent']:+.1f}%)")


def main():
    """Main function to run the tests."""
    logger.info("Starting enhanced valuation tests\n" + "=" * 50)
    
    try:
        # Test different valuation methods
        test_valuation_methods()
        
        # Test GIS integration for valuation
        test_gis_integration()
        
        # Test property modifications 
        test_property_modifications()
        
        logger.info("\nAll tests completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())