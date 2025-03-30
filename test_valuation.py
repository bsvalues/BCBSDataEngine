"""
Test script to verify enhanced valuation functionality.
"""

import pandas as pd
import logging
from src.valuation import ValuationEngine, AdvancedValuationEngine, GISFeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_valuation_classes():
    """Test the functionality of valuation classes."""
    print("Testing valuation classes...")
    
    # Create test property data
    property_data = pd.DataFrame({
        'property_id': ['BENTON-12345'],
        'address': ['123 Main St, Kennewick, WA'],
        'bedrooms': [3],
        'bathrooms': [2.5],
        'square_footage': [2200],
        'lot_size': [0.25],
        'year_built': [1995],
        'price': [350000],
        'latitude': [46.2122],
        'longitude': [-119.1372]
    })
    
    # Test basic valuation engine
    print("\nTesting ValuationEngine...")
    basic_engine = ValuationEngine()
    basic_result = basic_engine.calculate_valuation('BENTON-12345')
    print(f"Basic valuation result: {basic_result}")
    
    # Test advanced valuation engine
    print("\nTesting AdvancedValuationEngine...")
    advanced_engine = AdvancedValuationEngine()
    advanced_result = advanced_engine.calculate_valuation(
        'BENTON-12345',
        latitude=46.2122, 
        longitude=-119.1372,
        neighborhood_quality=0.85,
        include_model_metrics=True
    )
    print(f"Advanced valuation result: {advanced_result}")
    
    # Test GIS feature engine
    print("\nTesting GISFeatureEngine...")
    gis_engine = GISFeatureEngine()
    adjustment = gis_engine.calculate_gis_adjustment('BENTON-12345')
    print(f"GIS adjustment factor: {adjustment}")
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    test_valuation_classes()