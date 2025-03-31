"""
Core valuation module tests.

These tests verify the functionality of the basic property valuation algorithms.
"""
import unittest
import sys
import os
from datetime import datetime

# Add the parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db
from models import Property, Valuation


class CoreValuationTestCase(unittest.TestCase):
    """Test cases for the core valuation module."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
        self.client = self.app.test_client()
        
        with self.app.app_context():
            # Create tables
            db.create_all()
            
            # Create test property
            self.test_property = Property(
                address="123 Test St",
                city="Testville",
                state="TS",
                zip_code="12345",
                property_type="single_family",
                bedrooms=3,
                bathrooms=2,
                square_feet=1500,
                lot_size=0.25,
                year_built=2000,
                latitude=47.6062,
                longitude=-122.3321,
                neighborhood="Test Neighborhood"
            )
            db.session.add(self.test_property)
            db.session.commit()
            
            # Get the property ID
            self.property_id = self.test_property.id
    
    def tearDown(self):
        """Clean up after each test."""
        with self.app.app_context():
            db.session.remove()
            db.drop_all()
    
    def test_basic_valuation_calculation(self):
        """Test that basic valuation calculation returns expected values."""
        from valuation.core import calculate_basic_valuation
        
        with self.app.app_context():
            property_obj = Property.query.get(self.property_id)
            
            # Calculate valuation
            result = calculate_basic_valuation(property_obj)
            
            # Verify the result structure
            self.assertIn('estimated_value', result)
            self.assertIn('confidence_score', result)
            
            # Verify the values are in expected ranges
            self.assertGreater(result['estimated_value'], 0)
            self.assertGreaterEqual(result['confidence_score'], 0)
            self.assertLessEqual(result['confidence_score'], 1)
            
            # Verify the calculation is based on square footage
            # For a 1500 sq ft property at $250/sq ft, we expect around $375,000
            self.assertGreaterEqual(result['estimated_value'], 300000)
            self.assertLessEqual(result['estimated_value'], 450000)
    
    def test_valuation_with_missing_data(self):
        """Test that valuation works even with missing property data."""
        from valuation.core import calculate_basic_valuation
        
        with self.app.app_context():
            # Create a property with minimal data
            minimal_property = Property(
                address="456 Minimal Ave",
                city="Testville",
                state="TS",
                zip_code="12345",
                property_type="single_family"
                # Omitting other fields
            )
            db.session.add(minimal_property)
            db.session.commit()
            
            # Calculate valuation
            result = calculate_basic_valuation(minimal_property)
            
            # Verify the result still has required structure
            self.assertIn('estimated_value', result)
            self.assertIn('confidence_score', result)
            
            # Confidence should be lower due to missing data
            self.assertLess(result['confidence_score'], 0.8)
    
    def test_valuation_with_edge_cases(self):
        """Test valuation with edge case property data."""
        from valuation.core import calculate_basic_valuation
        
        with self.app.app_context():
            # Create properties with extreme values
            tiny_property = Property(
                address="123 Tiny House",
                city="Testville",
                state="TS",
                zip_code="12345",
                property_type="single_family",
                bedrooms=1,
                bathrooms=1,
                square_feet=200,  # Very small
                lot_size=0.05,
                year_built=2022
            )
            
            mansion_property = Property(
                address="789 Mansion Ave",
                city="Testville",
                state="TS",
                zip_code="12345",
                property_type="single_family",
                bedrooms=10,
                bathrooms=12,
                square_feet=15000,  # Very large
                lot_size=5.0,
                year_built=2022
            )
            
            old_property = Property(
                address="321 Historic St",
                city="Testville",
                state="TS",
                zip_code="12345",
                property_type="single_family",
                bedrooms=3,
                bathrooms=2,
                square_feet=1800,
                lot_size=0.4,
                year_built=1890  # Very old
            )
            
            db.session.add_all([tiny_property, mansion_property, old_property])
            db.session.commit()
            
            # Test tiny property
            tiny_result = calculate_basic_valuation(tiny_property)
            self.assertGreater(tiny_result['estimated_value'], 0)
            
            # Test mansion property
            mansion_result = calculate_basic_valuation(mansion_property)
            self.assertGreater(mansion_result['estimated_value'], tiny_result['estimated_value'])
            
            # Test old property - should account for age
            old_result = calculate_basic_valuation(old_property)
            self.assertGreater(old_result['estimated_value'], 0)
    
    def test_save_valuation_to_database(self):
        """Test saving a valuation to the database."""
        from valuation.core import calculate_basic_valuation, save_valuation
        
        with self.app.app_context():
            property_obj = Property.query.get(self.property_id)
            
            # Calculate valuation
            result = calculate_basic_valuation(property_obj)
            
            # Save to database
            valuation_id = save_valuation(property_obj.id, result)
            
            # Verify the valuation was saved
            saved_valuation = Valuation.query.get(valuation_id)
            self.assertIsNotNone(saved_valuation)
            self.assertEqual(saved_valuation.property_id, property_obj.id)
            self.assertEqual(saved_valuation.estimated_value, result['estimated_value'])
            self.assertEqual(saved_valuation.confidence_score, result['confidence_score'])


if __name__ == '__main__':
    unittest.main()