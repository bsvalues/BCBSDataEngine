"""
Unit tests for data validation module.
"""
import unittest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from etl.data_validation import DataValidator

class TestDataValidator(unittest.TestCase):
    """
    Test cases for the DataValidator class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock database
        self.mock_db = MagicMock()
        
        # Create validator
        self.validator = DataValidator(self.mock_db)
        
        # Create sample property data
        current_date = datetime.now()
        past_date = current_date - timedelta(days=365)
        
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'address': ['123 Main St', '456 Oak Ave', '789 Pine St', '321 Elm St', '654 Maple Ave'],
            'city': ['Anytown', 'Sometown', 'Othertown', 'Thistown', 'Thattown'],
            'state': ['CA', 'CA', 'NY', 'TX', 'FL'],
            'zip_code': ['12345', '54321', '67890', '09876', '54321'],
            'property_type': ['Single Family', 'Condo', 'Multi-Family', 'Single Family', 'Townhome'],
            'square_feet': [2000, 1200, 3500, 1800, 1500],
            'lot_size': [0.25, 0.0, 0.5, 0.3, 0.1],
            'year_built': [1990, 2005, 1975, 2000, 2015],
            'bedrooms': [3, 2, 5, 3, 2],
            'bathrooms': [2, 1, 3, 2, 2.5],
            'list_price': [500000, 300000, 800000, 450000, 350000],
            'last_sale_price': [450000, 275000, 700000, 400000, 325000],
            'last_sale_date': [past_date, past_date, past_date, past_date, past_date],
            'data_source': ['MLS', 'NARRPR', 'PACS', 'MLS', 'NARRPR'],
            'import_date': [current_date, current_date, current_date, current_date, current_date]
        })
    
    def test_validate_completeness_pass(self):
        """Test completeness validation with complete data."""
        # Call validate_completeness method
        result = self.validator.validate_completeness(self.sample_data)
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('message', result)
        self.assertIn('all_fields', result)
    
    def test_validate_completeness_fail(self):
        """Test completeness validation with incomplete data."""
        # Create data with missing values
        data = self.sample_data.copy()
        data.loc[0, 'address'] = None
        data.loc[1, 'city'] = None
        data.loc[2, 'square_feet'] = None
        data.loc[3, 'year_built'] = None
        data.loc[4, 'zip_code'] = None
        
        # Call validate_completeness method
        result = self.validator.validate_completeness(data)
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('message', result)
        self.assertIn('failed_fields', result)
        self.assertIn('all_fields', result)
    
    def test_validate_data_types_pass(self):
        """Test data type validation with correct types."""
        # Call validate_data_types method
        result = self.validator.validate_data_types(self.sample_data)
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('message', result)
    
    def test_validate_data_types_fail(self):
        """Test data type validation with incorrect types."""
        # Create data with invalid types
        data = self.sample_data.copy()
        data.loc[0, 'square_feet'] = 'Not a number'
        data.loc[1, 'list_price'] = 'Invalid'
        data.loc[2, 'year_built'] = 'Unknown'
        
        # Call validate_data_types method
        result = self.validator.validate_data_types(data)
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('message', result)
        self.assertIn('errors', result)
        self.assertGreaterEqual(len(result['errors']), 3)
    
    def test_validate_numeric_ranges_pass(self):
        """Test numeric range validation with valid ranges."""
        # Call validate_numeric_ranges method
        result = self.validator.validate_numeric_ranges(self.sample_data)
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('message', result)
    
    def test_validate_numeric_ranges_fail(self):
        """Test numeric range validation with invalid ranges."""
        # Create data with out-of-range values
        data = self.sample_data.copy()
        data.loc[0, 'square_feet'] = 50  # Too small
        data.loc[1, 'year_built'] = 1700  # Too old
        data.loc[2, 'list_price'] = 500  # Too low
        
        # Call validate_numeric_ranges method
        result = self.validator.validate_numeric_ranges(data)
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('message', result)
        self.assertIn('errors', result)
        self.assertGreaterEqual(len(result['errors']), 3)
    
    def test_validate_dates_pass(self):
        """Test date validation with valid dates."""
        # Call validate_dates method
        result = self.validator.validate_dates(self.sample_data)
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('message', result)
    
    def test_validate_dates_fail(self):
        """Test date validation with invalid dates."""
        # Create data with invalid dates
        data = self.sample_data.copy()
        future_date = datetime.now() + timedelta(days=365)
        very_old_date = datetime(1800, 1, 1)
        
        data.loc[0, 'last_sale_date'] = future_date  # Future date
        data.loc[1, 'last_sale_date'] = very_old_date  # Too old
        
        # Call validate_dates method
        result = self.validator.validate_dates(data)
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('message', result)
        self.assertIn('errors', result)
        self.assertIn('last_sale_date', result['errors'])
    
    def test_validate_duplicates_pass(self):
        """Test duplicate validation with no duplicates."""
        # Call validate_duplicates method
        result = self.validator.validate_duplicates(self.sample_data)
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('message', result)
        self.assertIn('checks', result)
    
    def test_validate_duplicates_fail(self):
        """Test duplicate validation with duplicates."""
        # Create data with duplicates
        data = pd.concat([self.sample_data, self.sample_data.iloc[0:2]])
        
        # Call validate_duplicates method
        result = self.validator.validate_duplicates(data)
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('message', result)
        self.assertIn('checks', result)
        self.assertGreater(result['checks']['exact_duplicates']['count'], 0)
    
    def test_validate_cross_source_consistency_pass(self):
        """Test cross-source consistency with consistent data."""
        # Create multi-source data that's consistent
        data = self.sample_data.copy()
        
        # Add duplicate properties from different sources
        new_row = data.iloc[0].copy()
        new_row['id'] = 6
        new_row['data_source'] = 'PACS'
        
        new_row2 = data.iloc[1].copy()
        new_row2['id'] = 7
        new_row2['data_source'] = 'MLS'
        
        data = pd.concat([data, pd.DataFrame([new_row, new_row2])])
        
        # Call validate_cross_source_consistency method
        result = self.validator.validate_cross_source_consistency(data)
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('message', result)
        self.assertIn('checks', result)
    
    def test_validate_cross_source_consistency_fail(self):
        """Test cross-source consistency with inconsistent data."""
        # Create multi-source data that's inconsistent
        data = self.sample_data.copy()
        
        # Add duplicate properties from different sources with inconsistent values
        new_row = data.iloc[0].copy()
        new_row['id'] = 6
        new_row['data_source'] = 'PACS'
        new_row['square_feet'] = 3000  # Significantly different
        new_row['year_built'] = 1980   # Different
        
        new_row2 = data.iloc[1].copy()
        new_row2['id'] = 7
        new_row2['data_source'] = 'MLS'
        new_row2['square_feet'] = 1800  # Significantly different
        
        data = pd.concat([data, pd.DataFrame([new_row, new_row2])])
        
        # Call validate_cross_source_consistency method
        result = self.validator.validate_cross_source_consistency(data)
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('message', result)
        self.assertIn('checks', result)
    
    def test_validate_all_pass(self):
        """Test comprehensive validation with valid data."""
        # Mock database to return sample data
        self.mock_db.get_all_properties.return_value = self.sample_data
        
        # Call validate_all method
        result = self.validator.validate_all()
        
        # Assertions
        self.assertEqual(result['status'], 'passed')
        self.assertIn('validations', result)
        self.assertIn('timestamp', result)
        
        # Check each validation
        for validation_name, validation_result in result['validations'].items():
            self.assertIn('status', validation_result)
            self.assertIn('message', validation_result)
    
    def test_validate_all_fail(self):
        """Test comprehensive validation with invalid data."""
        # Create data with various issues
        data = self.sample_data.copy()
        data.loc[0, 'square_feet'] = 50  # Too small
        data.loc[1, 'year_built'] = 1700  # Too old
        data.loc[2, 'list_price'] = 'Invalid'  # Invalid type
        
        # Mock database to return problematic data
        self.mock_db.get_all_properties.return_value = data
        
        # Call validate_all method
        result = self.validator.validate_all()
        
        # Assertions
        self.assertEqual(result['status'], 'failed')
        self.assertIn('validations', result)
        self.assertIn('timestamp', result)
        
        # At least one validation should fail
        failed_validations = [v for k, v in result['validations'].items() if v['status'] == 'failed']
        self.assertGreater(len(failed_validations), 0)
    
    def test_report_validation_results(self):
        """Test validation result reporting."""
        # Create validation results
        validation_results = {
            "status": "failed",
            "validations": {
                "completeness": {
                    "status": "passed",
                    "message": "All critical fields meet completeness requirements"
                },
                "data_types": {
                    "status": "failed",
                    "message": "Data type validation failed for some fields",
                    "errors": {
                        "square_feet": {
                            "expected": "numeric",
                            "error_count": 1,
                            "error_percentage": 20.0
                        }
                    }
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Call report_validation_results method
        self.validator.report_validation_results(validation_results)
        
        # Verify database method was called
        self.mock_db.store_validation_results.assert_called_once_with(validation_results)

if __name__ == '__main__':
    unittest.main()
