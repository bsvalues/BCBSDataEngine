"""
Unit tests for MLS scraper module.
"""
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from etl.mls_scraper import MLSScraper

class TestMLSScraper(unittest.TestCase):
    """
    Test cases for the MLSScraper class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = MLSScraper(batch_size=10)
        
        # Mock property data
        self.mock_property_data = {
            "properties": [
                {
                    "mlsId": "MLS123",
                    "listingId": "L123",
                    "propertyType": "Single Family",
                    "address": "123 Main St",
                    "city": "Anytown",
                    "state": "CA",
                    "zipCode": "12345",
                    "price": 500000,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "squareFeet": 2000,
                    "yearBuilt": 1990,
                    "listingDate": "2023-01-15",
                    "status": "Active"
                },
                {
                    "mlsId": "MLS456",
                    "listingId": "L456",
                    "propertyType": "Condo",
                    "address": "456 Oak Ave",
                    "city": "Sometown",
                    "state": "CA",
                    "zipCode": "54321",
                    "price": 300000,
                    "bedrooms": 2,
                    "bathrooms": 1,
                    "squareFeet": 1200,
                    "yearBuilt": 2005,
                    "listingDate": "2023-02-01",
                    "status": "Pending"
                }
            ]
        }
    
    @patch('etl.mls_scraper.requests.get')
    def test_extract_success(self, mock_get):
        """Test successful data extraction."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_property_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call extract method
        result = self.scraper.extract()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['mlsId'], 'MLS123')
        self.assertEqual(result.iloc[1]['mlsId'], 'MLS456')
        
        # Verify the API was called with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue(args[0].endswith('/properties'))
        self.assertEqual(kwargs['params']['limit'], 10)
    
    @patch('etl.mls_scraper.requests.get')
    def test_extract_empty_response(self, mock_get):
        """Test extraction with empty response."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"properties": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call extract method
        result = self.scraper.extract()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    @patch('etl.mls_scraper.requests.get')
    def test_extract_api_error(self, mock_get):
        """Test extraction with API error."""
        # Configure mock to raise exception
        mock_get.side_effect = Exception("API Error")
        
        # Call extract method and check for exception
        with self.assertRaises(Exception):
            self.scraper.extract()
    
    def test_transform_data(self):
        """Test data transformation."""
        # Create input data
        input_data = pd.DataFrame(self.mock_property_data['properties'])
        
        # Call transform method
        result = self.scraper._transform_data(input_data)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Check column renaming
        self.assertIn('mls_id', result.columns)
        self.assertIn('listing_id', result.columns)
        self.assertIn('property_type', result.columns)
        self.assertIn('address', result.columns)
        
        # Check data source and import date
        self.assertEqual(result.iloc[0]['data_source'], 'MLS')
        self.assertIsNotNone(result.iloc[0]['import_date'])
        
        # Check data type conversions
        self.assertTrue(pd.api.types.is_numeric_dtype(result['list_price']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['square_feet']))
    
    def test_load_data(self):
        """Test data loading."""
        # Create mock database
        mock_db = MagicMock()
        mock_db.insert_properties.return_value = 2
        
        # Create input data
        input_data = pd.DataFrame(self.mock_property_data['properties'])
        transformed_data = self.scraper._transform_data(input_data)
        
        # Call load method
        result = self.scraper._load_data(transformed_data, mock_db)
        
        # Assertions
        self.assertEqual(result, 2)
        mock_db.insert_properties.assert_called_once()
        args, kwargs = mock_db.insert_properties.call_args
        self.assertEqual(kwargs['source'], 'MLS')

if __name__ == '__main__':
    unittest.main()
