"""
Unit tests for NARRPR scraper module.
"""
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from etl.narrpr_scraper import NARRPRScraper

class TestNARRPRScraper(unittest.TestCase):
    """
    Test cases for the NARRPRScraper class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = NARRPRScraper(batch_size=10)
        
        # Mock property data
        self.mock_property_data = {
            "properties": [
                {
                    "propertyId": "P123456",
                    "addressLine1": "123 Main St",
                    "city": "Anytown",
                    "state": "CA",
                    "zipCode": "12345",
                    "propertyType": "Single Family",
                    "estimatedValue": 520000,
                    "lastSalePrice": 450000,
                    "lastSaleDate": "2020-06-15",
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "totalRooms": 7,
                    "squareFeet": 2100,
                    "lotSize": 0.25,
                    "yearBuilt": 1992
                },
                {
                    "propertyId": "P789012",
                    "addressLine1": "456 Oak Ave",
                    "city": "Sometown",
                    "state": "CA",
                    "zipCode": "54321",
                    "propertyType": "Condo",
                    "estimatedValue": 310000,
                    "lastSalePrice": 285000,
                    "lastSaleDate": "2019-03-22",
                    "bedrooms": 2,
                    "bathrooms": 1,
                    "totalRooms": 4,
                    "squareFeet": 1150,
                    "lotSize": 0.0,
                    "yearBuilt": 2005
                }
            ]
        }
    
    @patch('etl.narrpr_scraper.requests.get')
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
        self.assertEqual(result.iloc[0]['propertyId'], 'P123456')
        self.assertEqual(result.iloc[1]['propertyId'], 'P789012')
        
        # Verify the API was called with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue(args[0].endswith('/properties/search'))
        self.assertEqual(kwargs['params']['limit'], 10)
    
    @patch('etl.narrpr_scraper.requests.get')
    def test_extract_with_location(self, mock_get):
        """Test extraction with location parameter."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_property_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call extract method with location parameter
        result = self.scraper.extract(location="Anytown, CA")
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        
        # Verify the API was called with correct parameters
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs['params']['location'], "Anytown, CA")
    
    @patch('etl.narrpr_scraper.requests.get')
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
    
    @patch('etl.narrpr_scraper.requests.get')
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
        self.assertIn('property_id', result.columns)
        self.assertIn('address', result.columns)
        self.assertIn('property_type', result.columns)
        self.assertIn('estimated_value', result.columns)
        
        # Check data source and import date
        self.assertEqual(result.iloc[0]['data_source'], 'NARRPR')
        self.assertIsNotNone(result.iloc[0]['import_date'])
        
        # Check data type conversions
        self.assertTrue(pd.api.types.is_numeric_dtype(result['estimated_value']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['square_feet']))
        self.assertIsInstance(result.iloc[0]['last_sale_date'], pd.Timestamp)
    
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
        self.assertEqual(kwargs['source'], 'NARRPR')

if __name__ == '__main__':
    unittest.main()
