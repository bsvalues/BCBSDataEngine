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
        
        # Set API key for testing
        self.scraper.api_key = "test_api_key"
        
        # Call extract method without CSV path to use API
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
        
        # Set API key for testing
        self.scraper.api_key = "test_api_key"
        
        # Call extract method without CSV path to use API
        result = self.scraper.extract()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    @patch('etl.mls_scraper.requests.get')
    def test_extract_api_error(self, mock_get):
        """Test extraction with API error."""
        # Configure mock to raise exception
        mock_get.side_effect = Exception("API Error")
        
        # Set API key for testing
        self.scraper.api_key = "test_api_key"
        
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

    @patch('etl.mls_scraper.pd.read_csv')
    def test_read_csv_data(self, mock_read_csv):
        """Test reading data from CSV file."""
        # Create test data with intentional issues
        test_data = {
            'mls_id': ['MLS123', 'MLS456', 'MLS789', 'MLS123', None],  # Duplicate and None
            'address': ['123 Main St', '456 Oak Ave', '789 Pine Dr', '101 Maple Ln', '202 Cedar Blvd'],
            'city': ['Anytown', 'Sometown', 'Othertown', 'Newtown', None],  # None value
            'state': ['CA', 'NY', 'TX', 'FL', 'WA'],
            'zip_code': ['12345', '23456', '34567', '45678', '56789'],
            'list_price': [500000, 300000, 400000, '450000', None],  # String and None
            'square_feet': [2000, 1500, 1800, 2200, None],  # None value
            'year_built': [1990, 2005, 1975, None, 2020],  # None value
            'listing_date': ['2023-01-15', '01/15/2023', '2023-02-01', None, '2022-12-01']  # Mixed formats and None
        }
        
        # Create mock DataFrame
        mock_df = pd.DataFrame(test_data)
        mock_read_csv.return_value = mock_df
        
        # Test behavior with duplicate IDs - should raise ValueError
        with self.assertRaises(ValueError):
            self.scraper.read_csv_data('fake_path.csv')
        
        # Remove the duplicate to test successful processing
        revised_data = test_data.copy()
        revised_data['mls_id'] = ['MLS123', 'MLS456', 'MLS789', 'MLS999', None]
        mock_df = pd.DataFrame(revised_data)
        mock_read_csv.return_value = mock_df
        
        # The function should drop rows with None in critical columns like mls_id,
        # so we'll continue, but the resulting DataFrame will have that row dropped
        result = self.scraper.read_csv_data('fake_path.csv')
        self.assertEqual(len(result), 4)  # Only 4 rows should remain since one had None in mls_id
        
        # Now create a clean dataset without critical issues
        clean_data = revised_data.copy()
        clean_data['mls_id'] = ['MLS123', 'MLS456', 'MLS789', 'MLS999', 'MLS555']  # No None
        clean_data['city'] = ['Anytown', 'Sometown', 'Othertown', 'Newtown', 'Lasttown']  # No None
        mock_df = pd.DataFrame(clean_data)
        mock_read_csv.return_value = mock_df
        
        # Test successful processing
        result = self.scraper.read_csv_data('fake_path.csv')
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        
        # Check data cleaning
        # Numeric values should be converted and NaNs filled with 0
        self.assertEqual(result['list_price'].dtype, float)
        self.assertEqual(result['square_feet'].dtype, float)
        self.assertEqual(result.loc[3, 'year_built'], 0)  # Was None, should be 0 now
        
        # Dates should be standardized
        # Find non-NaT values and verify they're properly converted to Timestamp
        non_nat_dates = result['listing_date'].dropna()
        self.assertGreater(len(non_nat_dates), 0, "No valid dates found after processing")
        for date in non_nat_dates:
            self.assertIsInstance(date, pd.Timestamp)
        
    @patch('etl.mls_scraper.pd.read_csv')
    def test_extract_with_csv(self, mock_read_csv):
        """Test extraction with CSV file instead of API."""
        # Create test data
        test_data = {
            'mls_id': ['MLS123', 'MLS456', 'MLS789'],
            'address': ['123 Main St', '456 Oak Ave', '789 Pine Dr'],
            'city': ['Anytown', 'Sometown', 'Othertown'],
            'state': ['CA', 'NY', 'TX'],
            'zip_code': ['12345', '23456', '34567'],
            'list_price': [500000, 300000, 400000],
            'square_feet': [2000, 1500, 1800],
            'year_built': [1990, 2005, 1975],
            'listing_date': ['2023-01-15', '2023-02-01', '2023-03-15']
        }
        
        # Create mock DataFrame
        mock_df = pd.DataFrame(test_data)
        mock_read_csv.return_value = mock_df
        
        # Call extract with CSV path
        result = self.scraper.extract(csv_file_path='fake_path.csv')
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['mls_id'], 'MLS123')
        
        # Verify read_csv was called correctly
        mock_read_csv.assert_called_once_with('fake_path.csv', low_memory=False)

if __name__ == '__main__':
    unittest.main()
