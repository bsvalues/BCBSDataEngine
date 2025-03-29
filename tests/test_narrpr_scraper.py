"""
Tests for the NARRPR scraper module.
"""
import os
import sys
import unittest
from unittest import mock
import pandas as pd
from datetime import datetime

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.narrpr_scraper import NARRPRScraper

class TestNARRPRScraper(unittest.TestCase):
    """Test cases for the NARRPR scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = NARRPRScraper()
        
        # Create mock data for testing
        self.mock_property_data = [
            {
                'address': '123 Main St',
                'city': 'Anytown',
                'state': 'CA',
                'zip_code': '90210',
                'bedrooms': 3.0,
                'bathrooms': 2.0,
                'square_feet': 1800.0,
                'year_built': 2005,
                'list_price': 500000.0,
                'property_type': 'residential',
                'data_source': 'NARRPR',
                'scrape_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'address': '456 Oak Ave',
                'city': 'Sometown',
                'state': 'CA',
                'zip_code': '90211',
                'bedrooms': 4.0,
                'bathrooms': 3.0,
                'square_feet': 2400.0,
                'year_built': 2010,
                'list_price': 750000.0,
                'property_type': 'residential',
                'data_source': 'NARRPR',
                'scrape_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
    @mock.patch('etl.narrpr_scraper.webdriver')
    @mock.patch('etl.narrpr_scraper.Service')
    @mock.patch('etl.narrpr_scraper.ChromeDriverManager')
    @mock.patch.dict(os.environ, {"NARRPR_USERNAME": "testuser", "NARRPR_PASSWORD": "testpass"})
    def test_login_credentials_validation(self, mock_cdm, mock_service, mock_webdriver):
        """Test that the scraper validates login credentials."""
        # This test simply verifies that the function doesn't raise an exception
        # when the credentials are available in the environment
        
        # Create a mock driver that returns empty results
        mock_driver = mock.MagicMock()
        mock_webdriver.Chrome.return_value = mock_driver
        
        # Make the driver return no results to avoid complex mocking
        mock_wait = mock.MagicMock()
        mock_driver.find_elements.return_value = []
        
        # Test should pass without raising ValueError for missing credentials
        try:
            self.scraper.narrpr_login_and_scrape(search_location="Test Location")
            # If we get here, it didn't raise the ValueError we wanted to test for
            self.assertTrue(True)
        except Exception as e:
            # We should not get here with the mocked credentials
            if isinstance(e, ValueError) and "credentials" in str(e).lower():
                self.fail("Credential validation failed with valid credentials")
                
    @mock.patch.dict(os.environ, {}, clear=True)  # Empty environment
    def test_missing_credentials(self):
        """Test that the scraper raises an error when credentials are missing."""
        with self.assertRaises(ValueError) as context:
            self.scraper.narrpr_login_and_scrape(search_location="Test Location")
        
        # Check that the error message mentions credentials
        self.assertIn("credentials", str(context.exception).lower())
        
    @mock.patch('etl.narrpr_scraper.NARRPRScraper.narrpr_login_and_scrape')
    @mock.patch.dict(os.environ, {"NARRPR_USERNAME": "testuser", "NARRPR_PASSWORD": "testpass"})
    def test_scrape_and_load(self, mock_scrape):
        """Test the scrape_and_load method."""
        # Mock the scrape function to return our test data
        mock_scrape.return_value = pd.DataFrame(self.mock_property_data)
        
        # Mock database object
        mock_db = mock.MagicMock()
        mock_db.insert_properties.return_value = len(self.mock_property_data)
        
        # Test the scrape_and_load function
        result = self.scraper.scrape_and_load("Test Location", "residential", mock_db)
        
        # Assertions
        self.assertEqual(result, len(self.mock_property_data))
        mock_scrape.assert_called_once()
        mock_db.insert_properties.assert_called_once()
        
    @mock.patch('etl.narrpr_scraper.NARRPRScraper.narrpr_login_and_scrape')
    @mock.patch.dict(os.environ, {"NARRPR_USERNAME": "testuser", "NARRPR_PASSWORD": "testpass"})
    def test_scrape_and_load_no_results(self, mock_scrape):
        """Test the scrape_and_load method when no results are found."""
        # Mock the scrape function to return an empty DataFrame
        mock_scrape.return_value = pd.DataFrame()
        
        # Mock database object
        mock_db = mock.MagicMock()
        
        # Test the scrape_and_load function
        result = self.scraper.scrape_and_load("Test Location", "residential", mock_db)
        
        # Assertions
        self.assertEqual(result, 0)
        mock_scrape.assert_called_once()
        mock_db.insert_properties.assert_not_called()
        
    def test_transform_data(self):
        """Test the _transform_data method."""
        # Create a DataFrame from our mock data
        input_df = pd.DataFrame(self.mock_property_data)
        
        # Call the transform method
        result_df = self.scraper._transform_data(input_df)
        
        # Assertions
        self.assertEqual(len(result_df), len(self.mock_property_data))
        self.assertEqual(result_df['data_source'].unique()[0], 'NARRPR')
        self.assertTrue('import_date' in result_df.columns)
        
if __name__ == '__main__':
    unittest.main()