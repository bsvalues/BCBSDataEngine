"""
Unit tests for PACS importer module.
"""
import unittest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
from etl.pacs_import import PACSImporter

class TestPACSImporter(unittest.TestCase):
    """
    Test cases for the PACSImporter class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.importer = PACSImporter(batch_size=10)
        
        # Mock property data
        self.mock_property_data = {
            "properties": [
                {
                    "parcelId": "APN12345678",
                    "apn": "12345678",
                    "propertyAddress": "123 Main St",
                    "city": "Anytown",
                    "county": "Some County",
                    "state": "CA",
                    "zipCode": "12345",
                    "propertyType": "Single Family",
                    "landValue": 200000,
                    "improvementValue": 300000,
                    "totalValue": 500000,
                    "assessmentYear": 2023,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "buildingArea": 2000,
                    "lotSize": 0.25,
                    "yearBuilt": 1990,
                    "lastSalePrice": 450000,
                    "lastSaleDate": "2020-06-15"
                },
                {
                    "parcelId": "APN87654321",
                    "apn": "87654321",
                    "propertyAddress": "456 Oak Ave",
                    "city": "Sometown",
                    "county": "Other County",
                    "state": "CA",
                    "zipCode": "54321",
                    "propertyType": "Condo",
                    "landValue": 100000,
                    "improvementValue": 200000,
                    "totalValue": 300000,
                    "assessmentYear": 2023,
                    "bedrooms": 2,
                    "bathrooms": 1,
                    "buildingArea": 1200,
                    "lotSize": 0.0,
                    "yearBuilt": 2005,
                    "lastSalePrice": 275000,
                    "lastSaleDate": "2019-03-22"
                }
            ]
        }
    
    @patch('etl.pacs_import.requests.get')
    def test_extract_from_api_success(self, mock_get):
        """Test successful data extraction from API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_property_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call extract method
        result = self.importer.extract(county="Some County", state="CA")
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['parcelId'], 'APN12345678')
        self.assertEqual(result.iloc[1]['parcelId'], 'APN87654321')
        
        # Verify the API was called with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue(args[0].endswith('/properties'))
        self.assertEqual(kwargs['params']['county'], "Some County")
        self.assertEqual(kwargs['params']['state'], "CA")
        self.assertEqual(kwargs['params']['limit'], 10)
    
    def test_extract_from_csv_file(self):
        """Test extraction from CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_filename = temp_file.name
            
            # Write mock data to CSV
            csv_data = pd.DataFrame(self.mock_property_data['properties'])
            csv_data.to_csv(temp_filename, index=False)
        
        try:
            # Call extract method with file path
            result = self.importer.extract(file_path=temp_filename)
            
            # Assertions
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.iloc[0]['parcelId'], 'APN12345678')
            self.assertEqual(result.iloc[1]['parcelId'], 'APN87654321')
            
        finally:
            # Clean up temporary file
            os.unlink(temp_filename)
    
    def test_extract_from_unsupported_file(self):
        """Test extraction from unsupported file type."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        try:
            # Call extract method with unsupported file path
            with self.assertRaises(ValueError):
                self.importer.extract(file_path=temp_filename)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_filename)
    
    @patch('etl.pacs_import.requests.get')
    def test_extract_empty_response(self, mock_get):
        """Test extraction with empty response."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"properties": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call extract method
        result = self.importer.extract()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    def test_transform_data(self):
        """Test data transformation."""
        # Create input data
        input_data = pd.DataFrame(self.mock_property_data['properties'])
        
        # Call transform method
        result = self.importer._transform_data(input_data)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Check column renaming
        self.assertIn('parcel_id', result.columns)
        self.assertIn('apn', result.columns)
        self.assertIn('address', result.columns)
        self.assertIn('land_value', result.columns)
        
        # Check data source and import date
        self.assertEqual(result.iloc[0]['data_source'], 'PACS')
        self.assertIsNotNone(result.iloc[0]['import_date'])
        
        # Check data type conversions
        self.assertTrue(pd.api.types.is_numeric_dtype(result['land_value']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['square_feet']))
        self.assertIsInstance(result.iloc[0]['last_sale_date'], pd.Timestamp)
    
    def test_load_data(self):
        """Test data loading."""
        # Create mock database
        mock_db = MagicMock()
        mock_db.insert_properties.return_value = 2
        
        # Create input data
        input_data = pd.DataFrame(self.mock_property_data['properties'])
        transformed_data = self.importer._transform_data(input_data)
        
        # Call load method
        result = self.importer._load_data(transformed_data, mock_db)
        
        # Assertions
        self.assertEqual(result, 2)
        mock_db.insert_properties.assert_called_once()
        args, kwargs = mock_db.insert_properties.call_args
        self.assertEqual(kwargs['source'], 'PACS')

if __name__ == '__main__':
    unittest.main()
