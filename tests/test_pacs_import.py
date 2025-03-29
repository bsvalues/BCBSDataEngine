"""
Tests for the PACS importer module.
"""
import os
import sys
import unittest
from unittest import mock
import pandas as pd
from datetime import datetime
import tempfile

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.pacs_import import PACSImporter

class TestPACSImporter(unittest.TestCase):
    """Test cases for the PACS importer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.importer = PACSImporter()
        
        # Create sample PACS data for testing
        self.sample_pacs_data = {
            'ParcelID': ['12345', '67890'],
            'PropertyAddress': ['123 Main St', '456 Oak Ave'],
            'SitusCity': ['Anytown', 'Sometown'],
            'SitusState': ['CA', 'CA'],
            'SitusZip': ['90210', '90211'],
            'Bedrooms': [3, 4],
            'Bathrooms': [2, 3],
            'LivingSqFt': [1800, 2400],
            'LotSizeAcres': [0.25, 0.5],
            'YearBuilt': [2005, 2010],
            'PropertyType': ['R', 'R'],
            'HasPool': ['Y', 'N'],
            'Basement': ['FULL', 'NO'],
            'TotalAssessedValue': [500000, 750000],
            'LastSalePrice': [450000, 700000],
            'SaleDate': ['2020-01-15', '2019-06-30'],
            'LandValue': [200000, 300000],
            'ImprovementValue': [300000, 450000],
            'AssessmentYear': [2022, 2022],
            'AssessmentDate': ['2022-01-01', '2022-01-01']
        }
    
    def test_extract_file_not_found(self):
        """Test that FileNotFoundError is raised when file is not found."""
        with self.assertRaises(FileNotFoundError):
            self.importer.extract(file_path="nonexistent_file.csv")
    
    def test_transform_data(self):
        """Test the data transformation logic."""
        # Create a DataFrame from our sample data
        input_df = pd.DataFrame(self.sample_pacs_data)
        
        # Transform the data
        result_df = self.importer._transform_data(input_df)
        
        # Check that the correct number of records were transformed
        self.assertEqual(len(result_df), 2)
        
        # Check that columns were renamed correctly
        self.assertIn('parcel_id', result_df.columns)
        self.assertIn('address', result_df.columns)
        self.assertIn('city', result_df.columns)
        
        # Check that property types were mapped correctly
        self.assertEqual(result_df['property_type'][0], 'Residential')
        
        # Check that lot size was converted from acres to square feet
        # 0.25 acres = 10,890 sq ft
        self.assertAlmostEqual(result_df['lot_size'][0], 0.25 * 43560)
        
        # Check that boolean fields were mapped correctly
        self.assertEqual(result_df['pool'][0], 'Yes')
        self.assertEqual(result_df['pool'][1], 'No')
        self.assertEqual(result_df['basement'][0], 'Full')
        self.assertEqual(result_df['basement'][1], 'No')
        
        # Check that data source was set correctly
        self.assertEqual(result_df['data_source'][0], 'PACS')
    
    def test_extract_from_csv(self):
        """Test extracting data from a CSV file."""
        # Create a temporary CSV file with our sample data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            # Create a sample CSV file with PACS data
            df = pd.DataFrame(self.sample_pacs_data)
            df.to_csv(temp.name, index=False)
            temp_name = temp.name
        
        try:
            # Mock the CSV reading to avoid dealing with date parsing in this test
            with mock.patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame(self.sample_pacs_data)
                
                # Call the extract method
                result = self.importer.extract(file_path=temp_name)
                
                # Check that read_csv was called with the correct file path
                mock_read_csv.assert_called_once()
                args, kwargs = mock_read_csv.call_args
                self.assertEqual(args[0], temp_name)
                
                # Check the returned DataFrame
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(len(result), 2)
        finally:
            # Clean up the temporary file
            os.unlink(temp_name)
    
    def test_load_data(self):
        """Test loading transformed data into the database."""
        # Create a mock database object
        mock_db = mock.MagicMock()
        mock_db.insert_properties.return_value = 2
        
        # Create a DataFrame from our sample data
        data = pd.DataFrame(self.sample_pacs_data)
        
        # Transform the data
        transformed_data = self.importer._transform_data(data)
        
        # Load the data into the mock database
        records_loaded = self.importer._load_data(transformed_data, mock_db)
        
        # Check that insert_properties was called with the correct arguments
        mock_db.insert_properties.assert_called_once()
        args, kwargs = mock_db.insert_properties.call_args
        self.assertEqual(kwargs['source'], 'PACS')
        
        # Check that the correct number of records were loaded
        self.assertEqual(records_loaded, 2)
    
    def test_transform_and_load(self):
        """Test the combined transform_and_load method."""
        # Create a mock database object
        mock_db = mock.MagicMock()
        mock_db.insert_properties.return_value = 2
        
        # Create a DataFrame from our sample data
        data = pd.DataFrame(self.sample_pacs_data)
        
        # Mock the transform and load methods
        with mock.patch.object(self.importer, '_transform_data') as mock_transform:
            with mock.patch.object(self.importer, '_load_data') as mock_load:
                # Set up the transform mock to return transformed data
                mock_transform.return_value = pd.DataFrame({'transformed': [1, 2]})
                # Set up the load mock to return number of records loaded
                mock_load.return_value = 2
                
                # Call the transform_and_load method
                result = self.importer.transform_and_load(data, mock_db)
                
                # Check that the transform method was called with the input data
                mock_transform.assert_called_once_with(data)
                
                # Check that the load method was called with the transformed data
                mock_load.assert_called_once()
                
                # Check that the correct number of records were loaded
                self.assertEqual(result, 2)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        
        # Create a mock database object
        mock_db = mock.MagicMock()
        
        # Call the transform_and_load method with empty data
        result = self.importer.transform_and_load(empty_df, mock_db)
        
        # Check that no records were loaded
        self.assertEqual(result, 0)
        
        # Check that the database insert method was not called
        mock_db.insert_properties.assert_not_called()
    
    def test_read_sample_file(self):
        """Test the read_sample_file method."""
        # Create a temporary CSV file with our sample data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            # Create a sample CSV file with PACS data
            df = pd.DataFrame(self.sample_pacs_data)
            df.to_csv(temp.name, index=False)
            temp_name = temp.name
        
        try:
            # Mock the CSV reading function
            with mock.patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame(self.sample_pacs_data).head(3)
                
                # Call the read_sample_file method
                result = self.importer.read_sample_file(temp_name, n=3)
                
                # Check that read_csv was called with the correct arguments
                mock_read_csv.assert_called_once()
                args, kwargs = mock_read_csv.call_args
                self.assertEqual(args[0], temp_name)
                self.assertEqual(kwargs['nrows'], 3)
                
                # Check the returned DataFrame
                self.assertIsInstance(result, pd.DataFrame)
        finally:
            # Clean up the temporary file
            os.unlink(temp_name)

if __name__ == '__main__':
    unittest.main()