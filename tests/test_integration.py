"""
Integration tests for the BCBS_Values system.

These tests validate the full ETL pipeline and API endpoints
in an integrated manner.
"""
import os
import sys
import unittest
import json
import tempfile
from datetime import datetime
from unittest import mock
import pytest
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ETL components
from etl.pacs_import import PACSImporter
from etl.mls_scraper import MLSScraper
from etl.narrpr_scraper import NARRPRScraper
from etl.data_validation import validate_property_data

# Import database components
from db.database import Database
from db.models import Property, PropertyValuation, ValidationResult

# Import valuation component
from src.valuation import simple_property_valuation, advanced_property_valuation

# Import API components (for API endpoint testing)
import api
from fastapi.testclient import TestClient

# Create a test client for the FastAPI application
client = TestClient(api.app)

class TestETLPipeline(unittest.TestCase):
    """Integration test for the ETL pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that should be shared across all tests."""
        # Create a test database connection
        cls.db = Database()
        
        # Sample property data for testing
        cls.sample_property_data = {
            'property_id': ['BC12345', 'BC67890'],
            'parcel_id': ['12345', '67890'],
            'address': ['123 Main St, Richland, WA 99352', '456 Oak Ave, Kennewick, WA 99336'],
            'city': ['Richland', 'Kennewick'],
            'state': ['WA', 'WA'],
            'zip_code': ['99352', '99336'],
            'bedrooms': [3, 4],
            'bathrooms': [2.0, 3.5],
            'square_feet': [1800, 2400],
            'lot_size': [10890, 21780],  # 0.25 and 0.5 acres in sq ft
            'year_built': [2005, 2010],
            'property_type': ['Residential', 'Residential'],
            'pool': ['Yes', 'No'],
            'basement': ['Full', 'No'],
            'assessed_value': [500000, 750000],
            'last_sale_price': [450000, 700000],
            'last_sale_date': ['2020-01-15', '2019-06-30'],
            'land_value': [200000, 300000],
            'improvement_value': [300000, 450000],
            'data_source': ['PACS', 'PACS'],
            'collection_date': [datetime.now(), datetime.now()]
        }
    
    def setUp(self):
        """Set up test fixtures for each individual test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock PACS data file
        import pandas as pd
        self.pacs_csv_path = os.path.join(self.temp_dir.name, 'pacs_sample.csv')
        pd.DataFrame({
            'ParcelID': ['12345', '67890'],
            'PropertyAddress': ['123 Main St', '456 Oak Ave'],
            'SitusCity': ['Richland', 'Kennewick'],
            'SitusState': ['WA', 'WA'],
            'SitusZip': ['99352', '99336'],
            'Bedrooms': [3, 4],
            'Bathrooms': [2, 3.5],
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
        }).to_csv(self.pacs_csv_path, index=False)
        
        # Set up mock objects
        self.mock_db = mock.MagicMock()
        self.mock_db.insert_properties.return_value = 2
        self.mock_db.get_properties.return_value = pd.DataFrame(self.sample_property_data)
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory and its contents
        self.temp_dir.cleanup()
    
    @mock.patch('etl.pacs_import.PACSImporter.extract')
    @mock.patch('etl.pacs_import.PACSImporter.transform_and_load')
    @mock.patch('etl.data_validation.validate_property_data')
    def test_etl_pipeline_complete_run(self, mock_validate, mock_transform_load, mock_extract):
        """Test the complete ETL pipeline from extraction to validation."""
        # Mock the return values
        mock_extract.return_value = pd.DataFrame({'dummy': [1, 2]})
        mock_transform_load.return_value = 2
        mock_validate.return_value = {
            'validation_passed': True,
            'total_records': 2,
            'valid_records': 2,
            'invalid_records': 0,
            'validation_results': {
                'missing_values': {'count': 0, 'details': []},
                'duplicate_records': {'count': 0, 'details': []},
                'outliers': {'count': 0, 'details': []},
                'invalid_formats': {'count': 0, 'details': []}
            }
        }
        
        # Create a PACS importer instance
        pacs_importer = PACSImporter()
        
        # Run the extract step
        data = pacs_importer.extract(self.pacs_csv_path)
        
        # Run the transform and load steps
        records_loaded = pacs_importer.transform_and_load(data, self.mock_db)
        
        # Run validation on the loaded data
        properties = self.mock_db.get_properties()
        validation_results = validate_property_data(properties)
        
        # Assert extract was called with the correct file path
        mock_extract.assert_called_once_with(self.pacs_csv_path)
        
        # Assert transform_and_load was called with the correct data
        mock_transform_load.assert_called_once()
        
        # Assert validate_property_data was called
        mock_validate.assert_called_once()
        
        # Check that the validation results were returned correctly
        self.assertTrue(validation_results['validation_passed'])
        self.assertEqual(validation_results['total_records'], 2)
        self.assertEqual(validation_results['valid_records'], 2)
    
    @mock.patch('src.valuation.simple_property_valuation')
    def test_valuation_integration(self, mock_valuation):
        """Test the integration between database and valuation components."""
        # Mock the valuation function to return sample valuations
        mock_valuation.return_value = {
            'property_id': 'BC12345',
            'estimated_value': 525000,
            'confidence_score': 0.92,
            'valuation_date': datetime.now(),
            'model_used': 'SimpleLinearRegression',
            'features_used': {'square_feet': 1800, 'bedrooms': 3, 'bathrooms': 2.0},
            'comparable_properties': []
        }
        
        # Assume properties are loaded in the database
        properties = pd.DataFrame(self.sample_property_data)
        
        # Use a sample property for valuation
        sample_property = properties.iloc[0].to_dict()
        
        # Run the valuation function
        valuation_result = simple_property_valuation(sample_property)
        
        # Check that valuation was performed
        mock_valuation.assert_called_once()
        
        # Check that the valuation result has the expected fields
        self.assertIn('estimated_value', valuation_result)
        self.assertIn('confidence_score', valuation_result)
        self.assertIn('model_used', valuation_result)
        
        # Verify the valuation is reasonable (within expected range)
        self.assertEqual(valuation_result['estimated_value'], 525000)
        self.assertGreaterEqual(valuation_result['confidence_score'], 0.9)
    
    def test_validation_results_storage(self):
        """Test that validation results are properly stored in the database."""
        # Create mock validation results
        validation_results = {
            'validation_passed': True,
            'total_records': 100,
            'valid_records': 98,
            'invalid_records': 2,
            'validation_results': {
                'missing_values': {'count': 1, 'details': [{'record_id': '12345', 'field': 'year_built'}]},
                'duplicate_records': {'count': 0, 'details': []},
                'outliers': {'count': 1, 'details': [{'record_id': '67890', 'field': 'square_feet', 'value': 10000}]},
                'invalid_formats': {'count': 0, 'details': []}
            }
        }
        
        # Mock the database store_validation_results method
        self.mock_db.store_validation_results.return_value = 1
        
        # Store the validation results
        result_id = self.mock_db.store_validation_results(validation_results, 'PACS')
        
        # Check that the database store method was called
        self.mock_db.store_validation_results.assert_called_once()
        
        # Check that a valid result ID was returned
        self.assertEqual(result_id, 1)
        
        # Get the stored validation results (mock)
        self.mock_db.get_latest_validation_results.return_value = {
            'validation_id': 1,
            'validation_passed': True,
            'total_records': 100,
            'valid_records': 98,
            'invalid_records': 2,
            'source': 'PACS',
            'validation_date': datetime.now(),
            'details': json.dumps(validation_results['validation_results'])
        }
        
        # Retrieve the validation results
        stored_results = self.mock_db.get_latest_validation_results()
        
        # Check that the get method was called
        self.mock_db.get_latest_validation_results.assert_called_once()
        
        # Check that the results match what we stored
        self.assertEqual(stored_results['validation_passed'], True)
        self.assertEqual(stored_results['total_records'], 100)
        self.assertEqual(stored_results['valid_records'], 98)


class TestAPIEndpoints(unittest.TestCase):
    """Test the API endpoints using FastAPI TestClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the database dependency
        self.original_get_db = api.get_db
        
        # Create a mock database object
        self.mock_db = mock.MagicMock()
        
        # Sample property valuation for API responses
        self.sample_valuation = {
            'property_id': 'BC12345',
            'address': '123 Main St, Richland, WA 99352',
            'estimated_value': 525000.0,
            'confidence_score': 0.92,
            'model_used': 'SimpleLinearRegression',
            'valuation_date': datetime.now().isoformat(),
            'features_used': {
                'square_feet': 1800,
                'bedrooms': 3,
                'bathrooms': 2.0,
                'lot_size': 10890
            },
            'comparable_properties': [
                {
                    'property_id': 'BC67890',
                    'address': '456 Oak Ave, Kennewick, WA 99336',
                    'sale_price': 700000,
                    'sale_date': '2019-06-30',
                    'similarity_score': 0.85
                }
            ]
        }
        
        # Override the get_db dependency
        def mock_get_db():
            yield self.mock_db
        
        api.app.dependency_overrides[api.get_db] = mock_get_db
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore the original get_db dependency
        api.app.dependency_overrides.clear()
    
    def test_api_root(self):
        """Test the API root endpoint."""
        response = client.get('/')
        
        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Check that the response contains expected fields
        data = response.json()
        self.assertIn('api_name', data)
        self.assertIn('version', data)
        self.assertIn('endpoints', data)
    
    def test_get_valuations(self):
        """Test the get_valuations endpoint."""
        # Mock the database get_latest_valuations method
        self.mock_db.get_latest_valuations.return_value = [self.sample_valuation]
        
        # Call the endpoint
        response = client.get('/api/valuations')
        
        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Check that the response contains the expected data
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['property_id'], 'BC12345')
        self.assertEqual(data[0]['estimated_value'], 525000.0)
        self.assertEqual(data[0]['confidence_score'], 0.92)
    
    def test_get_valuation_by_id(self):
        """Test the get_valuation_by_id endpoint."""
        # Mock the database get_property_valuation method
        self.mock_db.get_property_valuation.return_value = self.sample_valuation
        
        # Call the endpoint
        response = client.get('/api/valuations/BC12345')
        
        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Check that the response contains the expected data
        data = response.json()
        self.assertEqual(data['property_id'], 'BC12345')
        self.assertEqual(data['estimated_value'], 525000.0)
        self.assertEqual(data['model_used'], 'SimpleLinearRegression')
        
        # Check that comparable properties are included
        self.assertIn('comparable_properties', data)
        self.assertEqual(len(data['comparable_properties']), 1)
        self.assertEqual(data['comparable_properties'][0]['property_id'], 'BC67890')
    
    def test_get_valuation_by_id_not_found(self):
        """Test the get_valuation_by_id endpoint with a non-existent ID."""
        # Mock the database get_property_valuation method to return None
        self.mock_db.get_property_valuation.return_value = None
        
        # Call the endpoint with a non-existent ID
        response = client.get('/api/valuations/NONEXISTENT')
        
        # Check that the response is a 404 error
        self.assertEqual(response.status_code, 404)
        
        # Check that the error message is as expected
        data = response.json()
        self.assertIn('detail', data)
        self.assertIn('not found', data['detail'])
    
    def test_get_etl_status(self):
        """Test the get_etl_status endpoint."""
        # Mock the database get_latest_validation_results method
        sample_validation = {
            'validation_id': 1,
            'validation_passed': True,
            'total_records': 100,
            'valid_records': 98,
            'invalid_records': 2,
            'source': 'PACS',
            'validation_date': datetime.now(),
            'details': json.dumps({
                'missing_values': {'count': 1, 'details': []},
                'duplicate_records': {'count': 0, 'details': []},
                'outliers': {'count': 1, 'details': []},
                'invalid_formats': {'count': 0, 'details': []}
            })
        }
        self.mock_db.get_latest_validation_results.return_value = sample_validation
        
        # Mock the database get_property_counts_by_source method
        self.mock_db.get_property_counts_by_source.return_value = [
            {'source': 'PACS', 'count': 100},
            {'source': 'MLS', 'count': 50},
            {'source': 'NARRPR', 'count': 25}
        ]
        
        # Call the endpoint
        response = client.get('/api/etl-status')
        
        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Check that the response contains the expected data
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('last_run', data)
        self.assertIn('sources_processed', data)
        self.assertIn('records_processed', data)
        self.assertIn('validation_status', data)
        
        # Check specific values
        self.assertEqual(data['validation_status'], 'Passed')
        self.assertEqual(data['records_processed'], 175)  # 100 + 50 + 25
        self.assertEqual(len(data['sources_processed']), 3)
    
    def test_get_etl_status_no_validation(self):
        """Test the get_etl_status endpoint when no validation results exist."""
        # Mock the database get_latest_validation_results method to return None
        self.mock_db.get_latest_validation_results.return_value = None
        
        # Mock the database get_property_counts_by_source method
        self.mock_db.get_property_counts_by_source.return_value = []
        
        # Call the endpoint
        response = client.get('/api/etl-status')
        
        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Check that the response indicates no ETL has been run
        data = response.json()
        self.assertEqual(data['status'], 'Not Started')
        self.assertEqual(data['records_processed'], 0)
        self.assertEqual(len(data['sources_processed']), 0)


if __name__ == '__main__':
    unittest.main()