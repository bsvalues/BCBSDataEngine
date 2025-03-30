"""
Integration tests for the BCBS_Values system.

These tests validate the full ETL pipeline and API endpoints
in an integrated manner, simulating real-world usage patterns.
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

# Import main module with ETL pipeline
import main

# Import API components (for API endpoint testing)
import api
from fastapi.testclient import TestClient

# Create a test client for the FastAPI application
client = TestClient(api.app)

# Sample data for testing
sample_properties = [
    {
        "property_id": "BC12345",
        "address": "123 Main St, Richland, WA 99352",
        "city": "Richland",
        "county": "Benton",
        "state": "WA",
        "zip_code": "99352",
        "latitude": 46.2807,
        "longitude": -119.2785,
        "property_type": "Single Family",
        "bedrooms": 3,
        "bathrooms": 2.0,
        "square_feet": 1800,
        "lot_size": 10890,
        "year_built": 1985,
        "estimated_value": 380000.0,
        "data_source": "PACS"
    },
    {
        "property_id": "BC67890",
        "address": "456 Oak Ave, Kennewick, WA 99336",
        "city": "Kennewick",
        "county": "Benton",
        "state": "WA",
        "zip_code": "99336",
        "latitude": 46.2087,
        "longitude": -119.1361,
        "property_type": "Single Family",
        "bedrooms": 4,
        "bathrooms": 3.0,
        "square_feet": 2500,
        "lot_size": 12000,
        "year_built": 2005,
        "estimated_value": 450000.0,
        "data_source": "MLS"
    }
]

@pytest.fixture
def mock_database():
    """
    Create a mock database for testing.
    This fixture creates a mock database that will be passed to functions
    that require database access.
    """
    with mock.patch('db.database.Database') as mock_db:
        # Configure the mock to return our test data
        session_mock = mock.MagicMock()
        mock_db.return_value.Session.return_value = session_mock
        
        # Setup query methods to return test data
        session_mock.query.return_value.filter.return_value.all.return_value = []
        
        yield mock_db


@pytest.fixture
def mock_etl_components():
    """
    Create mocks for the ETL components.
    This fixture patches the extract, transform, and load methods of the ETL
    components to avoid making actual API calls during testing.
    """
    with mock.patch('etl.pacs_import.extract_data') as mock_pacs_extract, \
         mock.patch('etl.pacs_import.transform_data') as mock_pacs_transform, \
         mock.patch('etl.pacs_import.load_data') as mock_pacs_load, \
         mock.patch('etl.mls_scraper.extract_data') as mock_mls_extract, \
         mock.patch('etl.mls_scraper.transform_data') as mock_mls_transform, \
         mock.patch('etl.mls_scraper.load_data') as mock_mls_load, \
         mock.patch('etl.narrpr_scraper.extract_data') as mock_narrpr_extract, \
         mock.patch('etl.narrpr_scraper.transform_data') as mock_narrpr_transform, \
         mock.patch('etl.narrpr_scraper.load_data') as mock_narrpr_load, \
         mock.patch('etl.data_validation.validate_data') as mock_validate:
         
        # Configure the extract mocks to return sample data
        raw_data = {'results': sample_properties}
        mock_pacs_extract.return_value = raw_data
        mock_mls_extract.return_value = raw_data
        mock_narrpr_extract.return_value = raw_data
        
        # Configure the transform mocks to return transformed data
        mock_pacs_transform.return_value = sample_properties
        mock_mls_transform.return_value = sample_properties
        mock_narrpr_transform.return_value = sample_properties
        
        # Configure the load mocks to return a count of loaded records
        mock_pacs_load.return_value = len(sample_properties)
        mock_mls_load.return_value = len(sample_properties)
        mock_narrpr_load.return_value = len(sample_properties)
        
        # Configure the validate mock to return successful validation
        mock_validate.return_value = {
            'validation_passed': True,
            'total_records': len(sample_properties),
            'valid_records': len(sample_properties),
            'invalid_records': 0,
            'checks_passed': 10,
            'checks_failed': 0,
            'validation_details': {
                'completeness': {
                    'status': 'passed',
                    'missing_fields': [],
                    'records_with_missing_data': 0
                },
                'consistency': {
                    'status': 'passed',
                    'inconsistent_records': []
                }
            }
        }
        
        yield {
            'pacs_extract': mock_pacs_extract,
            'pacs_transform': mock_pacs_transform,
            'pacs_load': mock_pacs_load,
            'mls_extract': mock_mls_extract,
            'mls_transform': mock_mls_transform,
            'mls_load': mock_mls_load,
            'narrpr_extract': mock_narrpr_extract,
            'narrpr_transform': mock_narrpr_transform,
            'narrpr_load': mock_narrpr_load,
            'validate': mock_validate
        }


class TestETLPipeline:
    """Test suite for the ETL pipeline."""
    
    def test_full_etl_run(self, mock_etl_components, mock_database, tmp_path):
        """
        Test running the full ETL pipeline.
        This test simulates running the full ETL pipeline with all data sources
        and verifies that the correct methods are called in sequence.
        """
        # Set up a temporary directory for output files
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Run the ETL pipeline with all sources
            result = main.run_etl_pipeline(sources=['pacs', 'mls', 'narrpr'], validate_only=False)
            
            # Verify all extract methods were called
            mock_etl_components['pacs_extract'].assert_called_once()
            mock_etl_components['mls_extract'].assert_called_once()
            mock_etl_components['narrpr_extract'].assert_called_once()
            
            # Verify all transform methods were called
            mock_etl_components['pacs_transform'].assert_called_once()
            mock_etl_components['mls_transform'].assert_called_once()
            mock_etl_components['narrpr_transform'].assert_called_once()
            
            # Verify all validation was performed
            assert mock_etl_components['validate'].call_count == 3
            
            # Verify all load methods were called (since validation passed)
            mock_etl_components['pacs_load'].assert_called_once()
            mock_etl_components['mls_load'].assert_called_once()
            mock_etl_components['narrpr_load'].assert_called_once()
            
            # Verify the result contains the expected information
            assert 'total_records_processed' in result
            assert 'total_records_loaded' in result
            assert 'validation_results' in result
            assert len(result['validation_results']) == 3  # One for each source
            
            # Check that the validation results file was created
            validation_files = list(tmp_path.glob('validation_results_*.json'))
            assert len(validation_files) == 1
            
            # Verify the validation results file contains the expected data
            with open(validation_files[0], 'r') as f:
                validation_data = json.load(f)
                assert 'total_records_processed' in validation_data
                assert 'total_records_loaded' in validation_data
                assert 'validation_results' in validation_data
                assert len(validation_data['validation_results']) == 3
        
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def test_validate_only_mode(self, mock_etl_components, mock_database):
        """
        Test running the ETL pipeline in validate-only mode.
        This test verifies that in validate-only mode, validation is performed
        but data is not loaded into the database.
        """
        # Run the ETL pipeline in validate-only mode
        result = main.run_etl_pipeline(sources=['pacs'], validate_only=True)
        
        # Verify extract and transform were called
        mock_etl_components['pacs_extract'].assert_called_once()
        mock_etl_components['pacs_transform'].assert_called_once()
        
        # Verify validation was performed
        mock_etl_components['validate'].assert_called_once()
        
        # Verify load was NOT called (validate-only mode)
        mock_etl_components['pacs_load'].assert_not_called()
        
        # Verify the result contains the expected information
        assert 'total_records_processed' in result
        assert 'total_records_loaded' in result
        assert result['total_records_loaded'] == 0  # No records loaded in validate-only mode


class TestAPIEndpoints:
    """Test suite for the API endpoints."""
    
    @pytest.fixture
    def override_db_dependency(self, mock_database):
        """Override the database dependency in the API."""
        # Store the original dependency
        original_dependency = api.get_db
        
        # Create a mock database session that returns our test data
        def mock_get_db():
            db = mock_database()
            try:
                yield db
            finally:
                db.close()
        
        # Override the dependency
        api.app.dependency_overrides[api.get_db] = mock_get_db
        
        yield
        
        # Restore the original dependency
        api.app.dependency_overrides[api.get_db] = original_dependency
    
    @pytest.fixture
    def mock_property_valuations(self):
        """Set up mock property valuations for API tests."""
        # Create sample property valuations for API responses
        valuations = []
        
        for prop in sample_properties:
            valuation = {
                'property_id': prop['property_id'],
                'address': prop['address'],
                'estimated_value': prop['estimated_value'],
                'confidence_score': 0.92,
                'model_used': 'SimpleLinearRegression',
                'valuation_date': datetime.now().isoformat(),
                'features_used': {
                    'square_feet': prop['square_feet'],
                    'bedrooms': prop['bedrooms'],
                    'bathrooms': prop['bathrooms'],
                    'lot_size': prop['lot_size']
                }
            }
            valuations.append(valuation)
        
        return valuations
    
    def test_api_root(self):
        """Test the API root endpoint."""
        # Call the root endpoint
        response = client.get("/")
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "BCBS_Values API" in data["name"]
    
    def test_get_valuations(self, override_db_dependency, mock_property_valuations):
        """
        Test the /api/valuations endpoint.
        This test verifies that the endpoint returns the expected property valuations
        and applies filters correctly.
        """
        # Mock the database query to return our sample data
        with mock.patch('api.session.query') as mock_query:
            # Setup the mock to return our test data
            mock_query_instance = mock_query.return_value
            mock_join = mock_query_instance.join.return_value
            mock_filter = mock_join.filter.return_value
            mock_subquery = mock.MagicMock()
            mock_join2 = mock_filter.join.return_value
            mock_order_by = mock_join2.order_by.return_value
            mock_limit = mock_order_by.limit.return_value
            
            # Create mock result objects that match what we expect from the database
            mock_results = []
            for i, val in enumerate(mock_property_valuations):
                mock_property = mock.MagicMock()
                mock_property.property_id = val['property_id']
                mock_property.address = val['address']
                
                mock_valuation = mock.MagicMock()
                mock_valuation.estimated_value = val['estimated_value']
                mock_valuation.confidence_score = val['confidence_score']
                mock_valuation.model_name = val['model_used']
                mock_valuation.valuation_date = datetime.fromisoformat(val['valuation_date'])
                mock_valuation.feature_importance = json.dumps(val['features_used'])
                mock_valuation.comparable_properties = None
                
                mock_results.append((mock_valuation, mock_property))
            
            mock_limit.all.return_value = mock_results
            
            # Call the endpoint
            response = client.get("/api/valuations")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == len(mock_property_valuations)
            
            # Verify response properties match our sample data
            for i, item in enumerate(data):
                assert item["property_id"] == mock_property_valuations[i]["property_id"]
                assert item["address"] == mock_property_valuations[i]["address"]
                assert item["estimated_value"] == mock_property_valuations[i]["estimated_value"]
    
    def test_get_etl_status(self, override_db_dependency):
        """
        Test the /api/etl-status endpoint.
        This test verifies that the endpoint returns information about the most recent
        ETL pipeline run and validation results.
        """
        # Mock the database query to return a sample validation result
        with mock.patch('api.session.query') as mock_query, \
             mock.patch('api.session.execute') as mock_execute:
            
            # Set up mock for ValidationResult query
            mock_query_instance = mock_query.return_value
            mock_order_by = mock_query_instance.order_by.return_value
            mock_first = mock_order_by.first.return_value
            
            # Create a mock validation result
            mock_validation = mock.MagicMock()
            mock_validation.id = 1
            mock_validation.timestamp = datetime.now()
            mock_validation.status = "success"
            mock_validation.results = json.dumps({
                'validation_passed': True,
                'total_records': 25,
                'valid_records': 23,
                'validation_details': {
                    'completeness': {'status': 'passed'},
                    'consistency': {'status': 'passed'},
                    'range_checks': {'status': 'warning'}
                }
            })
            
            mock_first = mock_validation
            
            # Set up mock for source counts query
            mock_execute_result = [
                ('PACS', 10),
                ('MLS', 8),
                ('NARRPR', 7)
            ]
            mock_execute.return_value = mock_execute_result
            
            # Call the endpoint
            response = client.get("/api/etl-status")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "last_run" in data
            assert "sources_processed" in data
            assert "records_processed" in data
            assert "validation_status" in data
            assert "validation_details" in data
            
            # Verify that sources processed contains our mock data
            assert len(data["sources_processed"]) == 3
            source_names = [s["name"] for s in data["sources_processed"]]
            assert "PACS" in source_names
            assert "MLS" in source_names
            assert "NARRPR" in source_names
    
    def test_get_agent_status(self):
        """
        Test the /api/agent-status endpoint.
        This test verifies that the endpoint returns information about the agents
        in the BCBS system.
        """
        # Call the endpoint
        response = client.get("/api/agent-status")
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "system_status" in data
        assert "active_agents" in data
        assert "tasks_in_progress" in data
        assert "tasks_completed_today" in data
        
        # Verify that the agents list contains our expected agents
        agent_ids = [a["agent_id"] for a in data["agents"]]
        assert "bcbs-bootstrap-commander" in agent_ids
        assert "bcbs-cascade-operator" in agent_ids
        assert "bcbs-tdd-validator" in agent_ids
        
        # Verify agent details
        for agent in data["agents"]:
            assert "name" in agent
            assert "status" in agent
            assert "last_active" in agent
            assert "queue_size" in agent
            assert "performance_metrics" in agent
            
            # Check performance metrics structure
            metrics = agent["performance_metrics"]
            assert "tasks_completed" in metrics
            assert "success_rate" in metrics


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])