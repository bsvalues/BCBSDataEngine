"""
Integration Tests for BCBS Values Advanced Functionality
======================================================

This test suite verifies the end-to-end functionality of the BCBS Values platform,
including ETL processes, API endpoints, and advanced valuation algorithms.

Tests cover:
- ETL pipeline execution and data validation/loading into PostgreSQL
- API endpoint responses with proper authentication
- Response structure validation against API contracts
- Query parameter and filtering functionality
- Failure scenario simulations with proper error handling
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import subprocess
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules for testing
from api import app
from db.database import Database
from db.models import Property, ValidationResult, PropertyValuation
from etl.data_validation import validate_property_data, DataValidator
from etl.pacs_import import import_pacs_data

# Create a TestClient for our API
client = TestClient(app)

# Define API routes for testing
API_ROUTES = {
    'valuations': '/api/valuations',
    'etl_status': '/api/etl-status',
    'agent_status': '/api/agent-status',
    'health': '/api/health',
    'property_valuation': '/api/valuations/{property_id}',
    'neighborhoods': '/api/neighborhoods',
    'market_trends': '/api/market-trends'
}

# Define test data constants
TEST_API_KEY = os.environ.get('BCBS_VALUES_API_KEY', 'test_api_key_for_integration')
TEST_PROPERTY_ID = 'BENTON-12345'


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def db_session():
    """
    Creates a test database session and cleans up after tests.
    Uses in-memory SQLite for testing to avoid affecting the production database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.database import Base
    
    # Create in-memory SQLite database for testing
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    
    # Create a new session for testing
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    # Cleanup after test
    session.close()


@pytest.fixture
def sample_property_data():
    """
    Returns a sample property dataset for testing ETL and valuation functions.
    """
    return pd.DataFrame({
        'property_id': [TEST_PROPERTY_ID, 'BENTON-67890', 'BENTON-54321'],
        'address': ['123 Main St, Kennewick, WA', '456 Oak Ave, Richland, WA', '789 Pine St, Prosser, WA'],
        'city': ['Kennewick', 'Richland', 'Prosser'],
        'state': ['WA', 'WA', 'WA'],
        'county': ['Benton', 'Benton', 'Benton'],
        'zip_code': ['99336', '99352', '99350'],
        'bedrooms': [3, 4, 2],
        'bathrooms': [2.5, 3.0, 1.0],
        'square_feet': [2200, 3100, 1500],
        'lot_size': [0.25, 0.4, 0.15],
        'year_built': [1995, 2005, 1975],
        'last_sale_date': ['2020-05-15', '2019-10-10', '2021-02-28'],
        'last_sale_price': [350000, 425000, 275000],
        'property_type': ['Single Family', 'Single Family', 'Single Family'],
        'latitude': [46.2122, 46.2851, 46.2068],
        'longitude': [-119.1372, -119.2785, -119.7683],
        'data_source': ['PACS', 'PACS', 'PACS']
    })


@pytest.fixture
def sample_gis_data():
    """
    Returns sample GIS feature data for testing spatial analysis functions.
    """
    return {
        TEST_PROPERTY_ID: {
            'school_proximity_score': 0.85,
            'park_proximity_score': 0.72,
            'shopping_proximity_score': 0.91,
            'highway_access_score': 0.65,
            'flood_risk_score': 0.12,
            'walkability_score': 0.78,
            'neighborhood_quality_score': 0.82,
            'spatial_cluster_id': 'SC-NW-KENNEWICK-01'
        }
    }


@pytest.fixture
def authenticated_client():
    """
    Returns a TestClient with authentication headers set.
    """
    test_client = TestClient(app)
    test_client.headers.update({"X-API-KEY": TEST_API_KEY})
    return test_client


@pytest.fixture
def sample_etl_status_data():
    """
    Returns sample ETL status data for mocking
    """
    return {
        "status": "completed",
        "last_run": datetime.now().isoformat(),
        "sources_processed": [
            {"name": "MLS", "status": "completed", "records_processed": 150, "progress": 1.0},
            {"name": "NARRPR", "status": "completed", "records_processed": 200, "progress": 1.0},
            {"name": "PACS", "status": "completed", "records_processed": 175, "progress": 1.0}
        ],
        "records_processed": 525,
        "validation_status": "passed",
        "validation_details": {"unique_ids": "passed", "date_formats": "passed", "numeric_ranges": "passed"},
        "records_validated": 525,
        "records_rejected": 0,
        "data_completeness": 0.98,
        "data_accuracy": 0.95,
        "anomalies_detected": [],
        "data_freshness": {
            "MLS": (datetime.now() - timedelta(days=1)).isoformat(),
            "NARRPR": (datetime.now() - timedelta(days=2)).isoformat(),
            "PACS": (datetime.now() - timedelta(days=1)).isoformat()
        },
        "validation_rule_results": {},
        "quality_score": 0.94
    }


@pytest.fixture
def sample_agent_status_data():
    """
    Returns sample agent status data for mocking
    """
    return {
        "agents": [
            {
                "agent_id": "etl-agent-001",
                "name": "ETL Processor",
                "status": "idle",
                "last_active": datetime.now().isoformat(),
                "current_task": None,
                "queue_size": 0,
                "performance_metrics": {"tasks_completed": 42, "success_rate": 0.95, "avg_response_time": 2.3},
                "execution_history": [
                    {"task_id": "task-001", "start_time": (datetime.now() - timedelta(hours=2)).isoformat(), 
                     "end_time": (datetime.now() - timedelta(hours=1, minutes=40)).isoformat(), "status": "completed"}
                ],
                "success_count": 40,
                "failure_count": 2,
                "average_execution_time": 1200.5,
                "last_execution_time": 1050.2,
                "error_rate": 0.05,
                "resource_usage": {"cpu": 0.15, "memory": 0.25},
                "agent_version": "1.0.0",
                "uptime": 86400,
                "health_score": 0.98
            },
            {
                "agent_id": "valuation-agent-001",
                "name": "Valuation Processor",
                "status": "active",
                "last_active": datetime.now().isoformat(),
                "current_task": "property-valuation-batch-12",
                "queue_size": 5,
                "performance_metrics": {"tasks_completed": 38, "success_rate": 0.92, "avg_response_time": 3.1},
                "execution_history": [
                    {"task_id": "task-002", "start_time": (datetime.now() - timedelta(minutes=30)).isoformat(), 
                     "end_time": (datetime.now() - timedelta(minutes=25)).isoformat(), "status": "completed"}
                ],
                "success_count": 35,
                "failure_count": 3,
                "average_execution_time": 1500.8,
                "last_execution_time": 1250.3,
                "error_rate": 0.08,
                "resource_usage": {"cpu": 0.45, "memory": 0.35},
                "agent_version": "1.0.0",
                "uptime": 43200,
                "health_score": 0.94
            }
        ],
        "system_status": "operational",
        "total_agents": 2,
        "active_agents": 1,
        "inactive_agents": 1,
        "system_health": 0.96,
        "total_tasks_processed": 80,
        "tasks_succeeded": 75,
        "tasks_failed": 5,
        "system_uptime": 86400,
        "system_resource_usage": {"cpu": 0.30, "memory": 0.40},
        "processing_capacity": 15.0,
        "task_queue_depth": 5
    }


@pytest.fixture
def mock_database_session(db_session, sample_property_data):
    """
    Setup a database session with sample data.
    """
    # Insert sample properties
    for _, row in sample_property_data.iterrows():
        property_data = Property(
            property_id=row['property_id'],
            address=row['address'],
            city=row['city'],
            state=row['state'],
            zip_code=row['zip_code'],
            bedrooms=row['bedrooms'],
            bathrooms=row['bathrooms'],
            square_feet=row['square_feet'],
            lot_size=row['lot_size'],
            year_built=row['year_built'],
            property_type=row['property_type'],
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        db_session.add(property_data)
    
    # Add a sample property valuation
    valuation = PropertyValuation(
        property_id=1,  # References the first property
        valuation_date=datetime.now(),
        estimated_value=380000.0,
        confidence_score=0.92,
        valuation_method="advanced_regression",
        valuation_notes="Test valuation"
    )
    db_session.add(valuation)
    
    db_session.commit()
    return db_session


# ======================================================================
# ETL Pipeline Integration Tests
# ======================================================================

def test_main_etl_execution():
    """
    Test the full ETL pipeline by executing main.py.
    
    This test:
    1. Executes the main.py script with python subprocess
    2. Verifies the process completes with success exit code
    3. Checks that data was properly processed
    """
    # Skip this test in CI/CD environments or if database isn't available
    if 'CI' in os.environ:
        pytest.skip("Skipping full ETL execution in CI environment")
    
    try:
        # Check database connection first
        db = Database()
        db.close()
    except Exception as e:
        pytest.skip(f"Database not available: {str(e)}")
    
    try:
        # Execute the main.py script
        result = subprocess.run(
            ['python', 'main.py', '--etl-only', '--test-mode'],
            capture_output=True,
            text=True,
            timeout=60  # Maximum 60 seconds to run
        )
        
        # Check execution was successful
        assert result.returncode == 0, f"ETL execution failed with output: {result.stderr}"
        
        # Validate output contains expected patterns
        assert "ETL process started" in result.stdout
        assert "Data validation complete" in result.stdout
        assert "ETL process completed successfully" in result.stdout
        
        # Access the database to verify data was inserted
        db = Database()
        properties_count = db.get_property_count()
        db.close()
        
        # Verify at least some properties were processed
        assert properties_count > 0, "No properties were processed by the ETL pipeline"
        
    except subprocess.TimeoutExpired:
        pytest.fail("ETL process timed out after 60 seconds")
    except Exception as e:
        pytest.fail(f"Error executing ETL process: {str(e)}")


@patch('etl.pacs_import.requests.get')
def test_etl_data_validation_and_loading(mock_get, db_session, sample_property_data):
    """
    Tests the data validation and database loading of the ETL pipeline.
    
    This test:
    1. Mocks the PACS API response with sample property data
    2. Executes the import and validation functions
    3. Verifies data validation and database loading
    """
    # Configure mock response from PACS API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'properties': sample_property_data.to_dict('records')}
    mock_get.return_value = mock_response
    
    # First validate the data
    validation_passed, validation_summary = validate_property_data(sample_property_data)
    
    # Verify validation results
    assert validation_passed, f"Data validation failed: {validation_summary}"
    assert validation_summary['validation_passed']
    assert validation_summary['record_count'] == len(sample_property_data)
    
    # Mock the database session
    with patch('etl.pacs_import.db_session', db_session):
        # Execute the ETL function
        import_results = import_pacs_data()
        
        # Verify import metrics
        assert import_results['total_records'] == len(sample_property_data)
        assert import_results['valid_records'] > 0
        assert import_results['invalid_records'] == 0
        
        # Verify data was inserted into the database
        properties = db_session.query(Property).all()
        assert len(properties) == len(sample_property_data)
        
        # Verify specific property was imported correctly
        test_property = db_session.query(Property).filter_by(
            property_id=TEST_PROPERTY_ID).first()
        assert test_property is not None
        assert test_property.address == '123 Main St, Kennewick, WA'
        assert test_property.bedrooms == 3
        assert test_property.bathrooms == 2.5


def test_etl_failure_scenarios(db_session):
    """
    Tests ETL pipeline error handling with various failure scenarios.
    
    This test:
    1. Tests validation with invalid property data
    2. Tests database loading with malformed data
    3. Verifies proper error handling and reporting
    """
    # Create invalid property data
    invalid_data = pd.DataFrame({
        'property_id': [None, 'BENTON-67890', 'BENTON-54321'],  # Missing ID
        'address': ['123 Main St, Kennewick, WA', '456 Oak Ave, Richland, WA', '789 Pine St, Prosser, WA'],
        'bedrooms': [3, -4, 2],  # Invalid bedroom count
        'bathrooms': [2.5, 3.0, 1.0],
        'square_feet': [200, 3100, -1500],  # Invalid square footage
        'lot_size': [0.25, 0.4, 9999999],   # Extreme lot size
        'year_built': [1995, 2050, 1975],   # Future year
        'last_sale_date': ['2020-05-15', 'invalid-date', '2021-02-28'],  # Invalid date
        'last_sale_price': [350000, 425000, 275000],
        'property_type': ['Single Family', 'Single Family', 'Single Family'],
        'latitude': [46.2122, 46.2851, 46.2068],
        'longitude': [-119.1372, -119.2785, -119.7683]
    })
    
    # Test data validation with invalid data
    validation_passed, validation_summary = validate_property_data(invalid_data)
    
    # Verify validation catches the errors
    assert not validation_passed, "Validation should fail with invalid data"
    assert not validation_summary['validation_passed']
    assert validation_summary['record_count'] == len(invalid_data)
    
    # Check for specific validation failures
    categories = validation_summary['categories']
    
    # Verify ID validation issues
    assert categories['unique_ids']['status'] == 'passed', "No duplicate IDs should be found"
    
    # Verify date format validation issues
    assert categories['date_formats']['status'] == 'failed', "Invalid date format should be detected"
    date_issues = categories['date_formats']['issues']
    assert any('invalid_count' in issue for issue in date_issues), "Invalid dates should be reported"
    
    # Verify numeric range validation issues
    assert categories['numeric_ranges']['status'] == 'failed', "Invalid numeric ranges should be detected"
    numeric_issues = categories['numeric_ranges']['issues']
    
    # Check for specific numeric validation issues
    bedroom_issues = [i for i in numeric_issues if i.get('field') == 'bedrooms' and i.get('issue_type') == 'below_minimum']
    assert bedroom_issues, "Negative bedroom count should be detected"
    
    square_feet_issues = [i for i in numeric_issues if i.get('field') == 'square_feet' and i.get('issue_type') == 'below_minimum']
    assert square_feet_issues, "Negative square footage should be detected"
    
    lot_size_issues = [i for i in numeric_issues if i.get('field') == 'lot_size' and i.get('issue_type') == 'above_maximum']
    assert lot_size_issues, "Extremely large lot size should be detected"


# ======================================================================
# API Endpoint Integration Tests
# ======================================================================

def test_valuation_endpoint_with_authentication(authenticated_client, mock_database_session):
    """
    Tests the property valuation API endpoint with authentication.
    
    This test:
    1. Calls the endpoint with a valid API key
    2. Verifies the response status and JSON structure
    3. Validates the property valuation data
    """
    with patch('api.get_db', return_value=mock_database_session):
        # Call the API endpoint
        response = authenticated_client.get(API_ROUTES['valuations'])
        
        # Verify response status
        assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"
        
        # Parse response data
        data = response.json()
        
        # Check response is a list of valuations
        assert isinstance(data, list), "Response should be a list of property valuations"
        assert len(data) > 0, "Response should contain at least one property valuation"
        
        # Verify structure of a property valuation
        valuation = data[0]
        
        # Check required fields in valuation response
        assert 'property_id' in valuation, "Response should contain property_id"
        assert 'estimated_value' in valuation, "Response should contain estimated_value"
        assert 'valuation_date' in valuation, "Response should contain valuation_date"
        assert 'confidence_score' in valuation, "Response should contain confidence_score"
        assert 'model_used' in valuation, "Response should contain model_used"
        
        # Verify data types and ranges
        assert isinstance(valuation['estimated_value'], (int, float)), "estimated_value should be numeric"
        assert isinstance(valuation['confidence_score'], (int, float)), "confidence_score should be numeric"
        assert 0 <= valuation['confidence_score'] <= 1, "confidence_score should be between 0 and 1"


def test_valuation_endpoint_with_query_parameters(authenticated_client, mock_database_session):
    """
    Tests the property valuation API endpoint with various query parameters.
    
    This test:
    1. Calls the endpoint with different query parameters
    2. Verifies parameter handling for filtering, pagination, etc.
    3. Validates the filtered responses
    """
    with patch('api.get_db', return_value=mock_database_session):
        # Test limit parameter
        response = authenticated_client.get(f"{API_ROUTES['valuations']}?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2, "Response should respect the limit parameter"
        
        # Test min_value and max_value parameters
        response = authenticated_client.get(f"{API_ROUTES['valuations']}?min_value=300000&max_value=400000")
        assert response.status_code == 200
        data = response.json()
        for valuation in data:
            assert 300000 <= valuation['estimated_value'] <= 400000, "Response should respect min/max value filters"
        
        # Test property_type parameter
        response = authenticated_client.get(f"{API_ROUTES['valuations']}?property_type=Single Family")
        assert response.status_code == 200
        data = response.json()
        for valuation in data:
            assert 'property_type' in valuation['features_used']
            assert valuation['features_used']['property_type'] == 'Single Family', "Response should filter by property type"


def test_valuation_endpoint_without_authentication(mock_database_session):
    """
    Tests the property valuation API endpoint without authentication.
    
    This test:
    1. Calls the endpoint without an API key
    2. Verifies the authentication failure response
    """
    with patch('api.get_db', return_value=mock_database_session):
        # Call the API endpoint without authentication
        response = client.get(API_ROUTES['valuations'])
        
        # Verify authentication failure
        assert response.status_code == 401, "Unauthenticated request should be rejected"
        
        # Check error message
        data = response.json()
        assert 'detail' in data, "Response should contain error details"
        assert 'message' in data['detail'], "Error details should contain a message"
        assert 'API Key is missing' in data['detail']['message'], "Error message should indicate missing API key"


@patch('api.get_etl_status')
def test_etl_status_endpoint(mock_get_etl_status, authenticated_client, sample_etl_status_data):
    """
    Tests the ETL status API endpoint.
    
    This test:
    1. Mocks the ETL status data
    2. Calls the endpoint with authentication
    3. Verifies the response structure
    """
    # Mock the ETL status response
    mock_get_etl_status.return_value = sample_etl_status_data
    
    # Call the ETL status endpoint
    response = authenticated_client.get(API_ROUTES['etl_status'])
    
    # Verify response status
    assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"
    
    # Parse response data
    data = response.json()
    
    # Check required fields
    assert 'status' in data, "Response should contain status"
    assert 'last_run' in data, "Response should contain last_run"
    assert 'sources_processed' in data, "Response should contain sources_processed"
    assert 'records_processed' in data, "Response should contain records_processed"
    assert 'validation_status' in data, "Response should contain validation_status"
    
    # Check enhanced data quality metrics
    assert 'data_completeness' in data, "Response should contain data_completeness"
    assert 'data_accuracy' in data, "Response should contain data_accuracy"
    assert 'quality_score' in data, "Response should contain quality_score"
    
    # Verify data types and ranges
    assert isinstance(data['data_completeness'], (int, float)), "data_completeness should be numeric"
    assert isinstance(data['quality_score'], (int, float)), "quality_score should be numeric"
    assert 0 <= data['data_completeness'] <= 1, "data_completeness should be between 0 and 1"
    assert 0 <= data['quality_score'] <= 1, "quality_score should be between 0 and 1"
    
    # Verify sources information
    assert isinstance(data['sources_processed'], list), "sources_processed should be a list"
    assert len(data['sources_processed']) > 0, "sources_processed should contain at least one source"
    for source in data['sources_processed']:
        assert 'name' in source, "Source should contain name"
        assert 'status' in source, "Source should contain status"
        assert 'records_processed' in source, "Source should contain records_processed"
        assert 'progress' in source, "Source should contain progress"


@patch('api.get_agent_status')
def test_agent_status_endpoint(mock_get_agent_status, authenticated_client, sample_agent_status_data):
    """
    Tests the agent status API endpoint.
    
    This test:
    1. Mocks the agent status data
    2. Calls the endpoint with authentication
    3. Verifies the response structure
    """
    # Mock the agent status response
    mock_get_agent_status.return_value = sample_agent_status_data
    
    # Call the agent status endpoint
    response = authenticated_client.get(API_ROUTES['agent_status'])
    
    # Verify response status
    assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"
    
    # Parse response data
    data = response.json()
    
    # Check required fields
    assert 'agents' in data, "Response should contain agents"
    assert 'system_status' in data, "Response should contain system_status"
    assert 'total_agents' in data, "Response should contain total_agents"
    assert 'active_agents' in data, "Response should contain active_agents"
    assert 'system_health' in data, "Response should contain system_health"
    
    # Verify agent list structure
    assert isinstance(data['agents'], list), "agents should be a list"
    assert len(data['agents']) > 0, "agents should contain at least one agent"
    
    # Check agent details
    agent = data['agents'][0]
    assert 'agent_id' in agent, "Agent should contain agent_id"
    assert 'name' in agent, "Agent should contain name"
    assert 'status' in agent, "Agent should contain status"
    assert 'last_active' in agent, "Agent should contain last_active"
    assert 'performance_metrics' in agent, "Agent should contain performance_metrics"
    
    # Check enhanced metrics
    assert 'health_score' in agent, "Agent should contain health_score"
    assert 'error_rate' in agent, "Agent should contain error_rate"
    assert 'resource_usage' in agent, "Agent should contain resource_usage"
    
    # Verify data types and ranges
    assert isinstance(agent['health_score'], (int, float)), "health_score should be numeric"
    assert isinstance(agent['error_rate'], (int, float)), "error_rate should be numeric"
    assert 0 <= agent['health_score'] <= 1, "health_score should be between 0 and 1"
    
    # Check system metrics
    assert isinstance(data['system_health'], (int, float)), "system_health should be numeric"
    assert 0 <= data['system_health'] <= 1, "system_health should be between 0 and 1"


def test_api_error_handling_with_missing_resource(authenticated_client, mock_database_session):
    """
    Tests API error handling with a non-existent resource ID.
    
    This test:
    1. Requests a property valuation with an invalid ID
    2. Verifies the 404 status code and error message
    """
    with patch('api.get_db', return_value=mock_database_session):
        # Request a non-existent property
        response = authenticated_client.get(API_ROUTES['property_valuation'].format(property_id='NONEXISTENT-ID'))
        
        # Verify 404 response
        assert response.status_code == 404, "Non-existent resource should return 404"
        
        # Check error message
        data = response.json()
        assert 'detail' in data, "Response should contain error details"
        assert isinstance(data['detail'], dict), "Error details should be a dictionary"
        assert 'message' in data['detail'], "Error details should contain a message"
        assert 'Property not found' in data['detail']['message'], "Error message should indicate resource not found"


def test_api_error_handling_with_invalid_parameters(authenticated_client, mock_database_session):
    """
    Tests API error handling with invalid query parameters.
    
    This test:
    1. Requests valuations with invalid parameter values
    2. Verifies the 400 status code and error message
    """
    with patch('api.get_db', return_value=mock_database_session):
        # Test with invalid numeric parameter
        response = authenticated_client.get(f"{API_ROUTES['valuations']}?min_value=invalid")
        
        # Verify 400 response for invalid type
        assert response.status_code == 422, "Invalid parameter type should return 422"
        
        # Test with out-of-range parameter
        response = authenticated_client.get(f"{API_ROUTES['valuations']}?limit=-10")
        
        # Check error response
        data = response.json()
        assert 'detail' in data, "Response should contain error details"
        assert isinstance(data['detail'], list), "Error details should be a list for validation errors"


# ======================================================================
# Failure Scenario Tests
# ======================================================================

def test_missing_gis_data_handling(authenticated_client, mock_database_session):
    """
    Tests API handling of missing GIS data.
    
    This test:
    1. Simulates a property without GIS data
    2. Verifies the API handles the missing data gracefully
    3. Checks that the response includes appropriate fallback values
    """
    # Create a property with valid info but no GIS data
    property_no_gis = Property(
        property_id='BENTON-NO-GIS',
        address='999 No GIS St, Kennewick, WA',
        city='Kennewick',
        state='WA',
        zip_code='99336',
        bedrooms=3,
        bathrooms=2,
        square_feet=2000,
        year_built=2000,
        property_type='Single Family',
        latitude=46.2122,
        longitude=-119.1372
    )
    mock_database_session.add(property_no_gis)
    mock_database_session.commit()
    
    with patch('api.get_db', return_value=mock_database_session), \
         patch('api.calculate_gis_features', return_value=None):  # Simulate missing GIS data
        
        # Call the valuation endpoint for this property
        response = authenticated_client.get(API_ROUTES['property_valuation'].format(property_id='BENTON-NO-GIS'))
        
        # Should still get a valid response
        assert response.status_code == 200, "API should handle missing GIS data gracefully"
        
        data = response.json()
        
        # Check that valuation still occurred
        assert 'estimated_value' in data, "Response should contain estimated_value even without GIS data"
        assert 'confidence_score' in data, "Response should contain confidence_score"
        
        # Confidence score should be lower due to missing GIS data
        assert data['confidence_score'] < 0.9, "Confidence score should be lower with missing GIS data"
        
        # GIS factors should be null or have default values
        assert 'gis_factors' in data, "Response should contain gis_factors field"
        if data['gis_factors'] is not None:
            assert 'location_quality' in data['gis_factors'], "Response should contain location_quality"
            assert data['gis_factors']['location_quality'] is None or \
                   data['gis_factors']['location_quality'] == 0.5, "Default location quality should be used"


def test_database_connection_error_handling(authenticated_client):
    """
    Tests API handling of database connection errors.
    
    This test:
    1. Mocks a database connection failure
    2. Verifies the API returns a proper 500 error
    3. Checks that the error message is appropriate
    """
    with patch('api.get_db', side_effect=Exception("Database connection error")):
        # Call the valuations endpoint
        response = authenticated_client.get(API_ROUTES['valuations'])
        
        # Verify 500 response
        assert response.status_code == 500, "Database error should return 500"
        
        # Check error message
        data = response.json()
        assert 'detail' in data, "Response should contain error details"
        assert 'message' in data['detail'], "Error details should contain a message"
        assert 'database' in data['detail']['message'].lower(), "Error message should mention database issue"
        assert 'error_code' in data['detail'], "Error details should contain an error code"


def test_invalid_property_data_handling(authenticated_client, mock_database_session):
    """
    Tests how the API handles a request with invalid property data for valuation.
    
    This test:
    1. Creates a property with invalid data (extreme values)
    2. Requests a valuation for this property
    3. Verifies the API handles the invalid data appropriately
    """
    # Create a property with invalid data (extreme values)
    invalid_property = Property(
        property_id='BENTON-INVALID',
        address='888 Invalid Data Rd, Richland, WA',
        city='Richland',
        state='WA',
        zip_code='99352',
        bedrooms=99,  # Extreme value
        bathrooms=50,  # Extreme value
        square_feet=100000,  # Extreme value
        year_built=1800,  # Very old
        property_type='Unknown',
        latitude=46.2851,
        longitude=-119.2785
    )
    mock_database_session.add(invalid_property)
    mock_database_session.commit()
    
    with patch('api.get_db', return_value=mock_database_session):
        # Call the valuation endpoint for this property
        response = authenticated_client.get(API_ROUTES['property_valuation'].format(property_id='BENTON-INVALID'))
        
        # The API should still return a result, but with low confidence
        assert response.status_code == 200, "API should handle invalid property data gracefully"
        
        data = response.json()
        
        # Check for appropriate flags in the response
        assert 'confidence_score' in data, "Response should contain confidence_score"
        assert data['confidence_score'] < 0.5, "Confidence score should be low for invalid data"
        
        # Should include warnings in the response
        assert 'features_used' in data, "Response should contain features_used"
        assert 'warnings' in data, "Response should contain warnings for extreme values"


# ======================================================================
# Run the tests
# ======================================================================

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])