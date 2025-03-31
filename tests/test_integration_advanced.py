"""
Integration tests for the BCBS Values platform.

This module tests the end-to-end functionality of the ETL pipeline,
API endpoints, and error handling scenarios. It validates that data flows
correctly through the system and that the API returns the expected responses.
"""

import os
import sys
import pytest
import json
from unittest import mock
from fastapi.testclient import TestClient
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to the path so we can import the application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules from the application
from api.main import app as fastapi_app
from models import Property, Valuation, ETLStatus
from db.database import get_db_session
import etl.pipeline as etl_pipeline

# Create a test client for the FastAPI application
client = TestClient(fastapi_app)

# Database configuration for tests
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture
def db_session():
    """
    Create a new database session for testing.
    
    This fixture provides an isolated database session for each test,
    ensuring that tests don't interfere with each other's data.
    """
    # Create test database engine
    engine = create_engine(TEST_DATABASE_URL)
    
    # Create tables in the test database
    from models import Base
    Base.metadata.create_all(engine)
    
    # Create a session factory
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create a new session for the test
    session = TestSessionLocal()
    
    try:
        # Provide the session to the test
        yield session
    finally:
        # Close the session after the test
        session.close()
        
        # Drop all tables after the test to ensure a clean state
        Base.metadata.drop_all(engine)


@pytest.fixture
def mock_etl_pipeline():
    """
    Mock the ETL pipeline components for testing.
    
    This fixture replaces actual ETL operations with mocks that return
    predefined test data, allowing tests to run without external dependencies.
    """
    # Mock sample data for MLS properties
    mls_data = pd.DataFrame({
        'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
        'city': ['Richland', 'Kennewick', 'Pasco'],
        'state': ['WA', 'WA', 'WA'],
        'zip_code': ['99352', '99336', '99301'],
        'property_type': ['single_family', 'condo', 'townhouse'],
        'bedrooms': [3, 2, 4],
        'bathrooms': [2.5, 2.0, 3.5],
        'square_feet': [2000, 1500, 2500],
        'year_built': [1980, 2000, 2015],
        'price': [350000, 275000, 425000]
    })
    
    # Mock sample data for NARRPR properties
    narrpr_data = pd.DataFrame({
        'address': ['321 Elm St', '654 Maple Dr', '987 Cedar Ln'],
        'city': ['Richland', 'Kennewick', 'West Richland'],
        'state': ['WA', 'WA', 'WA'],
        'zipcode': ['99352', '99336', '99353'],
        'type': ['single_family', 'multi_family', 'single_family'],
        'beds': [4, 6, 3],
        'baths': [3.0, 4.0, 2.0],
        'sqft': [2200, 3500, 1800],
        'year_constructed': [1995, 1975, 2010],
        'estimated_value': [380000, 450000, 320000]
    })
    
    # Mock sample data for PACS properties
    pacs_data = pd.DataFrame({
        'property_address': ['123 Main St', '789 Pine Rd', '555 Birch Ct'],
        'city_name': ['Richland', 'Pasco', 'Kennewick'],
        'state_code': ['WA', 'WA', 'WA'],
        'postal_code': ['99352', '99301', '99336'],
        'property_category': ['residential', 'residential', 'residential'],
        'bedroom_count': [3, 4, 4],
        'bathroom_count': [2.5, 3.5, 3.0],
        'building_area_sqft': [2000, 2500, 2300],
        'year_built': [1980, 2015, 2005],
        'assessed_value': [340000, 415000, 370000]
    })
    
    # Mock GIS data for spatial features
    gis_data = pd.DataFrame({
        'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', 
                    '654 Maple Dr', '987 Cedar Ln', '555 Birch Ct'],
        'latitude': [46.2804, 46.2112, 46.2393, 46.2854, 46.2054, 46.3054, 46.2312],
        'longitude': [-119.2752, -119.1368, -119.1031, -119.2952, -119.1568, -119.3052, -119.1268],
        'school_score': [8.5, 7.2, 6.8, 8.9, 7.5, 8.2, 7.8],
        'crime_score': [9.2, 8.5, 7.6, 9.0, 8.3, 8.8, 8.6],
        'amenities_score': [8.0, 9.1, 7.5, 7.8, 8.5, 7.2, 8.9]
    })
    
    # Create a mock object for the ETL pipeline
    with mock.patch('etl.pipeline.extract_mls_data', return_value=mls_data), \
         mock.patch('etl.pipeline.extract_narrpr_data', return_value=narrpr_data), \
         mock.patch('etl.pipeline.extract_pacs_data', return_value=pacs_data), \
         mock.patch('etl.pipeline.extract_gis_data', return_value=gis_data), \
         mock.patch('etl.pipeline.transform_data'), \
         mock.patch('etl.pipeline.load_data'):
        
        yield etl_pipeline


def test_etl_pipeline_execution(db_session, mock_etl_pipeline):
    """
    Test the end-to-end execution of the ETL pipeline.
    
    This test verifies that the ETL pipeline successfully extracts data from
    various sources, transforms it, and loads it into the database.
    """
    # Mock the database session to use our test session
    with mock.patch('db.database.get_db_session', return_value=db_session):
        # Execute the ETL pipeline
        etl_pipeline.run_pipeline()
        
        # Check that ETL status records were created
        etl_statuses = db_session.query(ETLStatus).all()
        assert len(etl_statuses) > 0, "ETL status records should be created"
        
        # Verify that all ETL processes completed successfully
        for status in etl_statuses:
            assert status.status == "completed", f"ETL process {status.process_name} failed"
            assert status.records_processed > 0, f"No records processed for {status.process_name}"
        
        # Check that properties were loaded into the database
        properties = db_session.query(Property).all()
        assert len(properties) > 0, "Properties should be loaded into the database"
        
        # Check that valuations were created for the properties
        valuations = db_session.query(Valuation).all()
        assert len(valuations) > 0, "Valuations should be created for properties"
        
        # Verify the properties have the expected fields populated
        for prop in properties:
            assert prop.address, "Property should have an address"
            assert prop.city, "Property should have a city"
            assert prop.state, "Property should have a state"
            assert prop.zip_code, "Property should have a zip code"
            
            # Check if the property has at least one valuation
            property_valuations = [v for v in valuations if v.property_id == prop.id]
            assert len(property_valuations) > 0, f"Property {prop.id} should have at least one valuation"


def test_valuations_api_endpoint():
    """
    Test the /api/valuations endpoint for retrieving property valuations.
    
    This test verifies that the API returns the expected JSON structure and
    that the data matches what's expected from the database.
    """
    # Make a GET request to the valuations endpoint
    response = client.get("/api/valuations")
    
    # Check that the response status code is 200 (OK)
    assert response.status_code == 200, "API should return 200 OK status"
    
    # Parse the JSON response
    data = response.json()
    
    # Verify the response structure
    assert "status" in data, "Response should include status field"
    assert data["status"] == "success", "Status should be 'success'"
    assert "data" in data, "Response should include data field"
    assert "valuations" in data["data"], "Data should include valuations list"
    
    # Check pagination data
    assert "pagination" in data["data"], "Response should include pagination info"
    pagination = data["data"]["pagination"]
    assert "page" in pagination, "Pagination should include page number"
    assert "limit" in pagination, "Pagination should include limit per page"
    assert "total" in pagination, "Pagination should include total count"
    
    # Test with filtering parameters
    filter_response = client.get("/api/valuations?neighborhood=Richland&property_type=single_family")
    assert filter_response.status_code == 200, "Filtered API call should succeed"
    
    filter_data = filter_response.json()
    assert filter_data["status"] == "success", "Filtered response status should be 'success'"
    assert "filters" in filter_data["data"], "Response should include filter parameters"
    
    # Verify filter parameters were applied
    filters = filter_data["data"]["filters"]
    assert filters["neighborhood"] == "Richland", "Neighborhood filter should be applied"
    assert filters["property_type"] == "single_family", "Property type filter should be applied"


def test_etl_status_api_endpoint():
    """
    Test the /api/etl-status endpoint for retrieving ETL pipeline status.
    
    This test verifies that the API returns information about the ETL pipeline
    processes, including their status and timing information.
    """
    # Make a GET request to the ETL status endpoint
    response = client.get("/api/etl-status")
    
    # Check that the response status code is 200 (OK)
    assert response.status_code == 200, "API should return 200 OK status"
    
    # Parse the JSON response
    data = response.json()
    
    # Verify the response structure
    assert "status" in data, "Response should include status field"
    assert data["status"] == "success", "Status should be 'success'"
    assert "data" in data, "Response should include data field"
    assert "etl_processes" in data["data"], "Data should include etl_processes list"
    
    # Check ETL processes data
    etl_processes = data["data"]["etl_processes"]
    assert len(etl_processes) > 0, "Response should include ETL processes"
    
    # Verify each ETL process has the expected fields
    for process in etl_processes:
        assert "id" in process, "ETL process should have an ID"
        assert "name" in process, "ETL process should have a name"
        assert "status" in process, "ETL process should have a status"
        assert "last_run" in process, "ETL process should have a last run timestamp"


def test_agent_status_api_endpoint():
    """
    Test the /api/agent-status endpoint for retrieving agent status.
    
    This test verifies that the API returns information about the system agents,
    including their status and performance metrics.
    """
    # Make a GET request to the agent status endpoint
    response = client.get("/api/agent-status")
    
    # Check that the response status code is 200 (OK)
    assert response.status_code == 200, "API should return 200 OK status"
    
    # Parse the JSON response
    data = response.json()
    
    # Verify the response structure
    assert "status" in data, "Response should include status field"
    assert data["status"] == "success", "Status should be 'success'"
    assert "data" in data, "Response should include data field"
    assert "agents" in data["data"], "Data should include agents list"
    
    # Check agents data
    agents = data["data"]["agents"]
    assert len(agents) > 0, "Response should include agents"
    
    # Verify each agent has the expected fields
    for agent in agents:
        assert "id" in agent, "Agent should have an ID"
        assert "name" in agent, "Agent should have a name"
        assert "status" in agent, "Agent should have a status"
        assert "tasks_completed" in agent, "Agent should have tasks_completed field"
        assert "success_rate" in agent, "Agent should have success_rate field"


def test_error_handling_missing_gis_data(db_session, mock_etl_pipeline):
    """
    Test error handling when GIS data is missing.
    
    This test simulates a scenario where GIS data is unavailable and verifies
    that the system handles the error properly without failing completely.
    """
    # Mock the GIS data extraction to raise an exception
    with mock.patch('etl.pipeline.extract_gis_data', side_effect=Exception("GIS data source unavailable")), \
         mock.patch('db.database.get_db_session', return_value=db_session):
        
        # Execute the ETL pipeline, which should continue despite the GIS error
        etl_pipeline.run_pipeline()
        
        # Check that the ETL status for GIS data processing is marked as "failed"
        etl_status = db_session.query(ETLStatus).filter_by(process_name="gis_data_extraction").first()
        assert etl_status is not None, "ETL status record for GIS extraction should exist"
        assert etl_status.status == "failed", "GIS extraction status should be 'failed'"
        assert "GIS data source unavailable" in etl_status.error_message, "Error message should be recorded"
        
        # Check that other ETL processes completed
        other_statuses = db_session.query(ETLStatus).filter(ETLStatus.process_name != "gis_data_extraction").all()
        for status in other_statuses:
            assert status.status == "completed", f"ETL process {status.process_name} should complete despite GIS failure"
        
        # Verify that properties were still loaded
        properties = db_session.query(Property).all()
        assert len(properties) > 0, "Properties should still be loaded despite GIS failure"
        
        # Verify valuations were created, but might have lower confidence scores
        valuations = db_session.query(Valuation).all()
        assert len(valuations) > 0, "Valuations should still be created despite GIS failure"
        
        # The advanced GIS valuation method should not be used
        advanced_gis_valuations = db_session.query(Valuation).filter_by(valuation_method="advanced_gis").all()
        assert len(advanced_gis_valuations) == 0, "No advanced GIS valuations should be created when GIS data is missing"


def test_api_error_handling():
    """
    Test that the API endpoints handle errors gracefully.
    
    This test verifies that the API returns appropriate error responses when
    invalid requests are made, and that it includes helpful error messages.
    """
    # Test with an invalid neighborhood parameter
    response = client.get("/api/valuations?neighborhood=InvalidNeighborhood")
    assert response.status_code == 200, "API should still return 200 OK for invalid neighborhood"
    
    # The response should indicate no results found
    data = response.json()
    assert data["status"] == "success", "Status should be 'success' even for empty results"
    assert len(data["data"]["valuations"]) == 0, "No valuations should be returned for invalid neighborhood"
    
    # Test with invalid property ID
    response = client.get("/api/property/999999")
    assert response.status_code == 404, "API should return 404 Not Found for invalid property ID"
    
    # Check the error response format
    data = response.json()
    assert data["status"] == "error", "Status should be 'error' for invalid property ID"
    assert "message" in data, "Error response should include message field"
    
    # Test with invalid API endpoint
    response = client.get("/api/invalid-endpoint")
    assert response.status_code == 404, "API should return 404 Not Found for invalid endpoint"


def test_what_if_analysis_endpoint():
    """
    Test the /api/what-if-analysis endpoint for performing what-if analysis.
    
    This test verifies that the API can process adjusted valuation parameters
    and return updated valuation results.
    """
    # Create a test payload with adjusted parameters
    payload = {
        "property_id": 1,
        "original_valuation": 350000,
        "parameters": {
            "location_weight": 1.2,
            "condition_weight": 0.9,
            "market_trend_adjustment": 0.05,
            "cap_rate": 0.045
        }
    }
    
    # Make a POST request to the what-if analysis endpoint
    response = client.post("/api/what-if-analysis", json=payload)
    
    # Check that the response status code is 200 (OK)
    assert response.status_code == 200, "API should return 200 OK status"
    
    # Parse the JSON response
    data = response.json()
    
    # Verify the response structure
    assert "status" in data, "Response should include status field"
    assert data["status"] == "success", "Status should be 'success'"
    assert "data" in data, "Response should include data field"
    assert "original_valuation" in data["data"], "Response should include original valuation"
    assert "adjusted_valuation" in data["data"], "Response should include adjusted valuation"
    assert "parameters" in data["data"], "Response should include the parameters used"
    assert "factors" in data["data"], "Response should include the contributing factors"
    
    # Test with missing payload
    response = client.post("/api/what-if-analysis", json={})
    assert response.status_code == 400, "API should return 400 Bad Request for missing payload"
    
    data = response.json()
    assert data["status"] == "error", "Status should be 'error' for missing payload"
    assert "message" in data, "Error response should include message field"


if __name__ == "__main__":
    pytest.main(["-v", __file__])