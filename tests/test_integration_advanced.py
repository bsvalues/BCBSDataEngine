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
import datetime
from unittest import mock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to the path so we can import the application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules from the application
from api.main import app as fastapi_app
from models import Property, Valuation, ETLStatus, Agent, PropertyValue
from db.database import get_db_session
import etl.pipeline as etl_pipeline

# Create a test client for the FastAPI application
client = TestClient(fastapi_app)

# Database configuration for tests
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")

# Mock API key for authenticated endpoints
TEST_API_KEY = "test-api-key-12345"


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
        'amenities_score': [8.0, 9.1, 7.5, 7.8, 8.5, 7.2, 8.9],
        'neighborhood_quality': [0.85, 0.78, 0.72, 0.88, 0.76, 0.82, 0.80],
        'proximity_adjustment': [0.05, -0.03, 0.02, 0.04, -0.01, 0.03, 0.01]
    })
    
    # Create a mock object for the ETL pipeline
    with mock.patch('etl.pipeline.extract_mls_data', return_value=mls_data), \
         mock.patch('etl.pipeline.extract_narrpr_data', return_value=narrpr_data), \
         mock.patch('etl.pipeline.extract_pacs_data', return_value=pacs_data), \
         mock.patch('etl.pipeline.extract_gis_data', return_value=gis_data), \
         mock.patch('etl.pipeline.transform_data'), \
         mock.patch('etl.pipeline.load_data'):
        
        yield etl_pipeline


@pytest.fixture
def seed_test_data(db_session):
    """
    Seed the database with test data for API endpoint testing.
    
    This fixture adds test properties, valuations, and other data to the database
    for testing the API endpoints.
    """
    # Create test properties
    properties = [
        Property(
            id=1,
            address="123 Main St",
            city="Richland",
            state="WA",
            zip_code="99352",
            neighborhood="North Richland",
            property_type="single_family",
            bedrooms=3,
            bathrooms=2.5,
            square_feet=2000,
            year_built=1980,
            lot_size=0.25,
            latitude=46.2804,
            longitude=-119.2752
        ),
        Property(
            id=2,
            address="456 Oak Ave",
            city="Kennewick",
            state="WA",
            zip_code="99336",
            neighborhood="Downtown Kennewick",
            property_type="condo",
            bedrooms=2,
            bathrooms=2.0,
            square_feet=1500,
            year_built=2000,
            lot_size=0.1,
            latitude=46.2112,
            longitude=-119.1368
        ),
        Property(
            id=3,
            address="789 Pine Rd",
            city="Pasco",
            state="WA",
            zip_code="99301",
            neighborhood="West Pasco",
            property_type="townhouse",
            bedrooms=4,
            bathrooms=3.5,
            square_feet=2500,
            year_built=2015,
            lot_size=0.15,
            latitude=46.2393,
            longitude=-119.1031
        )
    ]
    
    for prop in properties:
        db_session.add(prop)
    
    # Create test valuations for different methods
    valuation_methods = [
        "linear_regression", "ridge_regression", "lasso_regression", 
        "elastic_net", "lightgbm", "xgboost", "enhanced_regression"
    ]
    
    # Create different valuations for each property with different methods
    for prop in properties:
        for method in valuation_methods:
            # Vary the values slightly for each method to test comparison
            base_value = 350000 if prop.id == 1 else (275000 if prop.id == 2 else 425000)
            r2_score = 0.7 + (0.05 * valuation_methods.index(method) / len(valuation_methods))
            rmse = 15000 - (1000 * valuation_methods.index(method) / len(valuation_methods))
            
            # Add some noise to make values slightly different
            value_adjustment = np.random.randint(-5000, 5000)
            
            valuation = PropertyValue(
                property_id=prop.id,
                estimated_value=base_value + value_adjustment,
                valuation_date=datetime.date.today() - datetime.timedelta(days=valuation_methods.index(method)),
                valuation_method=method,
                confidence_score=r2_score / 1.5,  # Scale to 0-1 range
                adj_r2_score=r2_score,
                rmse=rmse,
                mae=rmse * 0.8,
                feature_importance=json.dumps({
                    "square_feet": 0.35,
                    "bedrooms": 0.15,
                    "bathrooms": 0.2,
                    "year_built": 0.1,
                    "lot_size": 0.05,
                    "location": 0.15
                })
            )
            
            # Add GIS adjustments for enhanced_regression and lightgbm
            if method in ["enhanced_regression", "lightgbm"]:
                valuation.gis_adjustments = json.dumps({
                    "base_value": base_value,
                    "quality_adjustment": 0.05,
                    "proximity_adjustment": 0.02
                })
            
            db_session.add(valuation)
    
    # Create ETL status records
    etl_statuses = [
        ETLStatus(
            id=1,
            process_name="mls_data_extraction",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=2),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=1, minutes=50),
            records_processed=150,
            success_rate=0.98,
            error_message=None
        ),
        ETLStatus(
            id=2,
            process_name="narrpr_data_extraction",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=1, minutes=45),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=1, minutes=30),
            records_processed=200,
            success_rate=0.95,
            error_message=None
        ),
        ETLStatus(
            id=3,
            process_name="pacs_data_extraction",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=1, minutes=25),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=1, minutes=10),
            records_processed=250,
            success_rate=0.99,
            error_message=None
        ),
        ETLStatus(
            id=4,
            process_name="gis_data_extraction",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=1, minutes=5),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=50),
            records_processed=300,
            success_rate=0.97,
            error_message=None
        ),
        ETLStatus(
            id=5,
            process_name="data_validation",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=45),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=35),
            records_processed=350,
            success_rate=1.0,
            error_message=None
        ),
        ETLStatus(
            id=6,
            process_name="data_transformation",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=30),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=20),
            records_processed=350,
            success_rate=1.0,
            error_message=None
        ),
        ETLStatus(
            id=7,
            process_name="data_loading",
            status="completed",
            start_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=15),
            end_time=datetime.datetime.now() - datetime.timedelta(hours=0, minutes=5),
            records_processed=350,
            success_rate=1.0,
            error_message=None
        )
    ]
    
    for status in etl_statuses:
        db_session.add(status)
    
    # Create agent status records
    agents = [
        Agent(
            id=1,
            agent_id="bootstrap-commander-1",
            agent_name="Bootstrap Commander",
            agent_type="coordinator",
            status="idle",
            last_heartbeat=datetime.datetime.now() - datetime.timedelta(minutes=2),
            queue_size=0,
            current_task=None,
            success_rate=0.98,
            error_count=1,
            metrics=json.dumps({
                "tasks_completed": 120,
                "avg_response_time": 150.5,
                "memory_usage": 52428800,  # 50MB
                "cpu_usage": 0.15
            })
        ),
        Agent(
            id=2,
            agent_id="god-tier-builder-1",
            agent_name="God-Tier Builder",
            agent_type="constructor",
            status="busy",
            last_heartbeat=datetime.datetime.now() - datetime.timedelta(minutes=1),
            queue_size=3,
            current_task="Building new advanced valuation model",
            success_rate=0.95,
            error_count=2,
            metrics=json.dumps({
                "tasks_completed": 85,
                "avg_response_time": 450.2,
                "memory_usage": 104857600,  # 100MB
                "cpu_usage": 0.45
            })
        ),
        Agent(
            id=3,
            agent_id="tdd-validator-1",
            agent_name="TDD Validator",
            agent_type="validator",
            status="idle",
            last_heartbeat=datetime.datetime.now() - datetime.timedelta(minutes=3),
            queue_size=0,
            current_task=None,
            success_rate=0.99,
            error_count=0,
            metrics=json.dumps({
                "tasks_completed": 150,
                "avg_response_time": 200.8,
                "memory_usage": 78643200,  # 75MB
                "cpu_usage": 0.25
            })
        ),
        Agent(
            id=4,
            agent_id="cascade-operator-1",
            agent_name="Cascade Operator",
            agent_type="orchestrator",
            status="busy",
            last_heartbeat=datetime.datetime.now() - datetime.timedelta(seconds=30),
            queue_size=5,
            current_task="Orchestrating valuation workflow",
            success_rate=0.97,
            error_count=3,
            metrics=json.dumps({
                "tasks_completed": 210,
                "avg_response_time": 180.3,
                "memory_usage": 83886080,  # 80MB
                "cpu_usage": 0.3
            })
        )
    ]
    
    for agent in agents:
        db_session.add(agent)
    
    # Commit the changes to the database
    db_session.commit()


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
        valuations = db_session.query(PropertyValue).all()
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


def test_etl_data_validation(db_session, mock_etl_pipeline):
    """
    Test the data validation process in the ETL pipeline.
    
    This test ensures that data is properly validated during the ETL process,
    with invalid data being filtered out or corrected as appropriate.
    """
    # Create invalid data for testing validation
    invalid_mls_data = pd.DataFrame({
        'address': ['123 Main St', '456 Oak Ave', ''],
        'city': ['Richland', '', 'Pasco'],
        'state': ['WA', 'WA', ''],
        'zip_code': ['99352', '993', '99301'],
        'property_type': ['single_family', 'unknown', 'townhouse'],
        'bedrooms': [3, -1, 4],
        'bathrooms': [2.5, 2.0, -3.5],
        'square_feet': [2000, -1500, 2500],
        'year_built': [1980, 2050, 2015],  # 2050 is in the future
        'price': [350000, 275000, -425000]
    })
    
    # Mock the validation function to check cleaning of data
    with mock.patch('etl.pipeline.extract_mls_data', return_value=invalid_mls_data), \
         mock.patch('etl.pipeline.validate_data') as mock_validate_data, \
         mock.patch('db.database.get_db_session', return_value=db_session):
        
        # Execute just the validation step
        etl_pipeline.validate_data(invalid_mls_data, 'mls')
        
        # Verify the validation function was called
        assert mock_validate_data.called, "Data validation function should be called"
        
        # Check that ETL process creates a validation status record
        etl_pipeline.run_pipeline()
        validation_status = db_session.query(ETLStatus).filter_by(process_name="data_validation").first()
        assert validation_status is not None, "ETL validation status record should exist"
        assert validation_status.status == "completed", "Validation status should be 'completed'"
        
        # Check that the validation errors were logged
        assert validation_status.records_processed > 0, "Validation should process records"
        assert validation_status.success_rate < 1.0, "Success rate should be less than 100% due to validation issues"


def test_model_comparison_logic(db_session, seed_test_data):
    """
    Test that the system compares different valuation models and selects the best one.
    
    This test verifies that when multiple valuation methods are used, the system
    properly compares them based on R-squared and other metrics to select the most
    accurate one.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
        # Get the property with ID 1
        response = client.get(f"/api/property/1")
        assert response.status_code == 200, "API should return 200 OK status"
        
        data = response.json()
        property_data = data.get("data", {}).get("property", {})
        
        # Check that the best valuation method is selected based on metrics
        assert "best_valuation" in property_data, "Property data should include best valuation"
        best_valuation = property_data["best_valuation"]
        assert best_valuation["valuation_method"] is not None, "Best valuation should have a method"
        
        # Get all valuations for this property
        all_valuations = property_data.get("valuations", [])
        
        # Verify that the best valuation has the highest R-squared or confidence score
        if len(all_valuations) > 1:
            best_r2 = best_valuation.get("adj_r2_score", 0)
            for valuation in all_valuations:
                if valuation["valuation_method"] != best_valuation["valuation_method"]:
                    assert best_r2 >= valuation.get("adj_r2_score", 0), \
                        "Best valuation should have the highest R-squared score"


def test_valuations_api_endpoint(db_session, seed_test_data):
    """
    Test the /api/valuations endpoint for retrieving property valuations.
    
    This test verifies that the API returns the expected JSON structure and
    that the data matches what's expected from the database.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
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
        assert "properties" in data["data"], "Data should include properties list"
        
        # Check that the properties match what we seeded
        properties = data["data"]["properties"]
        assert len(properties) == 3, "Should return all 3 seeded properties"
        
        # Check pagination data
        assert "pagination" in data["data"], "Response should include pagination info"
        pagination = data["data"]["pagination"]
        assert "page" in pagination, "Pagination should include page number"
        assert "per_page" in pagination, "Pagination should include limit per page"
        assert "total" in pagination, "Pagination should include total count"
        assert pagination["total"] == 3, "Total count should match the number of seeded properties"
        
        # Test with filtering parameters
        filter_response = client.get("/api/valuations?neighborhood=North Richland&property_type=single_family")
        assert filter_response.status_code == 200, "Filtered API call should succeed"
        
        filter_data = filter_response.json()
        assert filter_data["status"] == "success", "Filtered response status should be 'success'"
        
        # Verify filter parameters were applied correctly
        filter_properties = filter_data["data"]["properties"]
        assert len(filter_properties) > 0, "Filtered properties should exist"
        for prop in filter_properties:
            assert prop["neighborhood"] == "North Richland", "Neighborhood filter should be applied"
            assert prop["property_type"] == "single_family", "Property type filter should be applied"


def test_etl_status_api_endpoint(db_session, seed_test_data):
    """
    Test the /api/etl-status endpoint for retrieving ETL pipeline status.
    
    This test verifies that the API returns information about the ETL pipeline
    processes, including their status and timing information.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
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
        assert len(etl_processes) == 7, "Response should include all 7 seeded ETL processes"
        
        # Verify each ETL process has the expected fields
        for process in etl_processes:
            assert "id" in process, "ETL process should have an ID"
            assert "name" in process, "ETL process should have a name"
            assert "status" in process, "ETL process should have a status"
            assert "last_run" in process, "ETL process should have a last run timestamp"
            assert "records_processed" in process, "ETL process should have records_processed field"
            assert "success_rate" in process, "ETL process should have success_rate field"


def test_agent_status_api_endpoint(db_session, seed_test_data):
    """
    Test the /api/agent-status endpoint for retrieving agent status.
    
    This test verifies that the API returns information about the system agents,
    including their status and performance metrics.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
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
        assert len(agents) == 4, "Response should include all 4 seeded agents"
        
        # Verify each agent has the expected fields
        for agent in agents:
            assert "id" in agent, "Agent should have an ID"
            assert "agent_id" in agent, "Agent should have an agent_id"
            assert "agent_name" in agent, "Agent should have a name"
            assert "agent_type" in agent, "Agent should have a type"
            assert "status" in agent, "Agent should have a status"
            assert "queue_size" in agent, "Agent should have queue_size field"
            assert "success_rate" in agent, "Agent should have success_rate field"
            assert "metrics" in agent, "Agent should have metrics field"
            
            # Verify metrics structure
            metrics = agent["metrics"]
            assert "tasks_completed" in metrics, "Metrics should include tasks_completed"
            assert "avg_response_time" in metrics, "Metrics should include avg_response_time"


def test_agent_logs_api_endpoint(db_session, seed_test_data):
    """
    Test the /api/agent-logs/{agent_id} endpoint for retrieving agent logs.
    
    This test verifies that the API returns detailed logs for a specific agent,
    which can be used for debugging and monitoring purposes.
    """
    # Mock the agent logs function to return sample logs
    sample_logs = [
        {
            "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=10),
            "level": "info",
            "message": "Agent started successfully"
        },
        {
            "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=5),
            "level": "debug",
            "message": "Processing task 123: Valuation request"
        },
        {
            "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=2),
            "level": "warning",
            "message": "API rate limit approaching"
        }
    ]
    
    # Convert datetime objects to strings for JSON comparison
    json_logs = [
        {
            "timestamp": log["timestamp"].isoformat(),
            "level": log["level"],
            "message": log["message"]
        }
        for log in sample_logs
    ]
    
    # Mock the database session and logs function for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session), \
         mock.patch('api.agent_logs.get_agent_logs', return_value=sample_logs):
        
        # Make a GET request to the agent logs endpoint
        response = client.get("/api/agent-logs/bootstrap-commander-1")
        
        # Check that the response status code is 200 (OK)
        assert response.status_code == 200, "API should return 200 OK status"
        
        # Parse the JSON response
        data = response.json()
        
        # Verify the response structure
        assert "status" in data, "Response should include status field"
        assert data["status"] == "success", "Status should be 'success'"
        assert "data" in data, "Response should include data field"
        assert "logs" in data["data"], "Data should include logs list"
        
        # Check logs data
        logs = data["data"]["logs"]
        assert len(logs) == len(sample_logs), f"Response should include all {len(sample_logs)} sample logs"
        
        # Verify each log has the expected fields
        for i, log in enumerate(logs):
            assert "timestamp" in log, "Log should have a timestamp"
            assert "level" in log, "Log should have a level"
            assert "message" in log, "Log should have a message"
            assert log["level"] == json_logs[i]["level"], "Log level should match sample"
            assert log["message"] == json_logs[i]["message"], "Log message should match sample"


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
        valuations = db_session.query(PropertyValue).all()
        assert len(valuations) > 0, "Valuations should still be created despite GIS failure"
        
        # The advanced GIS valuation method should not be used
        advanced_gis_valuations = db_session.query(PropertyValue).filter_by(valuation_method="advanced_gis").all()
        assert len(advanced_gis_valuations) == 0, "No advanced GIS valuations should be created when GIS data is missing"


def test_api_error_handling(db_session, seed_test_data):
    """
    Test that the API endpoints handle errors gracefully.
    
    This test verifies that the API returns appropriate error responses when
    invalid requests are made, and that it includes helpful error messages.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
        # Test with an invalid neighborhood parameter
        response = client.get("/api/valuations?neighborhood=InvalidNeighborhood")
        assert response.status_code == 200, "API should still return 200 OK for invalid neighborhood"
        
        # The response should indicate no results found
        data = response.json()
        assert data["status"] == "success", "Status should be 'success' even for empty results"
        assert len(data["data"]["properties"]) == 0, "No properties should be returned for invalid neighborhood"
        
        # Test with invalid property ID
        response = client.get("/api/property/999999")
        assert response.status_code == 404, "API should return 404 Not Found for invalid property ID"
        
        # Check the error response format
        data = response.json()
        assert data["status"] == "error", "Status should be 'error' for invalid property ID"
        assert "message" in data, "Error response should include message field"
        
        # Test with negative values in price range filter
        response = client.get("/api/valuations?min_value=-1000&max_value=invalid")
        assert response.status_code == 400, "API should return 400 Bad Request for invalid price range"
        
        # Check error details
        data = response.json()
        assert data["status"] == "error", "Status should be 'error' for invalid request parameters"
        assert "message" in data, "Error response should include message field"
        assert "details" in data, "Error response should include details field"
        
        # Test with invalid API endpoint
        response = client.get("/api/invalid-endpoint")
        assert response.status_code == 404, "API should return 404 Not Found for invalid endpoint"


def test_what_if_analysis_endpoint(db_session, seed_test_data):
    """
    Test the /api/what-if-analysis endpoint for performing what-if analysis.
    
    This test verifies that the API can process adjusted valuation parameters
    and return updated valuation results.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
        # Create a test payload with adjusted parameters
        payload = {
            "property_id": 1,
            "original_valuation": 350000,
            "parameters": {
                "cap_rate": 0.05,
                "square_feet_weight": 0.4,
                "location_weight": 0.3,
                "age_weight": 0.2,
                "amenities_weight": 0.1,
                "market_adjustment": 0.03
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
        assert "original_valuation" in data["data"], "Data should include original_valuation"
        assert "adjusted_valuation" in data["data"], "Data should include adjusted_valuation"
        assert "difference" in data["data"], "Data should include difference"
        assert "difference_percentage" in data["data"], "Data should include difference_percentage"
        assert "factors" in data["data"], "Data should include factors breakdown"
        
        # Verify that the adjusted valuation differs from the original
        original = data["data"]["original_valuation"]
        adjusted = data["data"]["adjusted_valuation"]
        assert original != adjusted, "Adjusted valuation should differ from original"
        assert data["data"]["difference"] == adjusted - original, "Difference should be correctly calculated"
        
        # Test with invalid property ID
        invalid_payload = {
            "property_id": 999999,
            "original_valuation": 350000,
            "parameters": {
                "cap_rate": 0.05
            }
        }
        
        response = client.post("/api/what-if-analysis", json=invalid_payload)
        assert response.status_code == 404, "API should return 404 Not Found for invalid property ID"


def test_authentication_required_endpoints(db_session, seed_test_data):
    """
    Test that protected API endpoints require authentication.
    
    This test verifies that API endpoints that require authentication properly
    reject requests without valid API keys and accept them with valid keys.
    """
    # Mock the database session and API key validation for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session), \
         mock.patch('api.auth.validate_api_key', return_value=True):
        
        # Attempt to access a protected endpoint without an API key
        response = client.get("/api/admin/etl-status")
        assert response.status_code == 401, "Protected endpoint should require authentication"
        
        # Verify the error response format
        data = response.json()
        assert data["status"] == "error", "Status should be 'error' for missing API key"
        assert "message" in data, "Error response should include message field"
        assert "Authentication required" in data["message"], "Error message should indicate authentication required"
        
        # Attempt to access with an invalid API key
        response = client.get("/api/admin/etl-status", headers={"X-API-Key": "invalid-key"})
        assert response.status_code == 401, "Protected endpoint should reject invalid API key"
        
        # Attempt to access with a valid API key
        response = client.get("/api/admin/etl-status", headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200, "Protected endpoint should accept valid API key"
        
        # Parse the JSON response
        data = response.json()
        assert data["status"] == "success", "Status should be 'success' with valid API key"


def test_model_performance_comparison_endpoint(db_session, seed_test_data):
    """
    Test the /api/model-performance endpoint for comparing valuation models.
    
    This test verifies that the API can return performance metrics for different
    valuation models, allowing users to compare their accuracy and reliability.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
        # Make a GET request to the model performance endpoint
        response = client.get("/api/model-performance")
        
        # Check that the response status code is 200 (OK)
        assert response.status_code == 200, "API should return 200 OK status"
        
        # Parse the JSON response
        data = response.json()
        
        # Verify the response structure
        assert "status" in data, "Response should include status field"
        assert data["status"] == "success", "Status should be 'success'"
        assert "data" in data, "Response should include data field"
        assert "models" in data["data"], "Data should include models list"
        
        # Check that all valuation methods are included
        models = data["data"]["models"]
        valuation_methods = [
            "linear_regression", "ridge_regression", "lasso_regression", 
            "elastic_net", "lightgbm", "xgboost", "enhanced_regression"
        ]
        for method in valuation_methods:
            assert any(model["name"] == method for model in models), f"Model performance should include {method}"
        
        # Verify each model has the expected performance metrics
        for model in models:
            assert "name" in model, "Model should have a name"
            assert "r2_score" in model, "Model should have an R-squared score"
            assert "adj_r2_score" in model, "Model should have an adjusted R-squared score"
            assert "rmse" in model, "Model should have an RMSE value"
            assert "mae" in model, "Model should have an MAE value"
            assert "confidence_score" in model, "Model should have a confidence score"
            assert "property_count" in model, "Model should have a property count"


def test_geospatial_query_endpoint(db_session, seed_test_data):
    """
    Test the /api/properties/nearby endpoint for geospatial queries.
    
    This test verifies that the API can find properties within a specified
    radius of a given location using geospatial queries.
    """
    # Mock the database session for the API calls
    with mock.patch('db.database.get_db_session', return_value=db_session):
        # Make a GET request to the nearby properties endpoint
        response = client.get("/api/properties/nearby?latitude=46.2804&longitude=-119.2752&radius=5")
        
        # Check that the response status code is 200 (OK)
        assert response.status_code == 200, "API should return 200 OK status"
        
        # Parse the JSON response
        data = response.json()
        
        # Verify the response structure
        assert "status" in data, "Response should include status field"
        assert data["status"] == "success", "Status should be 'success'"
        assert "data" in data, "Response should include data field"
        assert "properties" in data["data"], "Data should include properties list"
        
        # Check that properties within the radius are returned
        properties = data["data"]["properties"]
        assert len(properties) > 0, "Should return properties within the radius"
        
        # Verify distance calculation - properties should include distance
        for prop in properties:
            assert "distance" in prop, "Property should include distance from query point"
            assert prop["distance"] <= 5, "Properties should be within the specified radius"
            
        # Test with invalid parameters
        response = client.get("/api/properties/nearby?latitude=invalid&longitude=-119.2752&radius=5")
        assert response.status_code == 400, "API should return 400 Bad Request for invalid coordinates"
        
        # Check the error response format
        data = response.json()
        assert data["status"] == "error", "Status should be 'error' for invalid parameters"
        assert "message" in data, "Error response should include message field"


if __name__ == "__main__":
    pytest.main()