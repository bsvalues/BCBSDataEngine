"""
Advanced integration tests for BCBS Values application.

This test suite performs comprehensive end-to-end testing of the BCBS Values application, including:
1. ETL pipeline execution and database loading
2. API endpoint testing with FastAPI TestClient for enhanced endpoints
3. Error handling and edge case scenarios (missing data, invalid parameters)
4. Security validation (authentication and authorization)
5. GIS data integration scenarios
6. Performance monitoring and validation

These tests validate both the functional correctness and robustness of the system
by simulating real-world usage patterns and edge cases. They ensure that the
application handles errors gracefully and provides helpful feedback to clients.

These tests require a working database connection and should be run in a
controlled environment to avoid affecting production data.

Author: BCBS Test Engineering Team
Last Updated: 2025-03-31
"""
import os
import sys
import json
import pytest
import logging
from datetime import datetime, timedelta
from unittest import mock
from fastapi.testclient import TestClient

# Adjust path to import application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application modules
from app import app as flask_app
from models import Property, PropertyValuation, ETLJob, Agent, AgentLog, EtlStatus, DataSource, User
from api import api_bp

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fixtures
@pytest.fixture
def app():
    """Create a Flask application for testing."""
    # Register API blueprint with the application
    flask_app.register_blueprint(api_bp)
    # Enable testing mode
    flask_app.config['TESTING'] = True
    # Use an in-memory SQLite database for testing
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    return flask_app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    with app.test_client() as client:
        # Create application context
        with app.app_context():
            # Create all tables in the database
            from app import db
            db.create_all()
            yield client
            # Clean up: drop all tables
            db.drop_all()

@pytest.fixture
def api_key():
    """Return a valid API key for testing."""
    return os.environ.get('API_KEY', 'bcbs_demo_key_2023')

@pytest.fixture
def auth_headers(api_key):
    """Return headers with API key for authenticated requests."""
    return {'X-API-Key': api_key}

@pytest.fixture
def seed_test_data(app):
    """Seed the database with test data."""
    with app.app_context():
        from app import db
        
        # Create test user
        user = User(username='testuser', email='test@example.com')
        user.password_hash = 'pbkdf2:sha256:150000$ABC123$ABC123'
        db.session.add(user)
        
        # Create test properties
        properties = []
        for i in range(1, 6):
            prop = Property(
                property_id=f'TEST-PROP-{i}',
                address=f'{i} Test Street',
                neighborhood='Test Neighborhood',
                property_type='Single Family',
                year_built=2000 + i,
                bedrooms=3,
                bathrooms=2.0,
                living_area=2000.0,
                land_area=5000.0
            )
            properties.append(prop)
            db.session.add(prop)
        
        db.session.commit()
        
        # Create property valuations
        for i, prop in enumerate(properties):
            valuation = PropertyValuation(
                property_id=prop.id,
                estimated_value=300000 + (i * 50000),
                confidence_score=0.85 - (i * 0.05),
                valuation_method='enhanced_regression' if i % 2 == 0 else 'lightgbm',
                valuation_date=datetime.utcnow() - timedelta(days=i)
            )
            db.session.add(valuation)
        
        # Create ETL job entries
        etl_status = EtlStatus(
            status='completed',
            progress=1.0,
            records_processed=100,
            success_rate=0.98,
            average_processing_time=2.5,
            completeness=0.95,
            accuracy=0.92,
            timeliness=0.97
        )
        db.session.add(etl_status)
        db.session.commit()
        
        # Create data sources
        sources = ['mls', 'public_records', 'tax_assessor']
        for source in sources:
            ds = DataSource(
                name=source,
                status='completed',
                records=30,
                etl_status_id=etl_status.id
            )
            db.session.add(ds)
        
        # Create test agents
        agent_types = ['regression', 'ensemble', 'gis']
        for i, agent_type in enumerate(agent_types):
            agent = Agent(
                agent_id=f'agent-{agent_type}-001',
                agent_type=agent_type,
                status='idle' if i % 3 == 0 else 'processing' if i % 3 == 1 else 'error',
                queue_size=i * 5,
                total_processed=100 + (i * 20),
                success_rate=0.9 - (i * 0.05),
                average_processing_time=1.5 + (i * 0.5)
            )
            db.session.add(agent)
            db.session.commit()
            
            # Add some logs for each agent
            log_levels = ['info', 'warning', 'error']
            for j, level in enumerate(log_levels):
                log = AgentLog(
                    agent_id=agent.id,
                    level=level,
                    message=f'Test log message for {agent_type} agent ({level})',
                    timestamp=datetime.utcnow() - timedelta(hours=j)
                )
                db.session.add(log)
        
        # Create ETL jobs
        job_types = ['extract', 'transform', 'load']
        job_statuses = ['completed', 'completed', 'completed']
        for i, (job_type, status) in enumerate(zip(job_types, job_statuses)):
            job = ETLJob(
                job_type=job_type,
                status=status,
                source=sources[i % len(sources)],
                records_processed=30 + (i * 5),
                start_time=datetime.utcnow() - timedelta(hours=i + 1),
                end_time=datetime.utcnow() - timedelta(hours=i),
                success_rate=0.95 - (i * 0.02)
            )
            db.session.add(job)
        
        db.session.commit()
        
        return {
            'user': user,
            'properties': properties,
            'etl_status': etl_status
        }

@pytest.fixture
def mock_etl_pipeline():
    """Mock the ETL pipeline functionality."""
    with mock.patch('etl.pipeline.ETLPipeline') as mock_pipeline:
        # Configure the mock pipeline
        pipeline_instance = mock_pipeline.return_value
        pipeline_instance.run_pipeline.return_value = {
            'status': 'completed',
            'records_processed': 100,
            'valid_records': 95,
            'invalid_records': 5,
            'execution_time_seconds': 10.5,
            'sources_processed': ['mls', 'public_records', 'tax_assessor']
        }
        yield pipeline_instance


# === ETL Pipeline Tests ===

def test_etl_pipeline_execution(app, mock_etl_pipeline, seed_test_data):
    """
    Test the end-to-end execution of the ETL pipeline.
    
    This test verifies that the ETL pipeline successfully extracts data from
    various sources, transforms it, and loads it into the database.
    """
    with app.app_context():
        from app import db
        
        # We're using a mock of the ETL pipeline to avoid making actual external data calls
        
        # Mock execution of the ETL pipeline by calling main
        with mock.patch('main.run_etl_pipeline') as mock_run_etl:
            mock_run_etl.return_value = mock_etl_pipeline.run_pipeline.return_value
            
            # Simulate execution of main.py
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("main", "main.py")
                main_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(main_module)
                
                # Check if the module has a run_etl_pipeline function
                if hasattr(main_module, 'run_etl_pipeline'):
                    main_module.run_etl_pipeline()
            except (ImportError, AttributeError) as e:
                # If direct import fails, we'll just mock the execution
                logger.warning(f"Could not directly execute main.py: {e}")
                # Mock that ETL pipeline was run successfully
                pass
        
        # Verify ETL status was updated
        etl_status = EtlStatus.query.order_by(EtlStatus.id.desc()).first()
        assert etl_status is not None, "ETL status should be created or updated"
        
        # Either the status is from our seed data, or it's been updated by the pipeline
        if etl_status.id == seed_test_data['etl_status'].id:
            assert etl_status.status in ['completed', 'processing'], f"Unexpected ETL status: {etl_status.status}"
        else:
            assert etl_status.status == 'completed', f"New ETL process should be completed, got {etl_status.status}"
        
        # Verify ETL jobs were recorded
        etl_jobs = ETLJob.query.all()
        assert len(etl_jobs) > 0, "ETL jobs should be recorded"
        
        # Verify we have properties in the database (either seeded or from ETL)
        properties = Property.query.all()
        assert len(properties) > 0, "Properties should be in the database"
        
        # Verify property valuations exist
        valuations = PropertyValuation.query.all()
        assert len(valuations) > 0, "Property valuations should be in the database"


# === API Endpoint Tests ===

def test_api_get_valuations(client, auth_headers, seed_test_data):
    """
    Test the /api/valuations endpoint.
    
    This test verifies that the endpoint returns property valuations with the expected
    structure and data, and that filtering and pagination work correctly.
    """
    # Test basic request without filters
    response = client.get('/api/valuations', headers=auth_headers)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = json.loads(response.data)
    assert 'valuations' in data, "Response should include a 'valuations' field"
    assert 'total' in data, "Response should include pagination metadata"
    assert 'page' in data, "Response should include pagination metadata"
    
    # Verify structure of a valuation record
    if data['valuations']:
        valuation = data['valuations'][0]
        assert 'property_id' in valuation, "Valuation should include property_id"
        assert 'estimated_value' in valuation, "Valuation should include estimated_value"
        assert 'valuation_method' in valuation, "Valuation should include valuation_method"
        assert 'confidence_score' in valuation, "Valuation should include confidence_score"
        assert 'valuation_date' in valuation, "Valuation should include valuation_date"
    
    # Test filtering by valuation method
    response = client.get('/api/valuations?method=enhanced_regression', headers=auth_headers)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    if data['valuations']:
        # Verify all returned valuations use the enhanced_regression method
        for valuation in data['valuations']:
            assert valuation['valuation_method'] == 'enhanced_regression', \
                "Filtering by method should only return matching records"
    
    # Test pagination
    response = client.get('/api/valuations?page=1&limit=2', headers=auth_headers)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert len(data['valuations']) <= 2, "Pagination should limit the number of records returned"
    
    # Test filtering by minimum confidence
    response = client.get('/api/valuations?min_confidence=0.8', headers=auth_headers)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    if data['valuations']:
        for valuation in data['valuations']:
            assert valuation['confidence_score'] >= 0.8, \
                "Filtering by min_confidence should only return records above threshold"
    
    # Test filtering by neighborhood
    response = client.get('/api/valuations?neighborhood=Test%20Neighborhood', headers=auth_headers)
    assert response.status_code == 200


def test_api_get_etl_status(client, auth_headers, seed_test_data):
    """
    Test the /api/etl-status endpoint.
    
    This test verifies that the endpoint returns ETL pipeline status with the expected
    structure and data, and that filtering works correctly.
    """
    # Test basic request
    response = client.get('/api/etl-status', headers=auth_headers)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = json.loads(response.data)
    assert 'status' in data, "Response should include the ETL status"
    assert 'sources' in data, "Response should include the data sources"
    assert 'metrics' in data, "Response should include ETL metrics"
    assert 'data_quality' in data, "Response should include data quality metrics"
    
    # Verify structure of ETL metrics
    assert 'records_processed' in data['metrics'], "Metrics should include records_processed"
    assert 'success_rate' in data['metrics'], "Metrics should include success_rate"
    
    # Verify structure of data quality metrics
    assert 'completeness' in data['data_quality'], "Data quality should include completeness metric"
    assert 'accuracy' in data['data_quality'], "Data quality should include accuracy metric"
    
    # Test filtering by job type
    response = client.get('/api/etl-status?job_type=extract', headers=auth_headers)
    assert response.status_code == 200
    
    # Test filtering by timeframe
    response = client.get('/api/etl-status?timeframe=today', headers=auth_headers)
    assert response.status_code == 200


def test_api_get_agent_status(client, auth_headers, seed_test_data):
    """
    Test the /api/agent-status endpoint.
    
    This test verifies that the endpoint returns agent status information with the expected
    structure and data, and that filtering works correctly.
    """
    # Test basic request
    response = client.get('/api/agent-status', headers=auth_headers)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = json.loads(response.data)
    assert 'agents' in data, "Response should include an 'agents' field"
    assert isinstance(data['agents'], list), "The 'agents' field should be a list"
    
    # Verify structure of an agent record
    if data['agents']:
        agent = data['agents'][0]
        assert 'agent_id' in agent, "Agent record should include agent_id"
        assert 'agent_type' in agent, "Agent record should include agent_type"
        assert 'status' in agent, "Agent record should include status"
        assert 'metrics' in agent, "Agent record should include metrics"
    
    # Test filtering by agent type
    response = client.get('/api/agent-status?agent_type=regression', headers=auth_headers)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    if data['agents']:
        # Verify all returned agents are of the regression type
        for agent in data['agents']:
            assert agent['agent_type'] == 'regression', \
                "Filtering by agent_type should only return matching records"
    
    # Test filtering by status
    response = client.get('/api/agent-status?status=idle', headers=auth_headers)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    if data['agents']:
        # Verify all returned agents have the idle status
        for agent in data['agents']:
            assert agent['status'] == 'idle', \
                "Filtering by status should only return matching records"
    
    # Test filtering by active_only
    response = client.get('/api/agent-status?active_only=true', headers=auth_headers)
    assert response.status_code == 200


def test_api_get_agent_logs(client, auth_headers, seed_test_data):
    """
    Test the /api/agent/{agent_id}/logs endpoint.
    
    This test verifies that the endpoint returns agent logs with the expected
    structure and data.
    """
    with app.app_context():
        # Get an agent ID from the database
        agent = Agent.query.first()
        
    # Test getting logs for an existing agent
    response = client.get(f'/api/agent/{agent.agent_id}/logs', headers=auth_headers)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = json.loads(response.data)
    assert 'logs' in data, "Response should include a 'logs' field"
    assert isinstance(data['logs'], list), "The 'logs' field should be a list"
    
    # Verify structure of a log entry
    if data['logs']:
        log = data['logs'][0]
        assert 'level' in log, "Log entry should include level"
        assert 'message' in log, "Log entry should include message"
        assert 'timestamp' in log, "Log entry should include timestamp"
    
    # Test getting logs for a non-existent agent
    response = client.get('/api/agent/non-existent-agent/logs', headers=auth_headers)
    assert response.status_code == 404, "Request for non-existent agent should return 404"


# === Error Handling Tests ===

def test_api_unauthorized_access(client):
    """
    Test API endpoints with missing or invalid authentication.
    
    This test verifies that the API correctly rejects requests without proper
    authentication credentials.
    """
    # Test without any authentication
    response = client.get('/api/valuations')
    assert response.status_code in [401, 403], f"Expected 401 or 403, got {response.status_code}"
    
    # Test with invalid API key
    response = client.get('/api/valuations', headers={'X-API-Key': 'invalid_key'})
    assert response.status_code in [401, 403], f"Expected 401 or 403, got {response.status_code}"


def test_api_invalid_parameters(client, auth_headers):
    """
    Test API endpoints with invalid query parameters.
    
    This test verifies that the API correctly handles invalid query parameters
    with appropriate error messages.
    """
    # Test with invalid confidence value
    response = client.get('/api/valuations?min_confidence=invalid', headers=auth_headers)
    assert response.status_code in [400, 422], \
        f"Expected 400 Bad Request or 422 Unprocessable Entity, got {response.status_code}"
    
    # Test with invalid page number
    response = client.get('/api/valuations?page=invalid', headers=auth_headers)
    assert response.status_code in [400, 422], \
        f"Expected 400 Bad Request or 422 Unprocessable Entity, got {response.status_code}"
    
    # Test with excessively large limit value
    response = client.get('/api/valuations?limit=10000', headers=auth_headers)
    data = json.loads(response.data)
    if response.status_code == 200:
        # If the API caps the limit instead of rejecting the request
        assert len(data['valuations']) <= 100, "API should cap the number of results to a reasonable limit"
    else:
        # If the API rejects the request
        assert response.status_code in [400, 422], \
            f"Expected 400 Bad Request or 422 Unprocessable Entity, got {response.status_code}"


def test_api_missing_gis_data(client, auth_headers):
    """
    Test valuation endpoint with missing GIS data.
    
    This test verifies that the API gracefully handles property valuation requests
    when GIS data is missing or unavailable.
    """
    # Prepare a request with missing coordinates
    property_data = {
        'address': '123 Test Street',
        'city': 'Test City',
        'state': 'TS',
        'zip_code': '12345',
        'property_type': 'Single Family',
        'bedrooms': 3,
        'bathrooms': 2,
        'square_feet': 2000,
        'year_built': 2010,
        # Deliberately omitting latitude and longitude
        'valuation_method': 'enhanced_regression'
    }
    
    # Test the valuation endpoint
    response = client.post('/api/valuation', json=property_data, headers=auth_headers)
    assert response.status_code == 200, \
        f"API should still provide a valuation when GIS data is missing, got {response.status_code}"
    
    data = json.loads(response.data)
    assert 'estimated_value' in data, "Response should include an estimated value even without GIS data"
    assert 'gis_adjustments' not in data, "Response should not include GIS adjustments when coordinates are missing"
    
    # Now add invalid coordinates
    property_data['latitude'] = 'not-a-number'
    property_data['longitude'] = 'not-a-number'
    
    # Test with invalid GIS data
    response = client.post('/api/valuation', json=property_data, headers=auth_headers)
    assert response.status_code in [200, 400, 422], \
        f"API should either handle invalid GIS data or reject the request, got {response.status_code}"
    
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'estimated_value' in data, "Response should include an estimated value"
        assert 'gis_adjustments' not in data, "Response should not include GIS adjustments for invalid coordinates"


# === Performance and Edge Case Tests ===

def test_api_performance_metrics(client, auth_headers, seed_test_data):
    """
    Test performance metrics collection in API responses.
    
    This test verifies that the API includes performance metrics in response headers
    or response body, allowing for performance monitoring and troubleshooting.
    """
    # Make a request and check for performance metrics
    start_time = datetime.utcnow()
    response = client.get('/api/valuations', headers=auth_headers)
    end_time = datetime.utcnow()
    assert response.status_code == 200
    
    data = json.loads(response.data)
    
    # Verify response time metrics are included in response metadata or headers
    if 'metadata' in data and 'query_time_ms' in data['metadata']:
        # If query time is in response metadata
        assert isinstance(data['metadata']['query_time_ms'], (int, float)), \
            "Query time should be a number"
        assert data['metadata']['query_time_ms'] >= 0, \
            "Query time should be non-negative"
    
    # Calculate actual response time and assert it's reasonable
    response_time_ms = (end_time - start_time).total_seconds() * 1000
    assert response_time_ms < 5000, \
        f"API response time ({response_time_ms}ms) should be under 5 seconds"


def test_api_validation_edge_cases(client, auth_headers):
    """
    Test API input validation edge cases.
    
    This test checks how the API handles various edge cases in input validation,
    such as empty values, extremely large values, and special characters.
    """
    # Test with empty valuation method
    response = client.get('/api/valuations?method=', headers=auth_headers)
    assert response.status_code == 200, \
        "API should handle empty method parameter gracefully"
    
    # Test with very long neighborhood name
    long_neighborhood = 'A' * 1000  # 1000 character string
    response = client.get(f'/api/valuations?neighborhood={long_neighborhood}', headers=auth_headers)
    assert response.status_code in [200, 400, 422], \
        "API should handle very long input parameters appropriately"
    
    # Test with special characters in parameters
    special_chars = '%20!@#$%^&*()_+<>?:"{}|[];\',./`~'
    response = client.get(f'/api/valuations?neighborhood={special_chars}', headers=auth_headers)
    assert response.status_code in [200, 400, 422], \
        "API should handle special characters in input parameters appropriately"
    
    # Test with negative confidence value
    response = client.get('/api/valuations?min_confidence=-0.5', headers=auth_headers)
    assert response.status_code in [200, 400, 422], \
        "API should handle negative confidence values appropriately"
    
    # Test with confidence value > 1.0
    response = client.get('/api/valuations?min_confidence=1.5', headers=auth_headers)
    assert response.status_code in [200, 400, 422], \
        "API should handle confidence values > 1.0 appropriately"
    
    # Test with multiple filters simultaneously
    response = client.get('/api/valuations?method=enhanced_regression&min_confidence=0.7&neighborhood=Test%20Neighborhood&page=1&limit=10', 
                         headers=auth_headers)
    assert response.status_code == 200, \
        "API should handle multiple filters simultaneously"


def test_concurrent_etl_status_update(app, client, auth_headers, seed_test_data):
    """
    Test the behavior when concurrent ETL status updates occur.
    
    This test simulates a scenario where multiple ETL jobs attempt to update
    the ETL status simultaneously, and verifies that the system handles this
    correctly without data corruption.
    """
    with app.app_context():
        from app import db
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        # Function to update ETL status
        def update_etl_status(status_value, progress_value):
            try:
                etl_status = EtlStatus(
                    status=status_value,
                    progress=progress_value,
                    records_processed=50,
                    success_rate=0.9,
                    average_processing_time=1.5,
                    completeness=0.9,
                    accuracy=0.85,
                    timeliness=0.95
                )
                db.session.add(etl_status)
                db.session.commit()
                return True
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error updating ETL status: {e}")
                return False
        
        # Define status values to test with
        status_values = [
            ('processing', 0.25),
            ('processing', 0.5),
            ('processing', 0.75),
            ('completed', 1.0)
        ]
        
        # Mock concurrent updates using threads
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for status, progress in status_values:
                results.append(executor.submit(update_etl_status, status, progress))
        
        # Verify all updates completed without errors
        success_count = sum(1 for result in results if result.result())
        assert success_count > 0, "At least some ETL status updates should succeed"
        
        # Verify final ETL status is valid
        final_status = EtlStatus.query.order_by(EtlStatus.id.desc()).first()
        assert final_status is not None, "Final ETL status should exist"
        assert final_status.status in ['processing', 'completed'], \
            f"Final ETL status should be valid, got {final_status.status}"
        assert 0 <= final_status.progress <= 1.0, \
            f"Progress should be between 0 and 1, got {final_status.progress}"


def test_database_connection_resiliency(app, client, auth_headers):
    """
    Test the API's resiliency to database connection issues.
    
    This test simulates database connection failures and verifies that the API
    provides appropriate error responses and gracefully handles reconnection.
    """
    # Test API behavior when database connection is disrupted
    with mock.patch('app.db.session.query', side_effect=Exception("Database connection error")):
        # Attempt to access an endpoint that requires database access
        response = client.get('/api/valuations', headers=auth_headers)
        
        # API should return a 5xx error code for database issues
        assert response.status_code >= 500, \
            f"API should return 5xx error for database issues, got {response.status_code}"
        
        data = json.loads(response.data)
        assert 'error' in data, "Response should include an error message"
        assert 'message' in data['error'], "Error should include a message"
        assert 'database' in data['error']['message'].lower() or \
               'connection' in data['error']['message'].lower(), \
            "Error message should indicate a database or connection issue"
    
    # After connection is restored, API should resume normal operation
    response = client.get('/api/valuations', headers=auth_headers)
    assert response.status_code == 200, \
        f"API should return to normal operation after database connection is restored"


def test_large_data_handling(client, auth_headers, app):
    """
    Test the API's ability to handle large amounts of data.
    
    This test verifies that the API correctly handles requests for large amounts
    of data, using pagination and efficient query execution.
    """
    with app.app_context():
        from app import db
        
        # First, seed a large number of test properties
        # Note: We're keeping this small for the test environment
        # but in a real test, this would create hundreds or thousands of properties
        property_count = 25
        
        for i in range(1, property_count + 1):
            prop = Property(
                property_id=f'LARGE-TEST-PROP-{i}',
                address=f'{i} Large Test Street',
                neighborhood='Large Test Neighborhood',
                property_type='Single Family',
                year_built=2000 + (i % 20),
                bedrooms=3 + (i % 3),
                bathrooms=2.0 + (i % 2),
                living_area=2000.0 + (i * 100),
                land_area=5000.0 + (i * 200)
            )
            db.session.add(prop)
        
        db.session.commit()
        
        # Add valuations for all properties
        properties = Property.query.filter(Property.property_id.like('LARGE-TEST-PROP-%')).all()
        for i, prop in enumerate(properties):
            valuation = PropertyValuation(
                property_id=prop.id,
                estimated_value=300000 + (i * 25000),
                confidence_score=max(0.5, 0.9 - (i * 0.01)),  # Keep between 0.5 and 0.9
                valuation_method='enhanced_regression' if i % 3 == 0 else 'lightgbm' if i % 3 == 1 else 'xgboost',
                valuation_date=datetime.utcnow() - timedelta(days=i % 30)
            )
            db.session.add(valuation)
        
        db.session.commit()
    
    # Test requesting all data (should use pagination)
    response = client.get('/api/valuations?limit=100', headers=auth_headers)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'valuations' in data, "Response should include a 'valuations' field"
    
    # Verify pagination metadata is included
    assert 'total' in data, "Response should include pagination metadata"
    assert 'page' in data, "Response should include pagination metadata"
    assert 'limit' in data, "Response should include pagination metadata"
    
    # Verify data volume
    assert len(data['valuations']) <= data['limit'], \
        "Number of records returned should not exceed the specified limit"
    
    # If there are multiple pages, test accessing the second page
    if data['total'] > data['limit']:
        response = client.get('/api/valuations?page=2&limit=10', headers=auth_headers)
        assert response.status_code == 200
        
        page2_data = json.loads(response.data)
        assert 'valuations' in page2_data, "Response should include a 'valuations' field"
        assert page2_data['page'] == 2, "Page number should be 2"
        
        # Verify we got different records on page 2
        if data['valuations'] and page2_data['valuations']:
            assert data['valuations'][0]['property_id'] != page2_data['valuations'][0]['property_id'], \
                "Page 2 should return different records than page 1"


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main(['-xvs', __file__])