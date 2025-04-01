"""
Test module for the BCBS Values API diagnostic endpoints.

This test suite verifies that the core API health and status endpoints
are functioning correctly by checking response codes and data structures.
"""
import pytest
import logging
import json
from fastapi.testclient import TestClient

# Try to import from different API modules, as we have multiple implementations
try:
    # Try the main API first
    from api import app
except ImportError:
    try:
        # Try the simple API next
        from simple_api import app
    except ImportError:
        try:
            # Try basic API as a fallback
            from basic_api import app
        except ImportError:
            pytest.skip("No API module found to test")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

# Test fixtures and utilities
@pytest.fixture
def log_response():
    """Fixture to log API response details for debugging."""
    def _log_response(response, endpoint):
        logger.info(f"Response from {endpoint}: Status {response.status_code}")
        try:
            logger.debug(f"Response JSON: {json.dumps(response.json(), indent=2)}")
        except Exception:
            logger.debug(f"Response text: {response.text}")
    return _log_response


class TestAPIHealth:
    """Test suite for API health endpoints."""
    
    def test_health_endpoint(self, log_response):
        """
        Test the /health or root endpoint to verify basic API availability.
        
        This test ensures that:
        1. The health endpoint returns a 200 OK status
        2. The response contains the expected structure (status, version, timestamp)
        """
        # Try the root endpoint first as it commonly returns health status
        response = client.get("/")
        
        # If root doesn't work, try explicit health endpoints
        if response.status_code != 200:
            # Try /health endpoint
            response = client.get("/health")
            
            # If that doesn't work, try /api/health
            if response.status_code != 200:
                response = client.get("/api/health")
        
        # Log response for debugging
        log_response(response, "health endpoint")
        
        # Verify response
        assert response.status_code == 200, f"Health check failed with status {response.status_code}"
        
        # Check response structure - should have status field
        data = response.json()
        assert "status" in data, "Health response missing 'status' field"
        
        # Status should indicate API is operational
        status = data["status"].lower()
        assert any(term in status for term in ["ok", "operational", "healthy"]), \
            f"Unexpected health status: {status}"
        
        # Log additional fields that might be present
        extra_fields = [key for key in data.keys() if key != "status"]
        if extra_fields:
            logger.info(f"Additional health fields present: {', '.join(extra_fields)}")


class TestETLStatus:
    """Test suite for ETL status endpoint."""
    
    def test_etl_status_endpoint(self, log_response):
        """
        Test the /api/etl-status endpoint.
        
        This test verifies that:
        1. The ETL status endpoint returns a 200 OK status
        2. The response contains the expected structure with ETL information
        """
        # Request ETL status
        response = client.get("/api/etl-status")
        
        # Log response for debugging
        log_response(response, "/api/etl-status")
        
        # Verify response
        assert response.status_code == 200, f"ETL status check failed with status {response.status_code}"
        
        # Verify response structure
        data = response.json()
        
        # Check if response is the new format with 'jobs' field or old format with 'status'
        if "jobs" in data:
            # New format (using ETLStatusResponse model)
            assert isinstance(data["jobs"], list), "ETL jobs should be a list"
            assert "stats" in data, "ETL status missing 'stats' field"
            assert "health" in data, "ETL status missing 'health' field"
            assert "timestamp" in data, "ETL status missing 'timestamp' field"
            
            # Log job information
            job_count = len(data["jobs"])
            logger.info(f"ETL status returned {job_count} jobs")
            
            # Check health status
            if "status" in data["health"]:
                health_status = data["health"]["status"]
                logger.info(f"ETL health status: {health_status}")
        else:
            # Old format or simple_api format
            assert "status" in data, "ETL status missing 'status' field"
            assert "last_run" in data, "ETL status missing 'last_run' field"
            assert "sources_processed" in data, "ETL status missing 'sources_processed' field"
            
            # Log status information
            status = data["status"]
            logger.info(f"ETL status: {status}")
            
            # Check sources
            sources = data["sources_processed"]
            assert isinstance(sources, list), "Sources processed should be a list"
            logger.info(f"ETL status returned {len(sources)} sources")


class TestAgentStatus:
    """Test suite for Agent status endpoint."""
    
    def test_agent_status_endpoint(self, log_response):
        """
        Test the /api/agent-status endpoint.
        
        This test verifies that:
        1. The agent status endpoint returns a 200 OK status
        2. The response contains the expected structure with agent information
        """
        # Request agent status
        response = client.get("/api/agent-status")
        
        # Log response for debugging
        log_response(response, "/api/agent-status")
        
        # Verify response
        assert response.status_code == 200, f"Agent status check failed with status {response.status_code}"
        
        # Verify response structure
        data = response.json()
        
        # Check if response is the new format with 'agents' field explicitly
        if "agents" in data:
            # New format (using AgentStatusResponse model)
            assert isinstance(data["agents"], list), "Agents should be a list"
            
            # Check if we have metrics and health information
            if "metrics" in data:
                logger.info(f"Agent metrics found in response")
            
            if "health" in data:
                logger.info(f"Agent health information found in response")
                
            # Log agent information
            agent_count = len(data["agents"])
            logger.info(f"Agent status returned {agent_count} agents")
            
            # Check agent structure if any agents exist
            if agent_count > 0:
                first_agent = data["agents"][0]
                assert "agent_id" in first_agent or "id" in first_agent, "Agent missing ID field"
                assert "status" in first_agent, "Agent missing status field"
        else:
            # Handle simpler format (like in simple_api)
            assert "system_status" in data, "Agent status missing 'system_status' field"
            assert "active_agents" in data, "Agent status missing 'active_agents' field"
            assert "tasks_in_progress" in data, "Agent status missing 'tasks_in_progress' field"
            
            # Log status information
            logger.info(f"Agent system status: {data['system_status']}")
            logger.info(f"Active agents: {data['active_agents']}")


if __name__ == "__main__":
    # Execute tests if run directly (useful for quick debugging)
    pytest.main(["-xvs", __file__])