#!/usr/bin/env python3
"""
Enhanced API Test Script

This script tests the enhanced API endpoints including token-based authentication,
valuations, ETL status, and agent status endpoints.
"""

import json
import os
import sys
import time
import unittest
import requests

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5002/api")
API_KEY = os.environ.get("API_KEY", "bcbs_demo_key_2023")
TEST_AGENT_ID = "test-agent-001"
TEST_AGENT_TYPE = "test_script"

class EnhancedAPITest(unittest.TestCase):
    """Test cases for the enhanced API functionality."""

    def setUp(self):
        """Set up the test case."""
        self.api_url = API_BASE_URL
        self.api_key = API_KEY
        self.token = None
        self.headers = {"X-API-Key": self.api_key}
        
        # Try to get a token
        try:
            self.get_token()
        except Exception as e:
            print(f"Warning: Could not get token: {e}")
    
    def get_token(self):
        """Get an authentication token."""
        data = {
            "agent_id": TEST_AGENT_ID,
            "agent_type": TEST_AGENT_TYPE,
            "api_key": self.api_key
        }
        response = requests.post(
            f"{self.api_url}/auth/token",
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            if "token" in result:
                self.token = result["token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
                return True
        return False
    
    def test_01_auth_token(self):
        """Test the authentication token endpoint."""
        data = {
            "agent_id": TEST_AGENT_ID,
            "agent_type": TEST_AGENT_TYPE,
            "api_key": self.api_key
        }
        response = requests.post(
            f"{self.api_url}/auth/token",
            json=data
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("token", result)
        self.assertIn("expires_in", result)
        self.assertIn("token_type", result)
        self.assertEqual(result["token_type"], "Bearer")
    
    def test_02_valuations(self):
        """Test the valuations endpoint."""
        response = requests.get(
            f"{self.api_url}/valuations",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("valuations", result)
        self.assertIn("page", result)
        self.assertIn("limit", result)
        self.assertIn("total", result)
    
    def test_03_valuations_filtering(self):
        """Test valuations endpoint with filtering."""
        response = requests.get(
            f"{self.api_url}/valuations?limit=5&sort_by=valuation_date&sort_dir=desc",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("valuations", result)
        self.assertLessEqual(len(result["valuations"]), 5)
    
    def test_04_etl_status(self):
        """Test the ETL status endpoint."""
        response = requests.get(
            f"{self.api_url}/etl-status",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("jobs", result)
        self.assertIn("stats", result)
        self.assertIn("health", result)
    
    def test_05_agent_status(self):
        """Test the agent status endpoint."""
        response = requests.get(
            f"{self.api_url}/agent-status",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("agents", result)
        self.assertIn("count", result)
        self.assertIn("metrics", result)
    
    def test_06_agent_status_filtering(self):
        """Test agent status endpoint with filtering."""
        response = requests.get(
            f"{self.api_url}/agent-status?active_only=true",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("agents", result)
        
        # Check that all agents are active
        for agent in result["agents"]:
            self.assertTrue(agent["is_active"])
    
    def test_07_agent_logs(self):
        """Test the agent logs endpoint."""
        # First get an agent ID
        response = requests.get(
            f"{self.api_url}/agent-status",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        if len(result["agents"]) > 0:
            agent_id = result["agents"][0]["id"]
            response = requests.get(
                f"{self.api_url}/agent-logs/{agent_id}",
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("agent_id", result)
            self.assertIn("logs", result)
        else:
            print("No agents found, skipping agent logs test")
    
    def test_08_invalid_token(self):
        """Test behavior with an invalid token."""
        # Save the current token
        original_token = self.token
        
        # Create an invalid token
        self.token = "invalid.token.here"
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
        response = requests.get(
            f"{self.api_url}/valuations",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 401)
        
        # Restore the original token
        self.token = original_token
        if self.token:
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {"X-API-Key": self.api_key}
    
    def test_09_invalid_api_key(self):
        """Test behavior with an invalid API key."""
        # Save the current headers
        original_headers = self.headers
        
        # Create headers with an invalid API key
        self.headers = {"X-API-Key": "invalid_api_key"}
        
        response = requests.get(
            f"{self.api_url}/valuations",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 401)
        
        # Restore the original headers
        self.headers = original_headers

def main():
    """Run the test suite."""
    print(f"Testing Enhanced API at {API_BASE_URL}")
    print(f"API Key: {API_KEY[:4]}...{API_KEY[-4:]}")
    
    # Check if the API is available
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print(f"Warning: API health check failed with status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {API_BASE_URL}")
        print("Make sure the API server is running and try again.")
        sys.exit(1)
    
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'])

if __name__ == "__main__":
    main()