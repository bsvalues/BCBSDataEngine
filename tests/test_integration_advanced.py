"""
Advanced Integration tests for the BCBS_Values system.

These tests validate the full ETL pipeline, advanced valuation functions, and API endpoints
in an integrated manner, simulating real-world usage patterns with a focus on enhanced features.
"""
import os
import sys
import json
import logging
import unittest
from unittest import mock
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Import valuation components
from src.valuation import estimate_property_value
from src.gis_integration import calculate_proximity_score

# Import API application
from api import app

# Initialize the test client
client = TestClient(app)

# Sample test data
TEST_PROPERTY = {
    "parcel_id": "TEST12345678",
    "address": "123 Test Street",
    "city": "Richland",
    "state": "WA",
    "zip_code": "99352",
    "bedrooms": 3,
    "bathrooms": 2.5,
    "square_feet": 2200,
    "lot_size": 8500,
    "year_built": 2010,
    "latitude": 46.2804,
    "longitude": -119.2752,
    "property_type": "single_family",
    "neighborhood": "Meadow Springs"
}

class TestETLPipeline:
    """Test the complete ETL pipeline with mocked data sources."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database connection for testing."""
        # Use a mock for the database to avoid actual DB operations
        with mock.patch('db.database.Database') as mock_db:
            # Configure the mock to return predictable values
            mock_db_instance = mock_db.return_value
            mock_db_instance.insert_property.return_value = 1
            mock_db_instance.insert_validation_result.return_value = 1
            mock_db_instance.get_properties.return_value = [TEST_PROPERTY]
            yield mock_db_instance
    
    @pytest.fixture
    def mock_pacs_data(self):
        """Create mock PACS data for testing."""
        return [TEST_PROPERTY]
    
    @pytest.fixture
    def mock_mls_data(self):
        """Create mock MLS data for testing."""
        mls_data = TEST_PROPERTY.copy()
        mls_data.update({
            "list_price": 450000,
            "days_on_market": 14,
            "listing_status": "active"
        })
        return [mls_data]
    
    @pytest.fixture
    def mock_narrpr_data(self):
        """Create mock NARRPR data for testing."""
        narrpr_data = TEST_PROPERTY.copy()
        narrpr_data.update({
            "estimated_value": 445000,
            "confidence_score": 0.85,
            "comparable_sales": 3
        })
        return [narrpr_data]
    
    def test_etl_pipeline_integration(self, mock_database, mock_pacs_data, mock_mls_data, mock_narrpr_data):
        """
        Test the full ETL pipeline integration.
        
        This test simulates the complete ETL process:
        1. Extracting data from PACS, MLS, and NARRPR sources
        2. Transforming and validating the data
        3. Loading the data into the database
        """
        # Mock the external data sources
        with mock.patch('etl.pacs_import.PACSImporter.fetch_properties', return_value=mock_pacs_data), \
             mock.patch('etl.mls_scraper.MLSScraper.fetch_listings', return_value=mock_mls_data), \
             mock.patch('etl.narrpr_scraper.NARRPRScraper.fetch_valuations', return_value=mock_narrpr_data):
            
            # Step 1: Extract data from PACS
            logger.info("Extracting data from PACS...")
            pacs_importer = PACSImporter()
            properties = pacs_importer.fetch_properties()
            
            # Verify PACS extraction
            assert len(properties) > 0, "No properties extracted from PACS"
            assert properties[0]['parcel_id'] == TEST_PROPERTY['parcel_id'], "Incorrect property data from PACS"
            
            # Step 2: Extract data from MLS
            logger.info("Extracting data from MLS...")
            mls_scraper = MLSScraper()
            listings = mls_scraper.fetch_listings()
            
            # Verify MLS extraction
            assert len(listings) > 0, "No listings extracted from MLS"
            assert 'list_price' in listings[0], "MLS data missing list price"
            
            # Step 3: Extract data from NARRPR
            logger.info("Extracting data from NARRPR...")
            narrpr_scraper = NARRPRScraper()
            valuations = narrpr_scraper.fetch_valuations()
            
            # Verify NARRPR extraction
            assert len(valuations) > 0, "No valuations extracted from NARRPR"
            assert 'estimated_value' in valuations[0], "NARRPR data missing estimated value"
            
            # Step 4: Merge and validate the data
            logger.info("Validating property data...")
            # Merge the first property with its corresponding listing and valuation
            merged_property = {**properties[0], **listings[0], **valuations[0]}
            
            # Validate the merged property data
            validation_results = validate_property_data([merged_property])
            
            # Verify validation
            assert len(validation_results) > 0, "No validation results returned"
            assert 'is_valid' in validation_results[0], "Validation result missing is_valid flag"
            assert validation_results[0]['is_valid'], "Property data failed validation"
            
            # Step 5: Load the validated data into the database
            logger.info("Loading validated data into database...")
            # Insert the property into the database
            property_id = mock_database.insert_property(merged_property)
            
            # Insert the validation result into the database
            validation_id = mock_database.insert_validation_result(
                property_id=property_id, 
                validation_result=validation_results[0]
            )
            
            # Verify database loading
            assert property_id is not None, "Failed to insert property into database"
            assert validation_id is not None, "Failed to insert validation result into database"
            
            logger.info("ETL pipeline integration test completed successfully")


class TestAdvancedValuation:
    """Test the advanced valuation functions with realistic data."""
    
    @pytest.fixture
    def sample_property_data(self):
        """Create sample property data for valuation testing."""
        return pd.DataFrame([
            {
                "parcel_id": "TEST12345678",
                "bedrooms": 3,
                "bathrooms": 2.5,
                "square_feet": 2200,
                "year_built": 2010,
                "latitude": 46.2804,
                "longitude": -119.2752,
                "sale_price": 450000,
                "neighborhood": "Meadow Springs",
                "city": "Richland",
                "property_type": "single_family"
            },
            {
                "parcel_id": "TEST87654321",
                "bedrooms": 4,
                "bathrooms": 3,
                "square_feet": 2800,
                "year_built": 2015,
                "latitude": 46.2822,
                "longitude": -119.2780,
                "sale_price": 550000,
                "neighborhood": "Meadow Springs",
                "city": "Richland",
                "property_type": "single_family"
            },
            {
                "parcel_id": "TEST24680123",
                "bedrooms": 2,
                "bathrooms": 1.5,
                "square_feet": 1500,
                "year_built": 1995,
                "latitude": 46.2112,
                "longitude": -119.1367,
                "sale_price": 320000,
                "neighborhood": "Southridge",
                "city": "Kennewick",
                "property_type": "single_family"
            }
        ])
    
    @pytest.fixture
    def sample_target_property(self):
        """Create a sample target property for valuation."""
        return {
            "parcel_id": "TESTTARGET123",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 2000,
            "year_built": 2008,
            "latitude": 46.2804,
            "longitude": -119.2752,
            "neighborhood": "Meadow Springs",
            "city": "Richland",
            "property_type": "single_family"
        }
    
    @pytest.fixture
    def ref_points(self):
        """Define reference points for GIS-enhanced valuation."""
        return {
            'downtown_richland': {
                'lat': 46.2804, 
                'lon': -119.2752, 
                'weight': 1.0
            },
            'downtown_kennewick': {
                'lat': 46.2112, 
                'lon': -119.1367, 
                'weight': 0.9
            }
        }
    
    @pytest.fixture
    def neighborhood_ratings(self):
        """Define neighborhood ratings for location quality adjustments."""
        return {
            'Richland': 1.15,
            'Kennewick': 1.0,
            'Meadow Springs': 1.2,
            'Southridge': 1.05,
            'Unknown': 1.0
        }
    
    def test_advanced_valuation_with_gis(self, sample_property_data, sample_target_property, ref_points, neighborhood_ratings):
        """
        Test the advanced valuation function with GIS integration.
        
        This test validates that the advanced valuation function:
        1. Successfully processes property data with GIS features
        2. Returns a valid estimated value with confidence interval
        3. Includes advanced metrics like R-squared and feature importances
        """
        try:
            # Convert sample data to DataFrame if it's not already
            if not isinstance(sample_property_data, pd.DataFrame):
                sample_property_data = pd.DataFrame(sample_property_data)
            
            # Step 1: Run the advanced valuation function with GIS features
            logger.info("Running advanced valuation with GIS integration...")
            valuation_result = estimate_property_value(
                property_data=sample_property_data,
                target_property=sample_target_property,
                gis_data=None,  # GIS data will be calculated internally based on coordinates
                ref_points=ref_points,
                neighborhood_ratings=neighborhood_ratings,
                use_gis_features=True,
                use_multiple_regression=True,
                include_advanced_metrics=True
            )
            
            # Step 2: Verify the valuation result contains all expected components
            assert valuation_result is not None, "Valuation result is None"
            assert 'estimated_value' in valuation_result, "Missing estimated value in result"
            assert valuation_result['estimated_value'] > 0, "Estimated value should be positive"
            
            # Step 3: Verify confidence interval
            assert 'confidence_interval' in valuation_result, "Missing confidence interval"
            assert len(valuation_result['confidence_interval']) == 2, "Confidence interval should have lower and upper bounds"
            
            # Step 4: Verify advanced metrics
            assert 'r_squared' in valuation_result, "Missing R-squared in advanced metrics"
            assert 0 <= valuation_result['r_squared'] <= 1, "R-squared should be between 0 and 1"
            
            assert 'adjusted_r_squared' in valuation_result, "Missing adjusted R-squared in advanced metrics"
            assert 'mean_absolute_error' in valuation_result, "Missing MAE in advanced metrics"
            
            # Step 5: Verify feature importances are included
            assert 'feature_importances' in valuation_result, "Missing feature importances"
            assert len(valuation_result['feature_importances']) > 0, "Feature importances should not be empty"
            
            # Step 6: Verify model coefficients are included
            assert 'model_coefficients' in valuation_result, "Missing model coefficients"
            
            # Log the successful result
            logger.info(f"Advanced valuation test completed successfully. Estimated value: ${valuation_result['estimated_value']:,.2f}")
            logger.info(f"R-squared: {valuation_result['r_squared']:.4f}")
            
        except Exception as e:
            pytest.fail(f"Advanced valuation test failed with exception: {str(e)}")


class TestAPIEndpoints:
    """Test the API endpoints using FastAPI's TestClient."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database with test data."""
        with mock.patch('api.get_db') as mock_get_db:
            mock_db = mock.MagicMock()
            
            # Mock property data
            mock_properties = [
                {
                    "id": 1,
                    "parcel_id": "TEST12345678",
                    "address": "123 Test Street",
                    "city": "Richland",
                    "state": "WA",
                    "zip_code": "99352",
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "square_feet": 2200,
                    "lot_size": 8500,
                    "year_built": 2010,
                    "latitude": 46.2804,
                    "longitude": -119.2752,
                    "property_type": "single_family",
                    "neighborhood": "Meadow Springs",
                    "created_at": "2025-03-29T22:15:00"
                }
            ]
            
            # Mock valuations
            mock_valuations = [
                {
                    "id": 1,
                    "property_id": 1,
                    "estimated_value": 450000,
                    "confidence_interval_low": 425000,
                    "confidence_interval_high": 475000,
                    "valuation_date": "2025-03-29T22:30:00",
                    "model_version": "advanced_regression_1.2.0",
                    "r_squared": 0.92,
                    "mean_absolute_error": 15000,
                    "mean_squared_error": 250000000,
                    "created_at": "2025-03-29T22:30:00"
                }
            ]
            
            # Set up the mock db to return test data
            mock_db.get_properties.return_value = mock_properties
            mock_db.get_property_by_id.return_value = mock_properties[0]
            mock_db.get_valuations.return_value = mock_valuations
            mock_db.get_valuation_by_property_id.return_value = mock_valuations[0]
            
            # Mock ETL status
            mock_db.get_etl_status.return_value = {
                "last_run": "2025-03-29T22:00:00",
                "status": "completed",
                "records_processed": 150,
                "success_rate": 0.98,
                "errors": 3,
                "next_scheduled_run": "2025-03-30T22:00:00"
            }
            
            # Mock agent status
            mock_db.get_agent_status.return_value = [
                {
                    "agent_id": "bcbs-cascade-operator",
                    "status": "active",
                    "last_activity": "2025-03-29T23:45:00",
                    "tasks_completed": 42,
                    "success_rate": 0.95
                },
                {
                    "agent_id": "bcbs-bootstrap-commander",
                    "status": "idle",
                    "last_activity": "2025-03-29T23:30:00",
                    "tasks_completed": 38,
                    "success_rate": 0.92
                }
            ]
            
            # Configure mock_get_db to return our mock_db
            mock_get_db.return_value = mock_db
            yield mock_db
    
    def test_health_endpoint(self):
        """Test the health check endpoint to ensure API is operational."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "OK"
    
    def test_valuations_endpoint(self, mock_db):
        """
        Test the valuations endpoint that returns property valuations.
        
        This test verifies that:
        1. The endpoint returns a 200 OK status
        2. The response contains a list of valuations
        3. The valuations have all required fields
        """
        response = client.get("/api/valuations")
        assert response.status_code == 200
        
        data = response.json()
        assert "valuations" in data
        assert isinstance(data["valuations"], list)
        assert len(data["valuations"]) > 0
        
        # Verify the first valuation has all required fields
        valuation = data["valuations"][0]
        assert "property_id" in valuation
        assert "estimated_value" in valuation
        assert "confidence_interval_low" in valuation
        assert "confidence_interval_high" in valuation
        assert "r_squared" in valuation
    
    def test_valuation_by_property_id(self, mock_db):
        """
        Test the endpoint for getting a valuation for a specific property.
        
        This test verifies that:
        1. The endpoint returns a 200 OK status for a valid property ID
        2. The response contains the correct property valuation
        3. The endpoint returns a 404 Not Found for an invalid property ID
        """
        # Test with valid property ID
        response = client.get("/api/valuations/1")
        assert response.status_code == 200
        
        valuation = response.json()
        assert valuation["property_id"] == 1
        assert valuation["estimated_value"] == 450000
        assert valuation["r_squared"] == 0.92
        
        # Test with invalid property ID
        response = client.get("/api/valuations/999")
        assert response.status_code == 404
    
    def test_etl_status_endpoint(self, mock_db):
        """
        Test the ETL status endpoint.
        
        This test verifies that:
        1. The endpoint returns a 200 OK status
        2. The response contains ETL status information
        3. The status information includes all required fields
        """
        response = client.get("/api/etl-status")
        assert response.status_code == 200
        
        status = response.json()
        assert "last_run" in status
        assert "status" in status
        assert "records_processed" in status
        assert "success_rate" in status
        assert status["status"] == "completed"
    
    def test_agent_status_endpoint(self, mock_db):
        """
        Test the agent status endpoint.
        
        This test verifies that:
        1. The endpoint returns a 200 OK status
        2. The response contains a list of agent statuses
        3. Each agent status includes all required fields
        """
        response = client.get("/api/agent-status")
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) > 0
        
        # Verify the first agent has all required fields
        agent = data["agents"][0]
        assert "agent_id" in agent
        assert "status" in agent
        assert "last_activity" in agent
        assert "tasks_completed" in agent
        assert "success_rate" in agent
    
    def test_what_if_valuation_endpoint(self, mock_db):
        """
        Test the what-if valuation endpoint.
        
        This test verifies that:
        1. The endpoint accepts a POST request with property parameters
        2. The response contains a valuation prediction with adjusted parameters
        3. The response includes sensitivity analysis on parameter changes
        """
        # Define test parameters for what-if analysis
        what_if_params = {
            "property_id": 1,
            "adjustments": {
                "bedrooms": 4,  # Increased from 3
                "bathrooms": 3,  # Increased from 2.5
                "square_feet": 2500,  # Increased from 2200
                "year_built": 2010,  # Same
                "neighborhood": "Meadow Springs",  # Same
                "city": "Richland"  # Same
            },
            "market_conditions": {
                "interest_rate": 4.5,
                "market_trend": "rising"
            },
            "quality_factors": {
                "school_quality": 8,
                "crime_rate": 2,
                "property_condition": 9
            }
        }
        
        # Mock the valuation function to return a predetermined result
        with mock.patch('api.estimate_property_value') as mock_valuation:
            mock_valuation.return_value = {
                "estimated_value": 490000,  # Increased from original 450000
                "confidence_interval": [465000, 515000],
                "r_squared": 0.93,
                "adjusted_r_squared": 0.91,
                "mean_absolute_error": 14500,
                "feature_importances": {
                    "square_feet": 0.48,
                    "bathrooms": 0.22,
                    "bedrooms": 0.15,
                    "property_age": 0.10,
                    "neighborhood_rating": 0.05
                },
                "sensitivity_analysis": {
                    "bedrooms": {
                        "impact": 15000,
                        "direction": "positive"
                    },
                    "bathrooms": {
                        "impact": 20000,
                        "direction": "positive"
                    },
                    "square_feet": {
                        "impact": 25000,
                        "direction": "positive"
                    }
                }
            }
            
            response = client.post(
                "/api/what-if-valuation",
                json=what_if_params
            )
            
            assert response.status_code == 200
            
            result = response.json()
            assert "estimated_value" in result
            assert result["estimated_value"] > 0
            
            # Verify sensitivity analysis
            assert "sensitivity_analysis" in result
            assert "square_feet" in result["sensitivity_analysis"]
            
            # Verify comparison with original value
            assert "original_value" in result
            assert "value_difference" in result
            assert "percentage_change" in result


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])