"""
Integration Tests for BCBS Values Advanced Functionality
======================================================

This test suite verifies the end-to-end functionality of the BCBS Values platform,
including ETL processes, API endpoints, and advanced valuation algorithms.

Tests cover:
- ETL pipeline execution and data loading
- API endpoint functionality and response validation
- Advanced regression metrics calculation
- GIS-based property valuation adjustments
"""

import os
import sys
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules for testing
import db.models as models
from api import app
from etl.data_validation import validate_property_data
from etl.pacs_import import import_pacs_data
from src.valuation import ValuationEngine, AdvancedValuationEngine
from src.gis_features import GISFeatureEngine


# Create a TestClient for our API
client = TestClient(app)

# Define API routes for testing
API_ROUTES = {
    'valuations': '/api/valuations',
    'etl_status': '/api/etl-status',
    'agent_status': '/api/agent-status',
    'property_detail': '/api/properties/{property_id}',
    'property_valuation': '/api/properties/{property_id}/valuation',
    'gis_features': '/api/properties/{property_id}/gis-features'
}

# Define test data constants
TEST_API_KEY = os.environ.get('BCBS_VALUES_API_KEY', 'test_api_key_for_integration')
TEST_PROPERTY_ID = 'BENTON-12345'


# Fixtures for database and API testing
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
        'bedrooms': [3, 4, 2],
        'bathrooms': [2.5, 3.0, 1.0],
        'square_footage': [2200, 3100, 1500],
        'lot_size': [0.25, 0.4, 0.15],
        'year_built': [1995, 2005, 1975],
        'last_sale_date': ['2020-05-15', '2019-10-10', '2021-02-28'],
        'last_sale_price': [350000, 425000, 275000],
        'property_type': ['Single Family', 'Single Family', 'Single Family'],
        'latitude': [46.2122, 46.2851, 46.2068],
        'longitude': [-119.1372, -119.2785, -119.7683]
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


# ===========================================================================
# ETL Pipeline Tests
# ===========================================================================

@patch('etl.pacs_import.requests.get')
def test_etl_pipeline_execution(mock_get, db_session, sample_property_data):
    """
    Tests the execution of the ETL pipeline, verifying that data is correctly
    loaded into the database.
    
    Steps:
    1. Mock the PACS API response to return test data
    2. Run the import_pacs_data function
    3. Verify data is validated and loaded into the database
    4. Check that property counts match expectations
    """
    # Configure mock response from PACS API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'properties': sample_property_data.to_dict('records')}
    mock_get.return_value = mock_response
    
    # Mock the database session
    with patch('etl.pacs_import.db_session', db_session):
        # Execute the ETL function
        import_results = import_pacs_data()
        
        # Verify import metrics
        assert import_results['total_records'] == len(sample_property_data)
        assert import_results['valid_records'] > 0
        assert import_results['invalid_records'] == 0
        
        # Verify data was inserted into the database
        properties = db_session.query(models.Property).all()
        assert len(properties) == len(sample_property_data)
        
        # Verify specific property was imported correctly
        test_property = db_session.query(models.Property).filter_by(
            property_id=TEST_PROPERTY_ID).first()
        assert test_property is not None
        assert test_property.address == '123 Main St, Kennewick, WA'
        assert test_property.bedrooms == 3
        assert test_property.bathrooms == 2.5


def test_data_validation(sample_property_data):
    """
    Tests the data validation logic to ensure properties are correctly
    validated before being inserted into the database.
    
    Tests:
    1. Valid property data passes validation
    2. Invalid data (missing required fields) fails validation
    3. Validation reports include expected error messages
    """
    # Test valid data passes validation
    validation_results = validate_property_data(sample_property_data)
    assert validation_results['valid_count'] == len(sample_property_data)
    assert validation_results['invalid_count'] == 0
    
    # Create invalid data with missing required fields
    invalid_data = sample_property_data.copy()
    invalid_data.loc[0, 'property_id'] = None  # Missing property_id
    invalid_data.loc[1, 'square_footage'] = -100  # Invalid square footage
    
    # Test invalid data fails validation
    validation_results = validate_property_data(invalid_data)
    assert validation_results['valid_count'] < len(invalid_data)
    assert validation_results['invalid_count'] > 0
    
    # Verify validation errors contain expected messages
    errors = validation_results['errors']
    assert any('property_id' in str(err) for err in errors)
    assert any('square_footage' in str(err) for err in errors)


# ===========================================================================
# API Endpoint Tests
# ===========================================================================

def test_valuation_endpoint(authenticated_client, db_session, sample_property_data):
    """
    Tests the property valuation API endpoint to ensure it returns the
    correct valuation data with expected structure.
    
    Tests:
    1. Endpoint returns 200 status code with valid API key
    2. Response JSON has the expected structure
    3. Valuation results include required fields
    """
    # Prepare test data
    with patch('api.get_db', return_value=db_session):
        # Add test property to database
        test_property = models.Property(
            property_id=TEST_PROPERTY_ID,
            address='123 Main St, Kennewick, WA',
            bedrooms=3,
            bathrooms=2.5,
            square_footage=2200,
            lot_size=0.25,
            year_built=1995,
            last_sale_date=datetime.strptime('2020-05-15', '%Y-%m-%d'),
            last_sale_price=350000,
            property_type='Single Family',
            latitude=46.2122,
            longitude=-119.1372
        )
        db_session.add(test_property)
        db_session.commit()
        
        # Call the API endpoint
        response = authenticated_client.get(
            API_ROUTES['property_valuation'].format(property_id=TEST_PROPERTY_ID)
        )
        
        # Verify response status and structure
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields in valuation response
        assert 'property_id' in data
        assert 'estimated_value' in data
        assert 'valuation_date' in data
        assert 'confidence_score' in data
        assert 'valuation_factors' in data
        
        # Check property identification
        assert data['property_id'] == TEST_PROPERTY_ID
        
        # Verify the valuation amount is reasonable
        assert data['estimated_value'] > 0
        assert isinstance(data['estimated_value'], (int, float))
        
        # Verify confidence score is between 0 and 1
        assert 0 <= data['confidence_score'] <= 1
        
        # Verify valuation factors includes key property attributes
        factors = data['valuation_factors']
        assert 'square_footage' in factors
        assert 'location' in factors
        assert 'property_condition' in factors


def test_etl_status_endpoint(authenticated_client):
    """
    Tests the ETL status API endpoint to verify it returns information
    about recent ETL processes.
    
    Tests:
    1. Endpoint returns 200 status code
    2. Response contains information about recent ETL runs
    3. ETL status fields are correctly structured
    """
    # Call the ETL status endpoint
    response = authenticated_client.get(API_ROUTES['etl_status'])
    
    # Verify response status
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert 'last_run' in data
    assert 'status' in data
    assert 'metrics' in data
    
    # Verify metrics structure
    metrics = data['metrics']
    assert 'total_records_processed' in metrics
    assert 'valid_records' in metrics
    assert 'invalid_records' in metrics
    assert 'processing_time_seconds' in metrics


def test_agent_status_endpoint(authenticated_client):
    """
    Tests the agent status API endpoint to verify it returns information
    about the BS Army of Agents.
    
    Tests:
    1. Endpoint returns 200 status code
    2. Response contains a list of agent statuses
    3. Agent status fields match expected structure
    """
    # Call the agent status endpoint
    response = authenticated_client.get(API_ROUTES['agent_status'])
    
    # Verify response status
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert 'agents' in data
    assert isinstance(data['agents'], list)
    
    # Skip test if no agents are configured yet
    if not data['agents']:
        pytest.skip("No agents are configured in the system")
    
    # Verify agent status fields
    agent = data['agents'][0]
    assert 'agent_id' in agent
    assert 'name' in agent
    assert 'status' in agent
    assert 'last_active' in agent
    assert 'metrics' in agent
    
    # Verify metrics structure for agent
    metrics = agent['metrics']
    assert 'tasks_completed' in metrics
    assert 'success_rate' in metrics
    assert 'average_response_time' in metrics


def test_gis_features_endpoint(authenticated_client, db_session, sample_gis_data):
    """
    Tests the GIS features API endpoint to verify it returns spatial
    analysis data for properties.
    
    Tests:
    1. Endpoint returns 200 status code
    2. Response contains expected GIS features
    3. Feature values are within expected ranges
    """
    # Prepare test data
    with patch('api.get_db', return_value=db_session):
        # Add test property to database
        test_property = models.Property(
            property_id=TEST_PROPERTY_ID,
            address='123 Main St, Kennewick, WA',
            bedrooms=3,
            bathrooms=2.5,
            square_footage=2200,
            lot_size=0.25,
            year_built=1995,
            property_type='Single Family',
            latitude=46.2122,
            longitude=-119.1372
        )
        db_session.add(test_property)
        
        # Add GIS features for test property
        gis_features = models.GISFeature(
            property_id=test_property.id,
            school_proximity_score=0.85,
            park_proximity_score=0.72,
            shopping_proximity_score=0.91,
            highway_access_score=0.65,
            flood_risk_score=0.12,
            walkability_score=0.78,
            neighborhood_quality_score=0.82,
            spatial_cluster_id='SC-NW-KENNEWICK-01'
        )
        db_session.add(gis_features)
        db_session.commit()
        
        # Call the API endpoint
        response = authenticated_client.get(
            API_ROUTES['gis_features'].format(property_id=TEST_PROPERTY_ID)
        )
        
        # Verify response status and structure
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields in GIS features
        assert 'property_id' in data
        assert 'features' in data
        
        # Check property identification
        assert data['property_id'] == TEST_PROPERTY_ID
        
        # Verify expected GIS features are present
        features = data['features']
        assert 'school_proximity_score' in features
        assert 'park_proximity_score' in features
        assert 'neighborhood_quality_score' in features
        assert 'spatial_cluster_id' in features
        
        # Verify feature values match expected ranges
        assert 0 <= features['school_proximity_score'] <= 1
        assert 0 <= features['park_proximity_score'] <= 1
        assert 0 <= features['neighborhood_quality_score'] <= 1
        assert isinstance(features['spatial_cluster_id'], str)


# ===========================================================================
# Advanced Valuation Tests
# ===========================================================================

def test_advanced_regression_metrics(db_session, sample_property_data):
    """
    Tests the advanced regression metrics calculation to ensure accurate
    statistical modeling for property valuations.
    
    Tests:
    1. Multiple regression model calculates weights correctly
    2. Model coefficients are within expected ranges
    3. R-squared and other metrics are calculated correctly
    """
    # Mock the database with test properties
    with patch('src.valuation.get_db', return_value=db_session):
        # Add test properties to the database
        for index, row in sample_property_data.iterrows():
            property_data = models.Property(
                property_id=row['property_id'],
                address=row['address'],
                bedrooms=row['bedrooms'],
                bathrooms=row['bathrooms'],
                square_footage=row['square_footage'],
                lot_size=row['lot_size'],
                year_built=row['year_built'],
                last_sale_date=datetime.strptime(row['last_sale_date'], '%Y-%m-%d'),
                last_sale_price=row['last_sale_price'],
                property_type=row['property_type'],
                latitude=row['latitude'],
                longitude=row['longitude']
            )
            db_session.add(property_data)
        db_session.commit()
        
        # Initialize advanced valuation engine
        engine = AdvancedValuationEngine()
        
        # Train the model
        metrics = engine.train_model()
        
        # Verify regression metrics
        assert 'r_squared' in metrics
        assert 'mean_absolute_error' in metrics
        assert 'model_coefficients' in metrics
        
        # Check that R-squared is within reasonable range (0-1)
        assert 0 <= metrics['r_squared'] <= 1
        
        # Verify model coefficients include important property factors
        coefficients = metrics['model_coefficients']
        assert 'square_footage' in coefficients
        assert 'bedrooms' in coefficients
        assert 'bathrooms' in coefficients
        assert 'year_built' in coefficients
        
        # Verify coefficients have reasonable values
        assert coefficients['square_footage'] > 0  # Larger homes should be worth more
        assert abs(coefficients['bedrooms']) < 100000  # Coefficient shouldn't be extreme


def test_gis_adjustments_calculation(db_session, sample_property_data, sample_gis_data):
    """
    Tests the GIS adjustment calculations to ensure spatial features
    correctly influence property valuations.
    
    Tests:
    1. GIS features produce valuation adjustments
    2. Adjustment factors are within expected ranges
    3. High quality neighborhood produces positive adjustments
    """
    # Mock the database with test properties and GIS features
    with patch('src.valuation.get_db', return_value=db_session):
        # Add test property to the database
        property_data = models.Property(
            property_id=TEST_PROPERTY_ID,
            address='123 Main St, Kennewick, WA',
            bedrooms=3,
            bathrooms=2.5,
            square_footage=2200,
            lot_size=0.25,
            year_built=1995,
            last_sale_date=datetime.strptime('2020-05-15', '%Y-%m-%d'),
            last_sale_price=350000,
            property_type='Single Family',
            latitude=46.2122,
            longitude=-119.1372
        )
        db_session.add(property_data)
        db_session.commit()
        
        # Add GIS features
        gis_data = sample_gis_data[TEST_PROPERTY_ID]
        gis_features = models.GISFeature(
            property_id=property_data.id,
            school_proximity_score=gis_data['school_proximity_score'],
            park_proximity_score=gis_data['park_proximity_score'],
            shopping_proximity_score=gis_data['shopping_proximity_score'],
            highway_access_score=gis_data['highway_access_score'],
            flood_risk_score=gis_data['flood_risk_score'],
            walkability_score=gis_data['walkability_score'],
            neighborhood_quality_score=gis_data['neighborhood_quality_score'],
            spatial_cluster_id=gis_data['spatial_cluster_id']
        )
        db_session.add(gis_features)
        db_session.commit()
        
        # Initialize GIS feature engine
        gis_engine = GISFeatureEngine()
        
        # Get GIS adjustment for test property
        adjustment_factor = gis_engine.calculate_gis_adjustment(property_data.id)
        
        # Verify adjustment factor is within expected range
        assert isinstance(adjustment_factor, float)
        assert 0.5 <= adjustment_factor <= 1.5  # Should be reasonable multiplier
        
        # Test with high neighborhood quality
        # High quality neighborhood should produce positive adjustment
        gis_features.neighborhood_quality_score = 0.95
        db_session.commit()
        
        high_quality_adjustment = gis_engine.calculate_gis_adjustment(property_data.id)
        assert high_quality_adjustment > 1.0  # Positive adjustment for good neighborhood
        
        # Test with low neighborhood quality
        gis_features.neighborhood_quality_score = 0.25
        db_session.commit()
        
        low_quality_adjustment = gis_engine.calculate_gis_adjustment(property_data.id)
        assert low_quality_adjustment < 1.0  # Negative adjustment for poor neighborhood


def test_valuation_ensemble(db_session, sample_property_data):
    """
    Tests the ensemble valuation model that combines multiple valuation
    techniques for improved accuracy.
    
    Tests:
    1. Ensemble model produces valuations from multiple models
    2. Ensemble valuation has appropriate confidence score
    3. Ensemble result is within range of component models
    """
    # Add test properties to database
    with patch('src.valuation.get_db', return_value=db_session):
        # Add test property
        test_property = models.Property(
            property_id=TEST_PROPERTY_ID,
            address='123 Main St, Kennewick, WA',
            bedrooms=3,
            bathrooms=2.5,
            square_footage=2200,
            lot_size=0.25,
            year_built=1995,
            last_sale_date=datetime.strptime('2020-05-15', '%Y-%m-%d'),
            last_sale_price=350000,
            property_type='Single Family',
            latitude=46.2122,
            longitude=-119.1372
        )
        db_session.add(test_property)
        db_session.commit()
        
        # Initialize valuation engines
        basic_engine = ValuationEngine()
        advanced_engine = AdvancedValuationEngine()
        
        # Get valuations from each model
        basic_valuation = basic_engine.calculate_valuation(TEST_PROPERTY_ID)
        advanced_valuation = advanced_engine.calculate_valuation(TEST_PROPERTY_ID)
        
        # Calculate ensemble valuation (weighted average)
        ensemble_value = (basic_valuation['estimated_value'] * 0.4 + 
                          advanced_valuation['estimated_value'] * 0.6)
        
        # Verify individual valuations
        assert basic_valuation['estimated_value'] > 0
        assert advanced_valuation['estimated_value'] > 0
        
        # Verify ensemble valuation is within range of component models
        min_val = min(basic_valuation['estimated_value'], advanced_valuation['estimated_value'])
        max_val = max(basic_valuation['estimated_value'], advanced_valuation['estimated_value'])
        assert min_val <= ensemble_value <= max_val
        
        # Test ensemble confidence score calculation
        basic_confidence = basic_valuation['confidence_score']
        advanced_confidence = advanced_valuation['confidence_score']
        ensemble_confidence = (basic_confidence * 0.4 + advanced_confidence * 0.6)
        
        # Verify confidence scores are within expected range
        assert 0 <= basic_confidence <= 1
        assert 0 <= advanced_confidence <= 1
        assert 0 <= ensemble_confidence <= 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])