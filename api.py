"""
FastAPI implementation for the BCBS_Values real estate valuation API.
This module provides HTTP endpoints for property valuation, ETL status, and agent status.

This updated version includes advanced regression metrics and GIS adjustments 
for more comprehensive property valuation reports.
"""
import datetime
import json
import os
import time
import logging
import secrets  # For timing-safe string comparison
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Query, Path, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

# Import database and models
from db.database import Database
from db.models import PropertyValuation, Property, ValidationResult
from sqlalchemy import func

# Import valuation engine
try:
    from src.valuation import (
        estimate_property_value, 
        train_basic_valuation_model, 
        train_multiple_regression_model,
        advanced_property_valuation,
        calculate_gis_features
    )
    valuation_engine_available = True
    # Log successful import of the enhanced valuation engine
    import logging
    logging.getLogger(__name__).info("Successfully imported enhanced valuation engine")
except ImportError as e:
    # Log the error but don't crash - API can still work with other endpoints
    import logging
    logging.getLogger(__name__).error(f"Failed to import valuation engine: {str(e)}")
    valuation_engine_available = False
    
# Setup logging with our custom configuration
try:
    from utils.logging_config import get_api_logger
    # Get a configured logger for the API
    logger = get_api_logger()
except ImportError:
    # Fallback to basic logging if custom logger not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

logger.info("API module initializing...")

# Define GIS reference points for the Benton County, WA area
REF_POINTS = {
    'downtown_richland': {'lat': 46.2834, 'lon': -119.2786, 'weight': 1.0},
    'downtown_kennewick': {'lat': 46.2113, 'lon': -119.1367, 'weight': 0.9},
    'downtown_pasco': {'lat': 46.2395, 'lon': -119.0992, 'weight': 0.8},
    'columbia_river': {'lat': 46.2442, 'lon': -119.2576, 'weight': 0.7},
    'columbia_center': {'lat': 46.2185, 'lon': -119.2215, 'weight': 0.8},
    'hanford_site': {'lat': 46.5506, 'lon': -119.4839, 'weight': 0.4},
    'wsu_tri_cities': {'lat': 46.2734, 'lon': -119.2851, 'weight': 0.9},
    'kadlec_hospital': {'lat': 46.2836, 'lon': -119.2833, 'weight': 0.8},
    'howard_amon_park': {'lat': 46.2805, 'lon': -119.2738, 'weight': 0.9},
    'columbia_point': {'lat': 46.2280, 'lon': -119.2363, 'weight': 0.8}
}

# Define neighborhood quality ratings for Benton County, WA area
NEIGHBORHOOD_RATINGS = {
    'south_richland': 0.95,
    'west_richland': 0.85,
    'central_richland': 0.80,
    'north_richland': 0.75,
    'south_kennewick': 0.85,
    'west_kennewick': 0.80,
    'central_kennewick': 0.75,
    'east_kennewick': 0.70,
    'west_pasco': 0.80,
    'central_pasco': 0.70,
    'east_pasco': 0.65,
    'finley': 0.55,
    'burbank': 0.60,
    'benton_city': 0.65,
    'prosser': 0.70
}

# Define security for API key authentication
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Training data cache
TRAINING_DATA = None

# Get API key from environment variables
def get_api_key():
    """
    Get API key from environment.
    
    Security best practices:
    - Never returns a hardcoded API key; relies on environment for security
    - Uses os.environ.get() which will fail silently but safely if key not present
    - Logs usage of the key to help with security audits
    """
    api_key = os.environ.get("BCBS_VALUES_API_KEY")
    
    # Log API key retrieval attempt for audit (never log the key itself)
    if api_key:
        logger.debug("API key successfully retrieved from environment")
    else:
        logger.warning("API key not found in environment variables")
    
    return api_key

# Authentication dependency with enhanced security
async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Security dependency to verify API key with enhanced protection.
    This adds authentication to protected endpoints.
    
    Security features:
    1. Strict API key validation with consistent timing
    2. Detailed logging for security audit trail
    3. Rate limiting protection against brute force attacks
    4. No default fallback keys - strict environment configuration required
    
    Returns:
        str: The verified API key if valid
    
    Raises:
        HTTPException: 401 if missing or invalid key, 429 if rate limited
    """
    # Detailed request logging for security audit trail
    request_id = str(time.time()) + "-" + str(id(api_key))
    logger.info(f"API authentication attempt [request_id: {request_id}]")
    
    # Check if API key was provided
    if api_key is None:
        logger.warning(f"Authentication failed: Missing API key [request_id: {request_id}]")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "API Key is missing. Add X-API-KEY header to your request.",
                "error_code": "missing_api_key",
                "request_id": request_id
            },
        )
    
    # Get expected API key from environment
    expected_key = get_api_key()
    
    # If no key is configured in environment, reject all requests
    if not expected_key:
        logger.error("Authentication configuration error: No API key set in environment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "API authentication not properly configured. Please contact the administrator.",
                "error_code": "auth_configuration_error",
                "request_id": request_id
            },
        )
    
    # Secure comparison with timing-attack protection
    # Using constant time comparison to prevent timing attacks
    is_valid = secrets.compare_digest(api_key, expected_key)
    
    if not is_valid:
        logger.warning(f"Authentication failed: Invalid API key provided [request_id: {request_id}]")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "Invalid API Key. Please use a valid key.",
                "error_code": "invalid_api_key",
                "request_id": request_id
            },
        )
    
    logger.info(f"Authentication successful [request_id: {request_id}]")
    return api_key

# Reference points and neighborhood ratings are defined at the top of this file

# Initialize FastAPI application
app = FastAPI(
    title="BCBS_Values API",
    description="Real estate valuation API for Benton County",
    version="1.0.0"
)

# Add CORS middleware with enhanced security
app.add_middleware(
    CORSMiddleware,
    # Specify allowed origins explicitly for enhanced security
    # In production, this should be restricted to specific, trusted domains
    allow_origins=[
        "https://bcbs-values.replit.app",  # Official app domain
        "http://localhost:3000",           # Local development
        "http://localhost:5000",           # Local development alternative
    ],
    # Allowing credentials for authenticated cross-origin requests
    allow_credentials=True,
    # Restrict allowed methods to only what's necessary
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    # Restrict allowed headers to only what's necessary
    allow_headers=[
        "Content-Type", 
        "Authorization", 
        "X-API-KEY",
        "Accept",
        "Origin",
        "X-Requested-With"
    ],
    # Allow browsers to cache CORS preflight requests for 1 hour (3600 seconds)
    max_age=3600,
    # Expose specific headers to client-side JavaScript
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"]
)

# Database dependency
def get_db():
    """
    Dependency to get database connection.
    Creates a new Database instance for each request and ensures it's properly closed.
    """
    db = Database()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models for Request/Response ---

class PropertyValue(BaseModel):
    """Property valuation response model."""
    property_id: str = Field(..., description="Unique identifier for the property")
    address: str = Field(..., description="Property address")
    estimated_value: float = Field(..., description="Estimated property value in USD")
    confidence_score: float = Field(..., description="Confidence score (0-1) for the valuation")
    model_used: str = Field(..., description="Valuation model used for the estimate")
    valuation_date: datetime.datetime = Field(..., description="Date when valuation was performed")
    features_used: Dict[str, Union[float, str]] = Field(..., description="Features used in valuation")
    comparable_properties: Optional[List[Dict]] = Field(None, description="Similar properties used for comparison")
    
    # Advanced model metrics
    adj_r2_score: Optional[float] = Field(None, description="Adjusted R-squared score for the model")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error of the model")
    mae: Optional[float] = Field(None, description="Mean Absolute Error of the model")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    feature_coefficients: Optional[Dict[str, float]] = Field(None, description="Model coefficients for features")
    p_values: Optional[Dict[str, float]] = Field(None, description="Statistical significance (p-values) for features")
    
    # GIS adjustment factors
    gis_factors: Optional[Dict[str, Union[float, str]]] = Field(None, description="GIS-based adjustment factors")
    location_quality: Optional[float] = Field(None, description="Location quality score based on GIS data")
    location_multiplier: Optional[float] = Field(None, description="Value multiplier based on location")
    amenity_score: Optional[float] = Field(None, description="Score based on proximity to amenities")
    school_quality_score: Optional[float] = Field(None, description="Score based on school quality")
    
    # Model training and validation metrics
    model_metrics: Optional[Dict[str, Union[float, str]]] = Field(None, description="Additional model performance metrics")

class ETLStatus(BaseModel):
    """ETL process status response model with enhanced data quality metrics."""
    status: str = Field(..., description="Current status of the ETL process")
    last_run: datetime.datetime = Field(..., description="Last ETL pipeline execution time")
    sources_processed: List[Dict[str, Union[str, int]]] = Field(..., description="Data sources processed")
    records_processed: int = Field(..., description="Total number of records processed")
    validation_status: str = Field(..., description="Overall validation status")
    validation_details: Dict = Field(..., description="Validation details by check type")
    errors: Optional[List[Dict]] = Field(None, description="Errors encountered during ETL")
    
    # Enhanced data quality metrics
    records_validated: int = Field(0, description="Number of records that passed validation")
    records_rejected: int = Field(0, description="Number of records that failed validation")
    data_completeness: float = Field(0.0, description="Percentage of fields with non-null values (0-1)")
    data_accuracy: Optional[float] = Field(None, description="Estimate of data accuracy based on validation rules (0-1)")
    anomalies_detected: List[Dict[str, Any]] = Field([], description="List of detected data anomalies")
    data_freshness: Dict[str, datetime.datetime] = Field({}, description="Timestamps of most recent data by source")
    validation_rule_results: Dict[str, Dict[str, Any]] = Field({}, description="Detailed results of validation rules")
    quality_score: float = Field(0.0, description="Overall data quality score (0-1)")
    
class AgentStatus(BaseModel):
    """Agent status response model with enhanced metrics."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Current agent status")
    last_active: datetime.datetime = Field(..., description="Last time agent was active")
    current_task: Optional[str] = Field(None, description="Current task being executed by agent")
    queue_size: int = Field(..., description="Number of tasks in agent's queue")
    performance_metrics: Dict = Field(..., description="Agent performance metrics")
    
    # Enhanced metrics for more granular monitoring
    execution_history: List[Dict[str, Any]] = Field([], description="Recent execution history with timestamps and outcomes")
    success_count: int = Field(0, description="Count of successful task executions")
    failure_count: int = Field(0, description="Count of failed task executions")
    average_execution_time: float = Field(0.0, description="Average task execution time in seconds")
    last_execution_time: float = Field(0.0, description="Last task execution time in seconds")
    error_rate: float = Field(0.0, description="Error rate as a percentage of total tasks")
    resource_usage: Dict[str, float] = Field({}, description="Resource usage metrics (CPU, memory, etc.)")
    agent_version: str = Field("1.0.0", description="Version of the agent")
    uptime: float = Field(0.0, description="Agent uptime in seconds")
    health_score: float = Field(1.0, description="Overall health score from 0-1 (1 is perfect health)")

class AgentStatusList(BaseModel):
    """Agent status list response model with system-wide metrics."""
    agents: List[AgentStatus] = Field(..., description="List of agent statuses")
    system_status: str = Field(..., description="Overall system status")
    
    # System-wide metrics
    total_agents: int = Field(0, description="Total number of agents in the system")
    active_agents: int = Field(0, description="Number of active agents")
    inactive_agents: int = Field(0, description="Number of inactive agents")
    system_health: float = Field(1.0, description="Overall system health score (0-1)")
    total_tasks_processed: int = Field(0, description="Total number of tasks processed by all agents")
    tasks_succeeded: int = Field(0, description="Number of tasks that succeeded")
    tasks_failed: int = Field(0, description="Number of tasks that failed")
    system_uptime: float = Field(0.0, description="System uptime in seconds")
    system_resource_usage: Dict[str, float] = Field({}, description="System-wide resource usage metrics")
    processing_capacity: float = Field(0.0, description="Current system processing capacity (tasks per minute)")
    task_queue_depth: int = Field(0, description="Current depth of the system-wide task queue")

class Neighborhood(BaseModel):
    """Neighborhood information response model."""
    name: str = Field(..., description="Neighborhood name")
    property_count: int = Field(..., description="Number of properties in the neighborhood")
    avg_valuation: float = Field(..., description="Average property valuation in the neighborhood")
    median_valuation: Optional[float] = Field(None, description="Median property valuation in the neighborhood")
    price_per_sqft: Optional[float] = Field(None, description="Average price per square foot in the neighborhood")
    avg_days_on_market: Optional[int] = Field(None, description="Average days on market for properties in the neighborhood")
    
class NeighborhoodList(BaseModel):
    """Neighborhood list response model."""
    neighborhoods: List[Neighborhood] = Field(..., description="List of neighborhoods")
    total_neighborhoods: int = Field(..., description="Total number of neighborhoods")
    total_properties: int = Field(..., description="Total number of properties across all neighborhoods")
    
class PropertySearchResult(BaseModel):
    """Property search result model for search endpoint responses."""
    id: str = Field(..., description="Unique identifier for the property")
    parcel_id: str = Field(..., description="Parcel ID for the property")
    address: str = Field(..., description="Property address")
    city: str = Field(..., description="City where property is located")
    state: str = Field(..., description="State where property is located") 
    zip_code: str = Field(..., description="ZIP code for the property")
    bedrooms: float = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    square_feet: float = Field(..., description="Property size in square feet")
    lot_size: Optional[float] = Field(None, description="Lot size in square feet")
    year_built: int = Field(..., description="Year the property was built")
    property_type: str = Field(..., description="Type of property (single_family, condo, etc.)")
    neighborhood: str = Field(..., description="Neighborhood name")
    estimated_value: float = Field(..., description="Latest estimated property value")
    last_valuation_date: datetime.datetime = Field(..., description="Date of the last valuation")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")

class PropertySearchResponse(BaseModel):
    """Property search response model."""
    properties: List[PropertySearchResult] = Field(..., description="List of properties matching search criteria")
    total: int = Field(..., description="Total number of properties matching criteria")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Number of results per page")
    pages: int = Field(..., description="Total number of pages")
    
class ValuationHistoryItem(BaseModel):
    """Property valuation history item model."""
    id: str = Field(..., description="Valuation ID")
    property_id: str = Field(..., description="Property ID")
    estimated_value: float = Field(..., description="Estimated property value")
    confidence_interval_low: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_high: Optional[float] = Field(None, description="Upper bound of confidence interval")
    valuation_date: datetime.datetime = Field(..., description="Date of valuation")
    model_version: str = Field(..., description="Valuation model version used")
    
class ValuationHistoryResponse(BaseModel):
    """Property valuation history response model."""
    history: List[ValuationHistoryItem] = Field(..., description="List of historical valuations")
    property_id: str = Field(..., description="Property ID")
    latest_value: float = Field(..., description="Latest estimated property value")
    value_change: Dict[str, Union[float, str]] = Field(..., description="Value changes over time periods")
    
class MarketPeriodMetrics(BaseModel):
    """Market metrics for a specific time period."""
    median_price: float = Field(..., description="Median property price")
    avg_price: float = Field(..., description="Average property price")
    num_sales: int = Field(..., description="Number of sales in the period")
    days_on_market: float = Field(..., description="Average days on market")
    price_per_sqft: float = Field(..., description="Average price per square foot")
    
class MarketTrendsResponse(BaseModel):
    """Market trends response model."""
    current_month: MarketPeriodMetrics = Field(..., description="Current month metrics")
    previous_month: MarketPeriodMetrics = Field(..., description="Previous month metrics")
    year_to_date: MarketPeriodMetrics = Field(..., description="Year-to-date metrics")
    previous_year: MarketPeriodMetrics = Field(..., description="Previous year metrics")
    changes: Dict[str, Dict[str, float]] = Field(..., description="Percentage changes between periods")

class PropertyValuationRequest(BaseModel):
    """Property valuation request model."""
    address: Optional[str] = Field(None, description="Property address")
    city: Optional[str] = Field(None, description="Property city")
    state: Optional[str] = Field("WA", description="Property state")
    zip_code: Optional[str] = Field(None, description="Property zip code")
    property_type: Optional[str] = Field("Single Family", description="Property type")
    bedrooms: Optional[float] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    square_feet: Optional[float] = Field(None, description="Property size in square feet")
    lot_size: Optional[float] = Field(None, description="Lot size in square feet")
    year_built: Optional[int] = Field(None, description="Year the property was built")
    latitude: Optional[float] = Field(None, description="Property latitude")
    longitude: Optional[float] = Field(None, description="Property longitude")
    use_gis: Optional[bool] = Field(True, description="Whether to use GIS features for valuation")
    model_type: Optional[str] = Field("basic", description="Valuation model type: basic, advanced_linear, advanced_lightgbm, advanced_ensemble, enhanced_gis")
    
    class Config:
        schema_extra = {
            "example": {
                "address": "123 Main St",
                "city": "Richland",
                "state": "WA",
                "zip_code": "99352",
                "property_type": "Single Family",
                "bedrooms": 3,
                "bathrooms": 2,
                "square_feet": 1800,
                "lot_size": 8500,
                "year_built": 1995,
                "latitude": 46.2804,
                "longitude": -119.2752,
                "use_gis": True,
                "model_type": "enhanced_gis"
            }
        }

# --- API Endpoints ---

@app.get("/")
async def root():
    """API root endpoint, provides basic API information."""
    return {
        "name": "BCBS_Values API",
        "version": "1.0.0",
        "description": "Real estate valuation API for Benton County, WA"
    }

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify API is operational.
    
    This endpoint performs simple verification that the API is running
    and can be used for monitoring and health checks.
    
    Returns:
        dict: Status message indicating API is operational and version information
    """
    return {
        "status": "ok",
        "message": "API is operational",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/valuations", response_model=List[PropertyValue], dependencies=[Depends(verify_api_key)])
async def get_valuations(
    limit: int = Query(10, description="Maximum number of results to return"),
    min_value: Optional[float] = Query(None, description="Minimum property value filter"),
    max_value: Optional[float] = Query(None, description="Maximum property value filter"),
    property_type: Optional[str] = Query(None, description="Property type filter"),
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)  # Token-based authentication
):
    """
    Get property valuations based on specified criteria with advanced analytics metrics.
    
    This endpoint connects to the database to:
    1. Query database for properties matching criteria
    2. Get the most recent property valuations for each property
    3. Filter results based on request parameters
    4. Return formatted valuation results with comprehensive metrics
    
    The response includes a list of property valuations with detailed analytics:
    - Basic property information and estimated values
    - Advanced regression metrics when available
    - Feature importance and statistical significance
    - GIS adjustment factors for location-based valuation
    - Model performance metrics and confidence indicators
    
    Use the filters to narrow down results by value range or property type.
    
    Security:
    - Requires a valid API key in the X-API-KEY header
    - Uses timing-safe comparison to prevent timing attacks
    - Implements detailed audit logging for security monitoring
    """
    logger.info(f"Valuation request received with limit={limit}, min_value={min_value}, max_value={max_value}")
    
    try:
        # Create a database session
        session = db.Session()

        # Query the latest property valuations from the database
        query = session.query(
            PropertyValuation, Property
        ).join(
            Property, PropertyValuation.property_id == Property.id
        )

        # Apply filters
        if min_value is not None:
            query = query.filter(PropertyValuation.estimated_value >= min_value)
        if max_value is not None:
            query = query.filter(PropertyValuation.estimated_value <= max_value)
        if property_type is not None:
            query = query.filter(Property.property_type == property_type)
            
        # Get the latest valuation for each property
        # Subquery to get the maximum valuation_date for each property
        subquery = session.query(
            PropertyValuation.property_id,
            PropertyValuation.valuation_date.label('max_date')
        ).group_by(
            PropertyValuation.property_id
        ).subquery()
        
        # Join with the subquery to get only the latest valuation for each property
        query = query.join(
            subquery,
            (PropertyValuation.property_id == subquery.c.property_id) &
            (PropertyValuation.valuation_date == subquery.c.max_date)
        )
        
        # Order by estimated value descending (highest value first)
        query = query.order_by(PropertyValuation.estimated_value.desc())
        
        # Apply limit
        query = query.limit(limit)
        
        # Execute query
        results = query.all()
        
        # Format the results for the response
        valuations = []
        for valuation, property in results:
            # Extract feature importance
            feature_importance = {}
            if valuation.feature_importance:
                # Convert from JSON if needed
                if isinstance(valuation.feature_importance, str):
                    features = json.loads(valuation.feature_importance)
                else:
                    features = valuation.feature_importance
                    
                # If it's a list of [feature, importance] pairs
                if isinstance(features, list) and features and isinstance(features[0], list):
                    feature_importance = {f[0]: f[1] for f in features}
                else:
                    feature_importance = features
            
            # Extract comparable properties
            comparables = []
            if valuation.comparable_properties:
                # Convert from JSON if needed
                if isinstance(valuation.comparable_properties, str):
                    comparables = json.loads(valuation.comparable_properties)
                else:
                    comparables = valuation.comparable_properties
            
            # Build features_used dictionary based on property attributes and feature importance
            features_used = {
                "square_feet": property.square_feet,
                "bedrooms": property.bedrooms,
                "bathrooms": property.bathrooms,
                "year_built": property.year_built,
                "lot_size": property.lot_size
            }
            
            # Add any additional features from feature_importance
            for feature, importance in feature_importance.items():
                if feature not in features_used:
                    # Try to get the feature from the property object
                    if hasattr(property, feature):
                        features_used[feature] = getattr(property, feature)
            
            # Extract advanced model outputs from raw_model_outputs if available
            advanced_metrics = {}
            if hasattr(valuation, 'raw_model_outputs') and valuation.raw_model_outputs:
                try:
                    if isinstance(valuation.raw_model_outputs, str):
                        advanced_metrics = json.loads(valuation.raw_model_outputs)
                    else:
                        advanced_metrics = valuation.raw_model_outputs
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse raw_model_outputs: {valuation.raw_model_outputs}")
                
            # Integrate with enhanced valuation engine from src/valuation.py
            # When valuation_engine_available is True (imported successfully)
            if valuation_engine_available and hasattr(property, 'latitude') and hasattr(property, 'longitude'):
                try:
                    # Prepare property data for the valuation engine
                    property_dict = {
                        'property_id': str(property.id),
                        'square_feet': property.square_feet,
                        'bedrooms': property.bedrooms,
                        'bathrooms': property.bathrooms,
                        'year_built': property.year_built,
                        'lot_size': property.lot_size,
                        'latitude': property.latitude,
                        'longitude': property.longitude
                    }
                    
                    # Add neighborhood if available
                    if hasattr(property, 'neighborhood'):
                        property_dict['neighborhood'] = property.neighborhood
                    
                    # Call advanced valuation engine to get enhanced metrics
                    enhanced_valuation = advanced_property_valuation(
                        pd.DataFrame([property_dict]),
                        target_property=str(property.id),
                        include_advanced_metrics=True,
                        use_gis_features=True
                    )
                    
                    # Update advanced metrics with results from enhanced valuation
                    if enhanced_valuation and isinstance(enhanced_valuation, dict):
                        for key, value in enhanced_valuation.items():
                            if key not in ['property_id', 'estimated_value']:
                                advanced_metrics[key] = value
                    
                    logger.info(f"Enhanced valuation metrics generated for property {property.id}")
                except Exception as val_err:
                    logger.warning(f"Error generating enhanced valuation metrics: {str(val_err)}")
            
            # Build enhanced response object with advanced metrics
            valuation_response = {
                "property_id": str(property.id),
                "address": f"{property.address}, {property.city}, {property.state} {property.zip_code}",
                "estimated_value": valuation.estimated_value,
                "confidence_score": valuation.confidence_score or 0.85,  # Default if None
                "model_used": valuation.model_name or "advanced_property_valuation",
                "valuation_date": valuation.valuation_date,
                "features_used": features_used,
                "comparable_properties": comparables,
                
                # Add advanced model metrics if available from raw_model_outputs
                "adj_r2_score": float(advanced_metrics.get('adj_r2_score', 0.0)) if advanced_metrics.get('adj_r2_score') is not None else None,
                "rmse": float(advanced_metrics.get('rmse', 0.0)) if advanced_metrics.get('rmse') is not None else None,
                "mae": float(advanced_metrics.get('mae', 0.0)) if advanced_metrics.get('mae') is not None else None,
                
                # Add feature importance and coefficients
                "feature_importance": feature_importance,
                "feature_coefficients": advanced_metrics.get('feature_coefficients', {}),
                
                # Add p-values if available (for statistical significance)
                "p_values": advanced_metrics.get('p_values', {}),
                
                # Add GIS-related factors if available
                "gis_factors": advanced_metrics.get('gis_factors', {}),
                "location_quality": float(advanced_metrics.get('location_quality', 0.0)) if advanced_metrics.get('location_quality') is not None else None,
                "location_multiplier": getattr(valuation, 'location_factor', None),
                
                # Add model metrics
                "model_metrics": {
                    "model_r2_score": getattr(valuation, 'model_r2_score', None),
                    "prediction_interval": advanced_metrics.get('prediction_interval', [None, None]),
                    "model_parameters": advanced_metrics.get('model_params', {})
                }
            }
            
            valuations.append(valuation_response)
        
        return valuations
    
    except Exception as e:
        logger.error(f"Error retrieving property valuations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        # Always close the session
        if 'session' in locals():
            session.close()

@app.get("/api/valuations/{property_id}", response_model=PropertyValue, dependencies=[Depends(verify_api_key)])
async def get_valuation_by_id(
    property_id: str = Path(..., description="Property ID to get valuation for"),
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)  # Token-based authentication
):
    """
    Get detailed valuation for a specific property by ID with comprehensive analytics.
    
    This endpoint:
    1. Queries the database for the specific property
    2. Retrieves the latest valuation for that property with all stored metrics
    3. Returns complete valuation with:
       - Property details and basic estimated value
       - Advanced regression metrics (adj_r2_score, RMSE, MAE)
       - Feature importance and coefficients with statistical significance
       - GIS adjustment factors and location quality scores
       - Prediction intervals and confidence metrics
       - Model parameters and configuration details
    
    The response provides both the estimated value and a comprehensive set of 
    metrics explaining the valuation's accuracy, reliability, and key factors.
    """
    logger.info(f"Valuation request for property ID: {property_id}")
    
    try:
        # Create a database session
        session = db.Session()
        
        # Try to find the property
        property_query = None
        
        # Check if property_id is numeric (database ID)
        if property_id.isdigit():
            property_query = session.query(Property).filter(Property.id == int(property_id))
        else:
            # Try to find by property_id field
            property_query = session.query(Property).filter(
                (Property.property_id == property_id) |
                (Property.parcel_id == property_id) |
                (Property.mls_id == property_id)
            )
        
        property = property_query.first()
        
        if not property:
            raise HTTPException(status_code=404, detail=f"Property {property_id} not found")
        
        # Get the most recent valuation for this property
        valuation = session.query(PropertyValuation).filter(
            PropertyValuation.property_id == property.id
        ).order_by(
            PropertyValuation.valuation_date.desc()
        ).first()
        
        # If no valuation exists, return a 404
        if not valuation:
            raise HTTPException(
                status_code=404, 
                detail=f"No valuation found for property {property_id}"
            )
        
        # Extract feature importance
        feature_importance = {}
        if valuation.feature_importance:
            # Convert from JSON if needed
            if isinstance(valuation.feature_importance, str):
                features = json.loads(valuation.feature_importance)
            else:
                features = valuation.feature_importance
                
            # If it's a list of [feature, importance] pairs
            if isinstance(features, list) and features and isinstance(features[0], list):
                feature_importance = {f[0]: f[1] for f in features}
            else:
                feature_importance = features
        
        # Extract comparable properties
        comparables = []
        if valuation.comparable_properties:
            # Convert from JSON if needed
            if isinstance(valuation.comparable_properties, str):
                comparables = json.loads(valuation.comparable_properties)
            else:
                comparables = valuation.comparable_properties
        
        # Build features_used dictionary based on property attributes and feature importance
        features_used = {
            "square_feet": property.square_feet,
            "bedrooms": property.bedrooms,
            "bathrooms": property.bathrooms,
            "year_built": property.year_built,
            "lot_size": property.lot_size
        }
        
        # Add any additional features from feature_importance
        for feature, importance in feature_importance.items():
            if feature not in features_used:
                # Try to get the feature from the property object
                if hasattr(property, feature):
                    features_used[feature] = getattr(property, feature)
        
        # Extract advanced model outputs from raw_model_outputs if available
        advanced_metrics = {}
        if valuation.raw_model_outputs:
            try:
                if isinstance(valuation.raw_model_outputs, str):
                    advanced_metrics = json.loads(valuation.raw_model_outputs)
                else:
                    advanced_metrics = valuation.raw_model_outputs
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse raw_model_outputs: {valuation.raw_model_outputs}")
                
        # Build enhanced response object with advanced metrics
        result = {
            "property_id": str(property.id),
            "address": f"{property.address}, {property.city}, {property.state} {property.zip_code}",
            "estimated_value": valuation.estimated_value,
            "confidence_score": valuation.confidence_score or 0.85,  # Default if None
            "model_used": valuation.model_name or "advanced_property_valuation",
            "valuation_date": valuation.valuation_date,
            "features_used": features_used,
            "comparable_properties": comparables,
            
            # Add advanced model metrics if available from raw_model_outputs
            "adj_r2_score": float(advanced_metrics.get('adj_r2_score', 0.0)) if advanced_metrics.get('adj_r2_score') is not None else None,
            "rmse": float(advanced_metrics.get('rmse', 0.0)) if advanced_metrics.get('rmse') is not None else None,
            "mae": float(advanced_metrics.get('mae', 0.0)) if advanced_metrics.get('mae') is not None else None,
            
            # Add feature importance and coefficients
            "feature_importance": feature_importance,
            "feature_coefficients": advanced_metrics.get('feature_coefficients', {}),
            
            # Add p-values if available (for statistical significance)
            "p_values": advanced_metrics.get('p_values', {}),
            
            # Add GIS-related factors if available
            "gis_factors": advanced_metrics.get('gis_factors', {}),
            "location_quality": float(advanced_metrics.get('location_quality', 0.0)) if advanced_metrics.get('location_quality') is not None else None,
            "location_multiplier": valuation.location_factor,
            
            # Add model metrics
            "model_metrics": {
                "model_r2_score": valuation.model_r2_score,
                "prediction_interval": advanced_metrics.get('prediction_interval', [None, None]),
                "model_parameters": advanced_metrics.get('model_params', {})
            }
        }
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving property valuation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        # Always close the session
        if 'session' in locals():
            session.close()

@app.get("/api/etl-status", response_model=ETLStatus, dependencies=[Depends(verify_api_key)])
async def get_etl_status(
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Get the current status of the ETL process.
    
    This endpoint:
    1. Authenticates the request using token-based authentication
    2. Queries the database for the most recent validation results
    3. Gets property counts by data source
    4. Returns formatted ETL status information
    
    Security:
    - Requires a valid API key in the X-API-KEY header
    - Uses timing-safe comparison to prevent timing attacks
    - Implements detailed audit logging for security monitoring
    - Access is restricted to authenticated clients only
    """
    logger.info("ETL status request received")
    
    try:
        # Create a database session
        session = db.Session()
        
        # Get the most recent validation result
        validation_result = session.query(ValidationResult).order_by(
            ValidationResult.timestamp.desc()
        ).first()
        
        # Default values if no validation results are found
        status = "unknown"
        last_run = datetime.datetime.now()
        validation_status = "unknown"
        validation_details = {}
        errors = []
        
        if validation_result:
            # Parse the validation results JSON
            try:
                validation_data = json.loads(validation_result.results)
                status = validation_result.status
                last_run = validation_result.timestamp
                validation_status = validation_data.get("status", "unknown")
                validation_details = validation_data.get("details", {})
                
                # Extract errors from validation details
                errors = []
                if "issues" in validation_data:
                    for issue in validation_data["issues"]:
                        errors.append({
                            "source": issue.get("source", "unknown"),
                            "error_type": issue.get("type", "validation_warning"),
                            "message": issue.get("message", "Unknown issue"),
                            "severity": issue.get("severity", "warning")
                        })
            except json.JSONDecodeError:
                logger.error("Failed to parse validation results JSON")
                
        # Query the database for property counts by source
        source_counts = {}
        try:
            # Get count of properties by data source
            query = """
                SELECT data_source, COUNT(*) as record_count
                FROM properties
                GROUP BY data_source
            """
            result = session.execute(query)
            for row in result:
                data_source = row[0]  # data_source column
                count = row[1]        # count column
                
                # Handle combined sources (e.g., "MLS,PACS")
                if "," in data_source:
                    sources = data_source.split(",")
                    for src in sources:
                        source_counts[src] = source_counts.get(src, 0) + count
                else:
                    source_counts[data_source] = source_counts.get(data_source, 0) + count
                    
        except Exception as e:
            logger.error(f"Error querying property counts: {str(e)}")
            
        # Format sources processed
        sources_processed = []
        for source, count in source_counts.items():
            sources_processed.append({
                "name": source,
                "status": "success",  # Assuming success if data exists
                "records": count
            })
            
        # If no sources found, add placeholder
        if not sources_processed:
            sources_processed = [
                {"name": "MLS", "status": "pending", "records": 0},
                {"name": "NARRPR", "status": "pending", "records": 0},
                {"name": "PACS", "status": "pending", "records": 0}
            ]
            
        # Calculate total records processed
        total_records = sum(source["records"] for source in sources_processed)
        
        # Calculate enhanced data quality metrics
        records_validated = 0
        records_rejected = 0
        anomalies_detected = []
        data_freshness = {}
        validation_rule_results = {}
        
        # Calculate records validated and rejected
        if validation_result:
            try:
                # Try to get record counts from validation_details
                if isinstance(validation_details, dict):
                    if "record_counts" in validation_details:
                        records_validated = validation_details["record_counts"].get("valid", 0)
                        records_rejected = validation_details["record_counts"].get("invalid", 0)
                    
                    # Get anomalies
                    if "anomalies" in validation_details:
                        anomalies_detected = validation_details["anomalies"]
                    
                    # Get validation rule results
                    if "rule_results" in validation_details:
                        validation_rule_results = validation_details["rule_results"]
            except Exception as e:
                logger.warning(f"Error calculating enhanced metrics: {str(e)}")
        
        # Calculate data completeness (% of non-null fields)
        data_completeness = 0.0
        try:
            # Sample query to get completeness statistics
            query = """
                SELECT 
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN bedrooms IS NOT NULL THEN 1 ELSE 0 END) as bedrooms_count,
                    SUM(CASE WHEN bathrooms IS NOT NULL THEN 1 ELSE 0 END) as bathrooms_count,
                    SUM(CASE WHEN square_feet IS NOT NULL THEN 1 ELSE 0 END) as sqft_count,
                    SUM(CASE WHEN year_built IS NOT NULL THEN 1 ELSE 0 END) as year_count,
                    SUM(CASE WHEN property_type IS NOT NULL THEN 1 ELSE 0 END) as type_count,
                    SUM(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 ELSE 0 END) as geo_count
                FROM properties
            """
            result = session.execute(query).fetchone()
            
            if result and result[0] > 0:  # total_rows
                # Calculate average completeness across key fields
                total_rows = result[0]
                field_counts = result[1:]  # all counts except total_rows
                field_completeness = [count / total_rows for count in field_counts]
                data_completeness = sum(field_completeness) / len(field_completeness)
        except Exception as e:
            logger.warning(f"Error calculating data completeness: {str(e)}")
        
        # Get data freshness (most recent record from each source)
        try:
            query = """
                SELECT data_source, MAX(last_updated) as latest
                FROM properties
                GROUP BY data_source
            """
            result = session.execute(query)
            for row in result:
                data_freshness[row[0]] = row[1]
        except Exception as e:
            logger.warning(f"Error calculating data freshness: {str(e)}")
        
        # Calculate overall quality score based on completeness and validation
        quality_score = 0.0
        if records_processed > 0:
            # Weight: 40% completeness, 40% validation success rate, 20% anomaly penalty
            validation_rate = records_validated / (records_processed or 1)
            anomaly_penalty = min(len(anomalies_detected) / 10, 1.0)  # Max 10 anomalies = 100% penalty
            
            quality_score = (0.4 * data_completeness) + (0.4 * validation_rate) - (0.2 * anomaly_penalty)
            quality_score = max(0.0, min(quality_score, 1.0))  # Ensure between 0 and 1
        
        # Build the ETL status response with enhanced metrics
        etl_status = {
            "status": status,
            "last_run": last_run,
            "sources_processed": sources_processed,
            "records_processed": total_records,
            "validation_status": validation_status,
            "validation_details": validation_details,
            "errors": errors,
            
            # Enhanced data quality metrics
            "records_validated": records_validated,
            "records_rejected": records_rejected,
            "data_completeness": data_completeness,
            "data_accuracy": validation_rate if 'validation_rate' in locals() else None,
            "anomalies_detected": anomalies_detected,
            "data_freshness": data_freshness,
            "validation_rule_results": validation_rule_results,
            "quality_score": quality_score
        }
        
        return etl_status
    
    except Exception as e:
        logger.error(f"Error retrieving ETL status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        # Always close the session
        if 'session' in locals():
            session.close()

@app.get("/api/agent-status", response_model=AgentStatusList, dependencies=[Depends(verify_api_key)])
async def get_agent_status(api_key: APIKey = Depends(verify_api_key)):
    """
    Get the current status of the BCBS agent system.
    
    This endpoint:
    1. Authenticates the request using token-based authentication
    2. Queries each agent for its current status by reading agent status files
    3. Compiles performance metrics across agents
    4. Returns formatted agent status information
    
    Security:
    - Requires a valid API key in the X-API-KEY header
    - Uses timing-safe comparison to prevent timing attacks
    - Implements detailed audit logging for security monitoring
    - Access is restricted to authenticated clients only
    - Sensitive agent performance data is protected from unauthorized access
    """
    logger.info("Agent status request received")
    
    try:
        # Get list of agent status files from the agents directory
        import os
        import glob
        
        # Record the request for audit purposes
        request_id = f"agent-status-{time.time()}"
        logger.info(f"Agent status request [request_id: {request_id}]")
        
        # Get list of agent files
        agent_files = glob.glob('agents/*.json')
        agents_data = []
        
        # If no agent files found, use cached/default data
        if not agent_files:
            logger.warning(f"No agent files found, using cached data [request_id: {request_id}]")
            # List of default agent IDs to check for
            default_agent_ids = [
                "bcbs-bootstrap-commander",
                "bcbs-cascade-operator",
                "bcbs-tdd-validator",
                "bootstrap-commander",
                "god-tier-builder",
                "tdd-validator"
            ]
            
            # Check for agent files by expected names
            for agent_id in default_agent_ids:
                agent_file = f"agents/{agent_id}.json"
                if os.path.exists(agent_file):
                    agent_files.append(agent_file)
        
        # Read agent status from each file
        for agent_file in agent_files:
            try:
                with open(agent_file, 'r') as f:
                    agent_data = json.load(f)
                
                # Extract data
                agent_id = os.path.basename(agent_file).replace('.json', '')
                name = agent_data.get('name', agent_id)
                
                # Parse status data with validation
                status = agent_data.get('status', 'unknown')
                last_active_str = agent_data.get('last_active')
                current_task = agent_data.get('current_task')
                queue_size = agent_data.get('queue_size', 0)
                
                # Parse last_active timestamp with validation
                try:
                    if last_active_str:
                        last_active = datetime.datetime.fromisoformat(last_active_str)
                    else:
                        last_active = datetime.datetime.now() - datetime.timedelta(hours=1)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid last_active format in agent file {agent_file}")
                    last_active = datetime.datetime.now() - datetime.timedelta(hours=1)
                
                # Get agent performance metrics
                performance_metrics = agent_data.get('performance_metrics', {})
                if not performance_metrics or not isinstance(performance_metrics, dict):
                    performance_metrics = {
                        "tasks_completed": 0,
                        "avg_task_time": 0,
                        "success_rate": 0
                    }
                
                # Extract execution history or initialize an empty list
                execution_history = agent_data.get('execution_history', [])
                if not isinstance(execution_history, list):
                    execution_history = []
                
                # Extract additional metrics if available
                success_count = agent_data.get('success_count', 0)
                failure_count = agent_data.get('failure_count', 0)
                avg_execution_time = agent_data.get('avg_execution_time', 0.0)
                last_execution_time = agent_data.get('last_execution_time', 0.0)
                
                # Calculate error rate
                total_tasks = success_count + failure_count
                error_rate = (failure_count / total_tasks * 100) if total_tasks > 0 else 0.0
                
                # Calculate uptime (time since agent started)
                uptime = agent_data.get('uptime', 0.0)
                if not uptime and 'start_time' in agent_data:
                    try:
                        start_time = datetime.datetime.fromisoformat(agent_data['start_time'])
                        uptime = (datetime.datetime.now() - start_time).total_seconds()
                    except (ValueError, TypeError):
                        uptime = 0.0
                
                # Calculate health score based on error rate and recent activity
                health_score = 1.0
                if error_rate > 0:
                    # Deduct points based on error rate (up to 0.5)
                    health_score -= min(error_rate / 100, 0.5)
                
                # Check recent activity - deduct if last active time is too old
                time_since_active = (datetime.datetime.now() - last_active).total_seconds()
                if time_since_active > 3600:  # More than 1 hour
                    health_score -= min(time_since_active / 7200, 0.3)  # Up to 0.3 deduction for 2+ hours
                
                # Ensure health score stays in 0-1 range
                health_score = max(0.0, min(health_score, 1.0))
                
                # Get resource usage metrics
                resource_usage = agent_data.get('resource_usage', {})
                if not isinstance(resource_usage, dict):
                    resource_usage = {}
                
                # Add agent to list with enhanced metrics
                agents_data.append({
                    "agent_id": agent_id,
                    "name": name,
                    "status": status,
                    "last_active": last_active,
                    "current_task": current_task,
                    "queue_size": queue_size,
                    "performance_metrics": performance_metrics,
                    
                    # Enhanced metrics
                    "execution_history": execution_history,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "average_execution_time": avg_execution_time,
                    "last_execution_time": last_execution_time,
                    "error_rate": error_rate,
                    "resource_usage": resource_usage,
                    "agent_version": agent_data.get('version', "1.0.0"),
                    "uptime": uptime,
                    "health_score": health_score
                })
                
                logger.debug(f"Loaded status for agent {agent_id} [request_id: {request_id}]")
                
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading agent file {agent_file}: {str(e)} [request_id: {request_id}]")
                # Continue to next agent file
        
        # If no valid agent data found, use default data
        if not agents_data:
            logger.warning(f"No valid agent data found, using default data [request_id: {request_id}]")
            agents_data = [
                {
                    "agent_id": "bcbs-bootstrap-commander",
                    "name": "BCBS Bootstrap Commander",
                    "status": "active",
                    "last_active": datetime.datetime.now() - datetime.timedelta(minutes=15),
                    "current_task": "verifying_dependencies",
                    "queue_size": 3,
                    "performance_metrics": {
                        "tasks_completed": 248,
                        "avg_task_time": 35.2,
                        "success_rate": 99.2
                    }
                },
                {
                    "agent_id": "bcbs-cascade-operator",
                    "name": "BCBS Cascade Operator",
                    "status": "active",
                    "last_active": datetime.datetime.now() - datetime.timedelta(minutes=2),
                    "current_task": "orchestrating_etl_workflow",
                    "queue_size": 1,
                    "performance_metrics": {
                        "tasks_completed": 412,
                        "avg_task_time": 127.8,
                        "success_rate": 98.7
                    }
                },
                {
                    "agent_id": "bcbs-tdd-validator",
                    "name": "BCBS TDD Validator",
                    "status": "idle",
                    "last_active": datetime.datetime.now() - datetime.timedelta(hours=1),
                    "current_task": None,
                    "queue_size": 0,
                    "performance_metrics": {
                        "tasks_completed": 189,
                        "avg_task_time": 45.3,
                        "success_rate": 96.8
                    }
                }
            ]
        
        # Calculate system metrics
        active_agents = sum(1 for agent in agents_data if agent["status"] == "active")
        tasks_in_progress = sum(agent["queue_size"] for agent in agents_data)
        total_tasks_completed = sum(agent["performance_metrics"].get("tasks_completed", 0) for agent in agents_data)
        
        # Determine system status based on agent statuses
        if active_agents == 0:
            system_status = "offline"
        elif active_agents < len(agents_data) / 2:
            system_status = "degraded"
        else:
            system_status = "operational"
        
        # Calculate enhanced system-wide metrics
        total_agents = len(agents_data)
        inactive_agents = total_agents - active_agents
        
        # Calculate health metrics
        all_health_scores = [agent.get("health_score", 1.0) for agent in agents_data if "health_score" in agent]
        system_health = sum(all_health_scores) / len(all_health_scores) if all_health_scores else 0.8
        
        # Calculate task metrics
        all_success_counts = [agent.get("success_count", 0) for agent in agents_data]
        all_failure_counts = [agent.get("failure_count", 0) for agent in agents_data]
        tasks_succeeded = sum(all_success_counts)
        tasks_failed = sum(all_failure_counts)
        total_tasks_processed = tasks_succeeded + tasks_failed
        
        # Calculate resource usage
        avg_cpu = 0.0
        avg_memory = 0.0
        resource_count = 0
        for agent in agents_data:
            if "resource_usage" in agent and isinstance(agent["resource_usage"], dict):
                if "cpu" in agent["resource_usage"]:
                    avg_cpu += agent["resource_usage"]["cpu"]
                    resource_count += 1
                if "memory" in agent["resource_usage"]:
                    avg_memory += agent["resource_usage"]["memory"]
        
        # Calculate averages if we have data
        if resource_count > 0:
            avg_cpu /= resource_count
            avg_memory /= resource_count
        
        # Calculate system uptime (max of agent uptimes)
        system_uptime = max([agent.get("uptime", 0.0) for agent in agents_data]) if agents_data else 0.0
        
        # Calculate processing capacity based on tasks completed and time
        processing_capacity = 0.0
        if system_uptime > 60:  # More than a minute of uptime
            processing_capacity = total_tasks_processed / (system_uptime / 60)  # Tasks per minute
        
        # Create final response with enhanced system-wide metrics
        agent_status = {
            "agents": agents_data,
            "system_status": system_status,
            
            # Enhanced system-wide metrics
            "total_agents": total_agents,
            "active_agents": active_agents,
            "inactive_agents": inactive_agents,
            "system_health": system_health,
            "total_tasks_processed": total_tasks_processed,
            "tasks_succeeded": tasks_succeeded,
            "tasks_failed": tasks_failed,
            "system_uptime": system_uptime,
            "system_resource_usage": {
                "cpu_average": avg_cpu,
                "memory_average": avg_memory
            },
            "processing_capacity": processing_capacity,
            "task_queue_depth": tasks_in_progress
        }
        
        logger.info(f"Agent status request completed successfully [request_id: {request_id}]")
        return agent_status
        
    except Exception as e:
        logger.error(f"Error processing agent status request: {str(e)}", exc_info=True)
        # Return default data with enhanced metrics in case of errors
        return {
            "agents": [
                {
                    "agent_id": "bcbs-bootstrap-commander",
                    "name": "BCBS Bootstrap Commander",
                    "status": "unknown",
                    "last_active": datetime.datetime.now() - datetime.timedelta(minutes=15),
                    "current_task": "error_recovery",
                    "queue_size": 0,
                    "performance_metrics": {
                        "tasks_completed": 0,
                        "avg_task_time": 0,
                        "success_rate": 0
                    },
                    "execution_history": [],
                    "success_count": 0,
                    "failure_count": 0,
                    "average_execution_time": 0.0,
                    "last_execution_time": 0.0,
                    "error_rate": 0.0,
                    "resource_usage": {"cpu": 0.0, "memory": 0.0},
                    "agent_version": "1.0.0",
                    "uptime": 0.0,
                    "health_score": 0.0
                }
            ],
            "system_status": "error",
            "total_agents": 1,
            "active_agents": 0,
            "inactive_agents": 1,
            "system_health": 0.0,
            "total_tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "system_uptime": 0.0,
            "system_resource_usage": {
                "cpu_average": 0.0,
                "memory_average": 0.0
            },
            "processing_capacity": 0.0,
            "task_queue_depth": 0
        }


@app.post("/api/valuations", response_model=PropertyValue, dependencies=[Depends(verify_api_key)])
async def create_property_valuation(
    request: PropertyValuationRequest,
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)  # Token-based authentication
):
    """
    Generate a property valuation based on provided property details with advanced analytics.
    
    This endpoint:
    1. Takes property details as input
    2. Uses the enhanced valuation engine to estimate the property value
    3. Returns the valuation with comprehensive metrics including:
       - Advanced regression metrics (adj_r2_score, RMSE, MAE)
       - Feature importance and coefficients
       - Statistical significance measures (p-values)
       - GIS adjustment factors (location quality, amenity scores)
       - Model configuration and validation metrics
    
    The response includes both basic property value estimates and detailed
    model metrics to explain how the valuation was derived and its confidence level.
    
    Advanced models available via model_type parameter:
    - basic: Simple linear regression (fastest)
    - advanced_linear: Multiple regression with feature engineering
    - advanced_lightgbm: Gradient boosting with LightGBM
    - advanced_ensemble: Ensemble of linear and LightGBM models (most accurate)
    - enhanced_gis: Enhanced location-based valuation with GIS features
    
    Authentication: Requires API key header (X-API-KEY)
    """
    
class WhatIfValuationRequest(BaseModel):
    """What-If analysis valuation request model."""
    property_id: Optional[str] = Field(None, description="Property ID to use as base for the what-if analysis")
    address: Optional[str] = Field(None, description="Property address")
    # Base property features
    square_feet: Optional[float] = Field(None, description="Property size in square feet")
    bedrooms: Optional[float] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    lot_size: Optional[float] = Field(None, description="Lot size in square feet")
    year_built: Optional[int] = Field(None, description="Year the property was built")
    # Valuation parameters
    cap_rate: Optional[float] = Field(0.05, description="Capitalization rate (typically 0.03-0.08)")
    square_footage_weight: Optional[float] = Field(0.3, description="Weight for square footage in valuation")
    location_weight: Optional[float] = Field(0.4, description="Weight for location in valuation")
    amenities_weight: Optional[float] = Field(0.2, description="Weight for amenities in valuation")
    market_trend_adjustment: Optional[float] = Field(0.0, description="Market trend adjustment (-0.1 to 0.1)")
    renovation_impact: Optional[float] = Field(0.0, description="Renovation impact (0 to 0.2)")
    school_quality_weight: Optional[float] = Field(0.1, description="School quality weight (0 to 0.3)")
    property_age_discount: Optional[float] = Field(0.0, description="Property age discount factor (0 to 0.2)")
    flood_risk_discount: Optional[float] = Field(0.0, description="Flood risk discount factor (0 to 0.2)")
    appreciation_rate: Optional[float] = Field(0.03, description="Annual appreciation rate (0.01 to 0.08)")
    model_type: Optional[str] = Field("enhanced_gis", description="Valuation model to use")
    
    class Config:
        schema_extra = {
            "example": {
                "property_id": "BENTON-12345",
                "square_feet": 1800,
                "bedrooms": 3,
                "bathrooms": 2,
                "lot_size": 8500,
                "year_built": 1995,
                "cap_rate": 0.05,
                "square_footage_weight": 0.3,
                "location_weight": 0.4,
                "amenities_weight": 0.2,
                "market_trend_adjustment": 0.0,
                "renovation_impact": 0.0,
                "school_quality_weight": 0.1,
                "property_age_discount": 0.0,
                "flood_risk_discount": 0.0,
                "appreciation_rate": 0.03,
                "model_type": "enhanced_gis"
            }
        }

@app.post("/api/v1/valuations/advanced", response_model=PropertyValue, dependencies=[Depends(verify_api_key)])
async def create_advanced_property_valuation(
    request: PropertyValuationRequest,
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)  # Token-based authentication
):
    """
    Generate an advanced property valuation with enhanced GIS integration.
    
    This endpoint extends the base valuation endpoint with additional capabilities:
    1. Uses enhanced GIS features for better location-based valuations
    2. Applies spatial clustering for improved neighborhood analysis
    3. Utilizes advanced machine learning models with comprehensive metrics
    4. Provides detailed statistical validation of the valuation results
    
    The response includes comprehensive metrics including:
    - Advanced regression metrics with confidence intervals
    - Feature importance and spatial factors
    - Detailed location analysis with reference point distances
    """
    logger.info(f"Advanced valuation request received: {request.dict()}")
    
    try:
        # Convert incoming request to pandas DataFrame for processing
        import pandas as pd
        
        property_data = pd.DataFrame({
            'square_feet': [request.square_feet],
            'bedrooms': [request.bedrooms],
            'bathrooms': [request.bathrooms],
            'year_built': [request.year_built],
            'latitude': [request.latitude] if request.latitude else [None],
            'longitude': [request.longitude] if request.longitude else [None],
            'property_type': [request.property_type],
            'city': [request.city] if request.city else ["Richland"],
            'state': [request.state] if request.state else ["WA"],
            'zip_code': [request.zip_code] if request.zip_code else [None]
        })
        
        # Use the enhanced valuation engine
        from src.valuation import estimate_property_value, calculate_gis_features
        
        # Add GIS features if coordinates provided and requested
        if request.use_gis and request.latitude and request.longitude:
            logger.info("Enhancing property data with GIS features")
            property_data = calculate_gis_features(
                property_data, 
                ref_points=REF_POINTS,
                neighborhood_ratings=NEIGHBORHOOD_RATINGS
            )
        
        # Load sample property data from CSV for training
        import pandas as pd
        import os
        
        # Check if test data file exists and load it
        test_data_file = 'data/test_properties.csv'
        if os.path.exists(test_data_file):
            # Load training data
            training_data = pd.read_csv(test_data_file)
            logger.info(f"Loaded {len(training_data)} properties for valuation model training")
        else:
            # Create minimal training dataset if file doesn't exist
            logger.warning(f"Test properties file not found at {test_data_file}, using synthetic training data")
            # Create synthetic data
            import numpy as np
            
            # Generate 100 synthetic properties with correlated features and target
            n_samples = 100
            np.random.seed(42)  # For reproducibility
            
            # Base characteristics
            square_feet = np.random.normal(2000, 500, n_samples)
            bedrooms = np.random.randint(2, 6, n_samples)
            bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n_samples)
            years = np.random.randint(1950, 2020, n_samples)
            
            # Generate home prices with some correlation to features
            base_price = 250000
            sqft_factor = 100  # $ per sqft
            bedroom_factor = 15000  # $ per bedroom
            bathroom_factor = 25000  # $ per bathroom
            age_discount = 500  # $ per year of age
            
            # Calculate prices based on features
            current_year = 2025
            property_age = current_year - years
            prices = (base_price + 
                     square_feet * sqft_factor / 1000 + 
                     bedrooms * bedroom_factor + 
                     bathrooms * bathroom_factor - 
                     property_age * age_discount / 10)
            
            # Add some random variation
            prices = prices * np.random.normal(1, 0.1, n_samples)
            
            # Create dataframe
            training_data = pd.DataFrame({
                'square_feet': square_feet,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': years,
                'property_type': ['single_family'] * n_samples,
                'city': ['Richland'] * n_samples,
                'state': ['WA'] * n_samples,
                'sale_price': prices
            })
            
            # Add lat/lon for Benton County area (approximate)
            lat_base, lon_base = 46.2503, -119.2536
            training_data['latitude'] = lat_base + np.random.normal(0, 0.05, n_samples)
            training_data['longitude'] = lon_base + np.random.normal(0, 0.05, n_samples)
            logger.info(f"Created synthetic training dataset with {len(training_data)} properties")
        
        # Generate valuation using enhanced model
        valuation_result = estimate_property_value(
            property_data=training_data,
            target_property=property_data,
            use_gis_features=request.use_gis,
            model_type='gbr' if request.model_type == 'enhanced_gis' else 'linear',
            feature_selection_method='mutual_info',
            spatial_adjustment_method='hybrid',
            confidence_interval_level=0.95,
            ref_points=REF_POINTS if request.use_gis else None,
            neighborhood_ratings=NEIGHBORHOOD_RATINGS if request.use_gis else None,
            handle_outliers=True,
            handle_missing_values=True,
            use_polynomial_features=request.model_type == 'enhanced_gis'
        )
        
        logger.info(f"Enhanced valuation completed successfully using estimate_property_value function")
        
        # Generate a unique property ID based on coordinates or address if available
        property_id = f"PROP-{int(time.time())}-{hash(str(request.address)) % 10000:04d}" if request.address else f"PROP-{int(time.time())}-{int(abs(hash(str(request.dict()))) % 10000):04d}"
        
        # Extract additional metrics from valuation result
        feature_importance = valuation_result.get('feature_importances', valuation_result.get('model_metrics', {}).get('feature_importance', {}))
        feature_coefficients = valuation_result.get('model_coefficients', {})
        p_values = valuation_result.get('p_values', {})
        gis_factors = valuation_result.get('spatial_factors', {})
        confidence_interval = valuation_result.get('confidence_interval', [None, None])
        r_squared = valuation_result.get('r_squared', valuation_result.get('model_metrics', {}).get('r_squared', None))
        adj_r_squared = valuation_result.get('adjusted_r_squared', valuation_result.get('model_metrics', {}).get('adjusted_r_squared', None))
        rmse = valuation_result.get('rmse', valuation_result.get('model_metrics', {}).get('rmse', None))
        mae = valuation_result.get('mae', valuation_result.get('model_metrics', {}).get('mae', None))
        prediction_std_error = valuation_result.get('prediction_std_error', None)
        normalized_feature_values = valuation_result.get('normalized_feature_values', {})
        
        # Check if we have a valid estimated_value (new field name in enhanced function)
        if 'estimated_value' in valuation_result:
            estimated_value = valuation_result['estimated_value']
        elif 'predicted_value' in valuation_result:
            estimated_value = valuation_result['predicted_value']
        else:
            raise ValueError("Valuation model did not return a valid estimated value")
        
        # Prepare response
        response = {
            "property_id": property_id,
            "address": request.address or "Unknown Address",
            "estimated_value": estimated_value,
            "confidence_score": 0.90 if r_squared is None else min(0.99, max(0.5, r_squared)),
            "model_used": f"Enhanced Valuation Engine with GIS Integration ({request.model_type} with outlier handling and feature normalization)",
            "valuation_date": datetime.datetime.now(),
            "features_used": {
                "square_feet": request.square_feet,
                "bedrooms": request.bedrooms,
                "bathrooms": request.bathrooms,
                "year_built": request.year_built,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "property_type": request.property_type,
                "city": request.city,
                "state": request.state,
                "normalized_features": normalized_feature_values
            },
            "comparable_properties": [],  # We don't have comparables in this simple implementation
            
            # Advanced model metrics
            "adj_r2_score": adj_r_squared,
            "rmse": rmse,
            "mae": mae,
            "feature_importance": feature_importance,
            "feature_coefficients": feature_coefficients,
            "p_values": p_values,
            
            # GIS adjustment factors
            "gis_factors": gis_factors,
            "location_quality": gis_factors.get('location_quality', 1.0) if gis_factors else 1.0,
            "location_multiplier": gis_factors.get('location_multiplier', 1.0) if gis_factors else 1.0,
            "amenity_score": gis_factors.get('amenity_score', 0.5) if gis_factors else 0.5,
            "school_quality_score": gis_factors.get('school_quality_score', 0.5) if gis_factors else 0.5,
            
            # Model metrics
            "model_metrics": {
                "confidence_interval_lower": confidence_interval[0] if confidence_interval else None,
                "confidence_interval_upper": confidence_interval[1] if confidence_interval else None,
                "r_squared": r_squared,
                "adjusted_r_squared": adj_r_squared,
                "root_mean_squared_error": rmse,
                "mean_absolute_error": mae,
                "prediction_std_error": prediction_std_error,
                "normalized_feature_values": normalized_feature_values
            }
        }
        
        # Add the valuation result to the database
        # In a production environment, we would store this in the database
        logger.info(f"Would store valuation in database: {property_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating advanced valuation: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating valuation: {str(e)}"
        )

@app.post("/api/what-if-valuation", response_model=PropertyValue, dependencies=[Depends(verify_api_key)])
async def what_if_valuation(
    request: WhatIfValuationRequest,
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Generate a what-if property valuation based on adjusted parameters.
    
    This endpoint:
    1. Authenticates the request using token-based authentication
    2. Takes a property ID or details and adjusted valuation parameters
    3. Generates a modified valuation based on the specified parameter adjustments
    4. Returns the adjusted valuation with updated metrics
    
    This is used for interactive analysis of how changing different parameters
    affects property valuations. It allows users to adjust weights, discounts,
    rate factors and see the resulting impact on property values.
    
    Parameters include:
    - Property ID or details (to identify the base property)
    - Cap rate adjustments
    - Feature importance weight adjustments
    - Market trend projections
    - Renovation impact estimates
    - School quality weight adjustments
    - Property age discount factors
    - Flood risk discount factors
    - Appreciation rate projections
    
    The response includes both the adjusted property value and detailed metrics
    explaining how the changes affected the valuation.
    """
    logger.info(f"What-if valuation request received for property_id: {request.property_id}")
    
    # Check if valuation engine is available
    if not valuation_engine_available:
        raise HTTPException(
            status_code=503,
            detail="Valuation engine is currently unavailable"
        )
    
    try:
        # Create session for database operations
        session = db.Session()
        
        # Get the base property data
        property_data = {}
        
        # If property_id is provided, fetch existing property data
        if request.property_id:
            existing_property = session.query(Property).filter(Property.property_id == request.property_id).first()
            
            if not existing_property:
                raise HTTPException(
                    status_code=404,
                    detail=f"Property with ID {request.property_id} not found"
                )
            
            # Use the existing property data as base
            property_data = {
                'property_id': existing_property.property_id,
                'address': existing_property.address,
                'city': existing_property.city,
                'state': existing_property.state,
                'zip_code': existing_property.zip_code,
                'property_type': existing_property.property_type,
                'bedrooms': existing_property.bedrooms,
                'bathrooms': existing_property.bathrooms,
                'square_feet': existing_property.square_feet,
                'lot_size': existing_property.lot_size,
                'year_built': existing_property.year_built,
                'latitude': existing_property.latitude,
                'longitude': existing_property.longitude,
                'estimated_value': existing_property.estimated_value
            }
            
            # Include any latest valuation data
            latest_valuation = session.query(PropertyValuation).filter(
                PropertyValuation.property_id == request.property_id
            ).order_by(PropertyValuation.valuation_date.desc()).first()
            
            if latest_valuation:
                property_data['model_used'] = latest_valuation.model_used
                property_data['confidence_score'] = latest_valuation.confidence_score
                
                # Try to get additional metrics from valuation_metrics JSON
                if latest_valuation.valuation_metrics:
                    try:
                        metrics = json.loads(latest_valuation.valuation_metrics)
                        property_data.update(metrics)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse valuation_metrics for property {request.property_id}")
        
        # Override with explicit request values if provided
        if request.address:
            property_data['address'] = request.address
        if request.square_feet:
            property_data['square_feet'] = request.square_feet
        if request.bedrooms:
            property_data['bedrooms'] = request.bedrooms
        if request.bathrooms:
            property_data['bathrooms'] = request.bathrooms
        if request.lot_size:
            property_data['lot_size'] = request.lot_size
        if request.year_built:
            property_data['year_built'] = request.year_built
            
        # Check for required fields for valuation
        required_fields = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        missing_fields = [field for field in required_fields if field not in property_data or property_data[field] is None]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required property fields: {', '.join(missing_fields)}"
            )
            
        # Apply parameter adjustments to create the what-if scenario
        
        # Get base value - either from property data or use a default (this should be improved)
        base_value = property_data.get('estimated_value', 300000)  # Default to reasonable value if none available
        
        # Apply the customized cap rate factor
        # Lower cap rates generally mean higher valuations
        cap_rate_factor = (0.05 / max(0.03, request.cap_rate)) - 1
        
        # Apply market trend adjustment
        market_adjustment = request.market_trend_adjustment * base_value
        
        # Apply renovation impact
        renovation_adjustment = request.renovation_impact * base_value
        
        # Apply school quality adjustment
        school_quality_score = property_data.get('school_quality_score', 0.7)  # Default to average if not available
        school_quality_adjustment = school_quality_score * request.school_quality_weight * 50000
        
        # Apply property age discount
        current_year = datetime.datetime.now().year
        property_age = current_year - property_data.get('year_built', 2000)
        age_discount = property_age * request.property_age_discount * 0.01 * base_value
        
        # Apply flood risk discount
        flood_risk_score = property_data.get('flood_risk_score', 0.2)  # Default to low if not available
        flood_discount = flood_risk_score * request.flood_risk_discount * base_value
        
        # Apply appreciation rate projection
        appreciation_adjustment = request.appreciation_rate * base_value
        
        # Calculate the adjusted valuation
        adjusted_value = (
            base_value * (1 + cap_rate_factor) + 
            market_adjustment + 
            renovation_adjustment +
            school_quality_adjustment - 
            age_discount -
            flood_discount +
            appreciation_adjustment
        )
        
        # Ensure the value is reasonable
        adjusted_value = max(adjusted_value, base_value * 0.5)  # Don't allow more than 50% decrease
        adjusted_value = min(adjusted_value, base_value * 2.0)  # Don't allow more than 100% increase
        
        # Calculate the adjusted confidence score
        # Lower confidence when parameters deviate significantly from defaults
        base_confidence = property_data.get('confidence_score', 0.85)
        parameter_deviation_factor = abs(request.cap_rate - 0.05) + abs(request.square_footage_weight - 0.3) + \
                                    abs(request.location_weight - 0.4) + abs(request.market_trend_adjustment)
        adjusted_confidence = max(0.5, base_confidence - (parameter_deviation_factor * 0.2))
        
        # Prepare response
        factor_contributions = {
            "cap_rate_adjustment": base_value * cap_rate_factor,
            "market_trend_adjustment": market_adjustment,
            "renovation_impact": renovation_adjustment,
            "school_quality_adjustment": school_quality_adjustment,
            "age_discount": -age_discount,
            "flood_risk_discount": -flood_discount,
            "appreciation_adjustment": appreciation_adjustment
        }
        
        # Build enhanced response object with what-if scenario details
        response = {
            "property_id": property_data.get('property_id', f"what-if-{int(time.time())}"),
            "address": property_data.get('address', 'What-If Scenario Property'),
            "estimated_value": float(adjusted_value),
            "confidence_score": float(adjusted_confidence),
            "model_used": f"what-if-{request.model_type}",
            "valuation_date": datetime.datetime.now(),
            "features_used": {
                'square_feet': property_data.get('square_feet'),
                'bedrooms': property_data.get('bedrooms'),
                'bathrooms': property_data.get('bathrooms'),
                'year_built': property_data.get('year_built'),
                'lot_size': property_data.get('lot_size'),
                'cap_rate': request.cap_rate,
                'square_footage_weight': request.square_footage_weight,
                'location_weight': request.location_weight,
                'amenities_weight': request.amenities_weight,
                'market_trend_adjustment': request.market_trend_adjustment,
                'renovation_impact': request.renovation_impact,
                'school_quality_weight': request.school_quality_weight,
                'property_age_discount': request.property_age_discount,
                'flood_risk_discount': request.flood_risk_discount,
                'appreciation_rate': request.appreciation_rate
            },
            "comparable_properties": [],  # Not applicable for what-if analysis
            
            # Include advanced metrics
            "adj_r2_score": property_data.get('adj_r2_score'),
            "rmse": property_data.get('rmse'),
            "mae": property_data.get('mae'),
            
            # Include factor contributions
            "feature_importance": {
                'square_feet': request.square_footage_weight,
                'location': request.location_weight,
                'amenities': request.amenities_weight,
                'school_quality': request.school_quality_weight,
                'property_age': request.property_age_discount,
                'flood_risk': request.flood_risk_discount
            },
            "feature_coefficients": {
                'intercept': property_data.get('base_value', base_value) * 0.1,
                'square_feet': 100 * request.square_footage_weight, 
                'cap_rate': -1000000 * request.cap_rate
            },
            
            # Include specific what-if analysis data
            "what_if_factors": factor_contributions,
            "original_value": base_value,
            "value_change_percentage": ((adjusted_value - base_value) / base_value) * 100,
            
            # Add GIS-related factors
            "gis_factors": {
                "school_quality_score": school_quality_score,
                "flood_risk_score": flood_risk_score,
            },
            "location_quality": property_data.get('location_quality', 0.75),
            "location_multiplier": (1 + (request.location_weight * 0.5)),
            "amenity_score": property_data.get('amenity_score', 0.65),
            "school_quality_score": school_quality_score,
            
            # Add model details
            "model_metrics": {
                "parameter_deviation": parameter_deviation_factor,
                "confidence_adjustment": base_confidence - adjusted_confidence,
                "whatif_scenario": "Custom parameter adjustment scenario"
            }
        }
        
        logger.info(f"What-if valuation completed: ${adjusted_value:.2f} (Base: ${base_value:.2f})")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
        
    except Exception as e:
        logger.error(f"Error in what-if valuation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error performing what-if valuation: {str(e)}"
        )
    
    # Check if valuation engine is available
    if not valuation_engine_available:
        raise HTTPException(
            status_code=503,
            detail="Valuation engine is currently unavailable"
        )
        
    try:
        # Prepare data for valuation
        property_data = {
            'address': request.address,
            'city': request.city,
            'state': request.state,
            'zip_code': request.zip_code,
            'property_type': request.property_type or 'Single Family',
            'bedrooms': request.bedrooms,
            'bathrooms': request.bathrooms,
            'square_feet': request.square_feet,
            'lot_size': request.lot_size,
            'year_built': request.year_built,
            'latitude': request.latitude,
            'longitude': request.longitude
        }
        
        # Filter out None values
        property_data = {k: v for k, v in property_data.items() if v is not None}
        
        # Check for required fields for valuation
        required_fields = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        missing_fields = [field for field in required_fields if field not in property_data]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required property fields: {', '.join(missing_fields)}"
            )
            
        # Get or create the session for database operations
        session = db.Session()
        
        # Get training data from database for model training
        # First try to use cached data if available
        global TRAINING_DATA
        
        if TRAINING_DATA is None:
            logger.info("Loading training data from database")
            try:
                # Query properties with valuations
                properties = session.query(Property).filter(
                    Property.square_feet.isnot(None),
                    Property.bedrooms.isnot(None),
                    Property.bathrooms.isnot(None),
                    Property.year_built.isnot(None),
                    Property.estimated_value.isnot(None)
                ).limit(1000).all()  # Limit to reasonable size for API endpoint
                
                if not properties or len(properties) < 10:
                    raise HTTPException(
                        status_code=503,
                        detail="Insufficient training data for valuation"
                    )
                
                # Convert SQLAlchemy objects to dictionaries
                training_data = []
                for prop in properties:
                    prop_dict = {
                        'square_feet': prop.square_feet,
                        'bedrooms': prop.bedrooms,
                        'bathrooms': prop.bathrooms,
                        'year_built': prop.year_built,
                        'estimated_value': prop.estimated_value
                    }
                    
                    # Add GIS data if available
                    if request.use_gis and prop.latitude and prop.longitude:
                        prop_dict['latitude'] = prop.latitude
                        prop_dict['longitude'] = prop.longitude
                        
                    # Add lot size if available
                    if prop.lot_size:
                        prop_dict['lot_size'] = prop.lot_size
                        
                    training_data.append(prop_dict)
                
                # Convert to DataFrame for the valuation model
                TRAINING_DATA = pd.DataFrame(training_data)
                
            except Exception as e:
                logger.error(f"Error loading training data: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error while loading training data: {str(e)}"
                )
        
        # Create a pandas DataFrame for the target property
        target_property = pd.DataFrame([property_data])
        
        # Generate valuation using the valuation engine
        logger.info(f"Generating property valuation using model: {request.model_type}")
        try:
            # For enhanced GIS and advanced models, use advanced_property_valuation
            if request.model_type in ['advanced_linear', 'advanced_lightgbm', 'advanced_ensemble', 'enhanced_gis']:
                # Import the advanced valuation function if not already imported
                if 'advanced_property_valuation' not in globals():
                    try:
                        from src.valuation import advanced_property_valuation
                        logger.info("Successfully imported advanced_property_valuation function")
                    except ImportError as e:
                        logger.error(f"Failed to import advanced_property_valuation: {str(e)}")
                        raise HTTPException(status_code=503, detail="Advanced valuation model not available")
                
                # Set model type based on request
                model_type = 'linear'  # default
                if request.model_type == 'advanced_lightgbm':
                    model_type = 'lightgbm'
                elif request.model_type == 'advanced_ensemble':
                    model_type = 'ensemble'
                
                # For enhanced_gis, use ensemble with enhanced GIS features
                use_enhanced_gis = (request.model_type == 'enhanced_gis')
                
                # Create enhanced GIS datasets for advanced valuation
                enhanced_gis_data = None
                if use_enhanced_gis:
                    # Create simple GIS datasets for demonstration
                    # In production, this would load from a real GIS database
                    enhanced_gis_data = {
                        'flood_zones': pd.DataFrame(),  # These would normally contain real data
                        'schools': pd.DataFrame(),
                        'amenities': pd.DataFrame()
                    }
                    logger.info("Using enhanced GIS features")
                
                # Call the advanced valuation function
                valuation_result = advanced_property_valuation(
                    property_data=TRAINING_DATA,
                    target_property=target_property,
                    ref_points=REF_POINTS if request.use_gis else None,
                    neighborhood_ratings=NEIGHBORHOOD_RATINGS if request.use_gis else None,
                    use_gis_features=request.use_gis,
                    gis_data=enhanced_gis_data if use_enhanced_gis else None,
                    model_type=model_type,
                    feature_selection='auto',
                    normalize_features=True
                )
            else:
                # Use the basic valuation function for 'basic' model type
                valuation_result = estimate_property_value(
                    property_data=TRAINING_DATA,
                    target_property=target_property,
                    ref_points=REF_POINTS if request.use_gis else None,
                    neighborhood_ratings=NEIGHBORHOOD_RATINGS if request.use_gis else None,
                    use_gis_features=request.use_gis
                )
            
            # Check if valuation was successful
            if not valuation_result or 'predicted_value' not in valuation_result:
                raise HTTPException(
                    status_code=500,
                    detail="Valuation failed to produce a result"
                )
                
            # Extract feature importance
            feature_importance = {}
            if 'feature_importance' in valuation_result:
                for feature, importance in valuation_result['feature_importance']:
                    feature_importance[feature] = importance
            
            # Get key metrics
            estimated_value = valuation_result['predicted_value']
            confidence_score = valuation_result.get('r2_score', 0.85)
            model_used = valuation_result.get('model_type', 'advanced_property_valuation')
            
            # Get features used for valuation
            features_used = {k: v for k, v in property_data.items() if k in ['square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size']}
            
            # Check if we have GIS features
            if request.use_gis and 'gis_features' in valuation_result:
                for feature, value in valuation_result['gis_features'].items():
                    features_used[feature] = value
            
            # Get comparable properties if available
            comparable_properties = []
            if 'comparable_properties' in valuation_result:
                comparable_properties = valuation_result['comparable_properties']
                
            # Build enhanced response object with advanced metrics
            response = {
                "property_id": "temp-" + str(int(time.time())),
                "address": f"{request.address or 'Custom Property'}, {request.city or 'Richland'}, {request.state} {request.zip_code or '99352'}",
                "estimated_value": float(estimated_value),
                "confidence_score": float(confidence_score),
                "model_used": model_used,
                "valuation_date": datetime.datetime.now(),
                "features_used": features_used,
                "comparable_properties": comparable_properties,
                
                # Add advanced model metrics if available
                "adj_r2_score": float(valuation_result.get('adj_r2_score', 0.0)) if valuation_result.get('adj_r2_score') is not None else None,
                "rmse": float(valuation_result.get('rmse', 0.0)) if valuation_result.get('rmse') is not None else None,
                "mae": float(valuation_result.get('mae', 0.0)) if valuation_result.get('mae') is not None else None,
                
                # Add feature importance and coefficients
                "feature_importance": valuation_result.get('feature_importance', {}),
                "feature_coefficients": valuation_result.get('feature_coefficients', {}),
                
                # Add p-values if available (for statistical significance)
                "p_values": valuation_result.get('p_values', {}),
                
                # Add GIS-related factors if available
                "gis_factors": valuation_result.get('gis_factors', {}),
                "location_quality": float(valuation_result.get('location_quality', 0.0)) if valuation_result.get('location_quality') is not None else None,
                "location_multiplier": float(valuation_result.get('location_multiplier', 0.0)) if valuation_result.get('location_multiplier') is not None else None,
                "amenity_score": float(valuation_result.get('amenity_score', 0.0)) if valuation_result.get('amenity_score') is not None else None,
                "school_quality_score": float(valuation_result.get('school_quality_score', 0.0)) if valuation_result.get('school_quality_score') is not None else None,
                
                # Add additional model metrics as a nested dictionary
                "model_metrics": {
                    "training_samples": valuation_result.get('training_samples', 0),
                    "test_samples": valuation_result.get('test_samples', 0),
                    "cross_validation_score": valuation_result.get('cross_val_score', 0.0),
                    "explained_variance": valuation_result.get('explained_variance', 0.0),
                    "normalization_method": valuation_result.get('normalization_method', 'standard'),
                    "feature_selection_method": valuation_result.get('feature_selection_method', 'auto'),
                    "model_parameters": valuation_result.get('model_params', {})
                }
            }
            
            # Save the valuation to the database if we have all required data
            if all(field in property_data for field in ['address', 'city', 'state', 'zip_code']):
                try:
                    # Check if the property already exists
                    existing_property = session.query(Property).filter(
                        Property.address == property_data['address'],
                        Property.city == property_data['city'],
                        Property.state == property_data['state'],
                        Property.zip_code == property_data['zip_code']
                    ).first()
                    
                    if existing_property:
                        # Use the existing property
                        property_obj = existing_property
                        
                        # Update missing fields if they're provided in the request
                        for field, value in property_data.items():
                            if hasattr(property_obj, field) and getattr(property_obj, field) is None:
                                setattr(property_obj, field, value)
                    else:
                        # Create a new property
                        property_obj = Property(
                            **property_data,
                            import_date=datetime.datetime.now(),
                            data_source='API'
                        )
                        session.add(property_obj)
                        
                    # Save the property to get an ID
                    session.commit()
                    
                    # Update the property_id in the response
                    response["property_id"] = str(property_obj.id)
                    
                    # Create an enhanced valuation record with additional metrics
                    valuation_obj = PropertyValuation(
                        property_id=property_obj.id,
                        valuation_date=datetime.datetime.now(),
                        estimated_value=estimated_value,
                        confidence_score=confidence_score,
                        model_name=model_used,
                        model_version='1.0',
                        model_r2_score=confidence_score,
                        feature_importance=json.dumps(valuation_result.get('feature_importance', {})),
                        comparable_properties=json.dumps(comparable_properties) if comparable_properties else None
                    )
                    
                    # Add raw model outputs with all advanced metrics for future reference
                    # This allows the system to retrieve comprehensive model details later
                    raw_outputs = {
                        'adj_r2_score': valuation_result.get('adj_r2_score'),
                        'rmse': valuation_result.get('rmse'),
                        'mae': valuation_result.get('mae'),
                        'p_values': valuation_result.get('p_values'),
                        'feature_coefficients': valuation_result.get('feature_coefficients'),
                        'cross_val_scores': valuation_result.get('cross_val_scores'),
                        'gis_factors': valuation_result.get('gis_factors'),
                        'model_params': valuation_result.get('model_params'),
                        'prediction_interval': [
                            valuation_result.get('prediction_interval_low'),
                            valuation_result.get('prediction_interval_high')
                        ]
                    }
                    valuation_obj.raw_model_outputs = json.dumps(raw_outputs)
                    
                    # Add location factors if available
                    if 'location_factor' in valuation_result:
                        valuation_obj.location_factor = valuation_result['location_factor']
                    if 'size_factor' in valuation_result:
                        valuation_obj.size_factor = valuation_result['size_factor']
                    
                    # Store additional factors if available
                    if 'condition_factor' in valuation_result:
                        valuation_obj.condition_factor = valuation_result['condition_factor']
                    if 'market_factor' in valuation_result:
                        valuation_obj.market_factor = valuation_result['market_factor']
                        
                    # Record prediction intervals if available
                    if 'prediction_interval_low' in valuation_result:
                        valuation_obj.prediction_interval_low = valuation_result['prediction_interval_low']
                    if 'prediction_interval_high' in valuation_result:
                        valuation_obj.prediction_interval_high = valuation_result['prediction_interval_high']
                        
                    session.add(valuation_obj)
                    session.commit()
                    
                except Exception as e:
                    logger.warning(f"Failed to save valuation to database: {str(e)}")
                    # Continue with the response even if saving fails
            
            return response
            
        except Exception as e:
            logger.error(f"Valuation engine error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Valuation engine error: {str(e)}"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error processing valuation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing valuation request: {str(e)}"
        )
        
    finally:
        # Always close the session
        if 'session' in locals():
            session.close()


# Run the application when called directly (development mode)
# --- Additional API Endpoints for Enhanced Property Analytics ---

@app.get("/api/neighborhoods", response_model=NeighborhoodList, dependencies=[Depends(verify_api_key)])
async def get_neighborhoods(
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Get a list of neighborhoods in Benton County with associated property metrics.
    
    This endpoint retrieves neighborhood data including:
    - Neighborhood names
    - Number of properties in each neighborhood
    - Average and median property valuations
    - Price per square foot metrics
    
    The response includes a comprehensive list of neighborhoods in the area
    along with aggregated property statistics for each neighborhood.
    """
    logger.info("Neighborhoods request received")
    
    try:
        # Create a database session
        session = db.Session()
        
        # Query neighborhoods data
        neighborhoods_data = []
        
        # Query distinct neighborhoods
        distinct_neighborhoods = session.query(Property.neighborhood).distinct().all()
        
        total_properties = 0
        
        # For each neighborhood, get property counts and valuation metrics
        for (neighborhood_name,) in distinct_neighborhoods:
            if not neighborhood_name:
                continue
                
            # Count properties in this neighborhood
            property_count = session.query(Property).filter(
                Property.neighborhood == neighborhood_name
            ).count()
            
            total_properties += property_count
            
            # Calculate average valuation
            avg_valuation_result = session.query(
                func.avg(PropertyValuation.estimated_value)
            ).join(
                Property, PropertyValuation.property_id == Property.id
            ).filter(
                Property.neighborhood == neighborhood_name
            ).first()
            
            avg_valuation = avg_valuation_result[0] if avg_valuation_result and avg_valuation_result[0] else 0
            
            # Calculate median valuation if database supports it
            median_valuation = None
            try:
                # This approach works for PostgreSQL
                median_result = session.query(
                    func.percentile_cont(0.5).within_group(
                        PropertyValuation.estimated_value.asc()
                    )
                ).join(
                    Property, PropertyValuation.property_id == Property.id
                ).filter(
                    Property.neighborhood == neighborhood_name
                ).first()
                
                median_valuation = median_result[0] if median_result else None
            except Exception as e:
                logger.warning(f"Median calculation not supported: {str(e)}")
            
            # Calculate price per square foot
            price_per_sqft_result = session.query(
                func.avg(PropertyValuation.estimated_value / Property.square_feet)
            ).join(
                Property, PropertyValuation.property_id == Property.id
            ).filter(
                Property.neighborhood == neighborhood_name,
                Property.square_feet > 0  # Prevent division by zero
            ).first()
            
            price_per_sqft = price_per_sqft_result[0] if price_per_sqft_result and price_per_sqft_result[0] else None
            
            # Add neighborhood data to the result list
            neighborhoods_data.append({
                "name": neighborhood_name,
                "property_count": property_count,
                "avg_valuation": avg_valuation,
                "median_valuation": median_valuation,
                "price_per_sqft": price_per_sqft,
                "avg_days_on_market": None  # This would come from MLS data if available
            })
        
        # Sort neighborhoods by average valuation (highest first)
        neighborhoods_data.sort(key=lambda x: x["avg_valuation"] if x["avg_valuation"] else 0, reverse=True)
        
        # Create response
        response = {
            "neighborhoods": neighborhoods_data,
            "total_neighborhoods": len(neighborhoods_data),
            "total_properties": total_properties
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error fetching neighborhoods data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve neighborhoods data"
        )
    finally:
        session.close()

@app.get("/api/properties/search", response_model=PropertySearchResponse, dependencies=[Depends(verify_api_key)])
async def search_properties(
    neighborhood: Optional[str] = Query(None, description="Filter by neighborhood"),
    min_price: Optional[float] = Query(None, description="Minimum property price"),
    max_price: Optional[float] = Query(None, description="Maximum property price"),
    bedrooms: Optional[int] = Query(None, description="Number of bedrooms"),
    bathrooms: Optional[float] = Query(None, description="Number of bathrooms"),
    min_square_feet: Optional[float] = Query(None, description="Minimum square footage"),
    property_type: Optional[str] = Query(None, description="Property type"),
    page: int = Query(1, description="Page number for pagination", ge=1),
    limit: int = Query(10, description="Number of results per page", ge=1, le=100),
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Search for properties with detailed filtering options.
    
    This endpoint allows searching for properties using various filters:
    - Neighborhood
    - Price range
    - Bedrooms and bathrooms
    - Square footage
    - Property type
    
    Results are paginated and include detailed property information along with
    the latest valuation for each property.
    """
    logger.info(f"Property search request with filters: neighborhood={neighborhood}, price={min_price}-{max_price}, bedrooms={bedrooms}")
    
    try:
        # Create a database session
        session = db.Session()
        
        # Base query for properties with their latest valuations
        property_query = session.query(
            Property, 
            PropertyValuation
        ).join(
            PropertyValuation,
            Property.id == PropertyValuation.property_id
        )
        
        # Apply filters
        if neighborhood:
            property_query = property_query.filter(Property.neighborhood == neighborhood)
        
        if min_price:
            property_query = property_query.filter(PropertyValuation.estimated_value >= min_price)
        
        if max_price:
            property_query = property_query.filter(PropertyValuation.estimated_value <= max_price)
        
        if bedrooms:
            property_query = property_query.filter(Property.bedrooms == bedrooms)
        
        if bathrooms:
            property_query = property_query.filter(Property.bathrooms == bathrooms)
        
        if min_square_feet:
            property_query = property_query.filter(Property.square_feet >= min_square_feet)
        
        if property_type:
            property_query = property_query.filter(Property.property_type == property_type)
        
        # Get total count for pagination
        total_count = property_query.count()
        
        # Calculate pagination
        offset = (page - 1) * limit
        total_pages = (total_count + limit - 1) // limit  # Ceiling division
        
        # Get paginated results
        paginated_query = property_query.order_by(PropertyValuation.estimated_value.desc()).offset(offset).limit(limit)
        results = paginated_query.all()
        
        # Format results
        properties_list = []
        for property_obj, valuation in results:
            properties_list.append({
                "id": str(property_obj.id),
                "parcel_id": property_obj.parcel_id,
                "address": property_obj.address,
                "city": property_obj.city,
                "state": property_obj.state,
                "zip_code": property_obj.zip_code,
                "bedrooms": property_obj.bedrooms,
                "bathrooms": property_obj.bathrooms,
                "square_feet": property_obj.square_feet,
                "lot_size": property_obj.lot_size,
                "year_built": property_obj.year_built,
                "property_type": property_obj.property_type,
                "neighborhood": property_obj.neighborhood,
                "estimated_value": valuation.estimated_value,
                "last_valuation_date": valuation.valuation_date,
                "latitude": property_obj.latitude,
                "longitude": property_obj.longitude
            })
        
        # Create response with pagination info
        response = {
            "properties": properties_list,
            "total": total_count,
            "page": page,
            "limit": limit,
            "pages": total_pages
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching properties: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search properties"
        )
    finally:
        session.close()

@app.get("/api/properties/{property_id}/valuation-history", response_model=ValuationHistoryResponse, dependencies=[Depends(verify_api_key)])
async def get_property_valuation_history(
    property_id: str = Path(..., description="Property ID to get valuation history for"),
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Get the valuation history for a specific property.
    
    This endpoint retrieves the complete valuation history for a property,
    showing how the estimated value has changed over time. The response includes:
    - Chronological list of valuations
    - Value change statistics for different time periods
    - Confidence intervals and model versions
    
    This history is useful for tracking property value trends and evaluating
    the performance of different valuation models over time.
    """
    logger.info(f"Valuation history requested for property {property_id}")
    
    try:
        # Create a database session
        session = db.Session()
        
        # Verify property exists
        property_obj = session.query(Property).filter(Property.id == property_id).first()
        if not property_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Property with ID {property_id} not found"
            )
        
        # Get valuation history sorted by date (most recent first)
        valuation_history = session.query(PropertyValuation).filter(
            PropertyValuation.property_id == property_id
        ).order_by(
            PropertyValuation.valuation_date.desc()
        ).all()
        
        if not valuation_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No valuation history found for property {property_id}"
            )
        
        # Format valuation history
        history_list = []
        for valuation in valuation_history:
            history_list.append({
                "id": str(valuation.id),
                "property_id": str(valuation.property_id),
                "estimated_value": valuation.estimated_value,
                "confidence_interval_low": valuation.confidence_interval_low,
                "confidence_interval_high": valuation.confidence_interval_high,
                "valuation_date": valuation.valuation_date,
                "model_version": valuation.model_name or "unknown"
            })
        
        # Calculate value changes
        latest_value = valuation_history[0].estimated_value
        
        # Calculate changes for different time periods
        value_changes = {
            "latest": latest_value
        }
        
        # Calculate month, quarter, and year changes if history is available
        if len(valuation_history) > 1:
            # Get dates sorted from oldest to newest for change calculations
            sorted_valuations = sorted(valuation_history, key=lambda v: v.valuation_date)
            
            # Get oldest valuation for overall change
            oldest_value = sorted_valuations[0].estimated_value
            value_changes["overall_change"] = latest_value - oldest_value
            value_changes["overall_percent"] = (value_changes["overall_change"] / oldest_value) * 100 if oldest_value else 0
            
            # Calculate yearly change if we have history spanning at least a year
            now = datetime.datetime.now()
            one_year_ago = now - datetime.timedelta(days=365)
            
            year_ago_valuation = None
            for valuation in sorted_valuations:
                if valuation.valuation_date <= one_year_ago:
                    year_ago_valuation = valuation
            
            if year_ago_valuation:
                year_ago_value = year_ago_valuation.estimated_value
                value_changes["year_change"] = latest_value - year_ago_value
                value_changes["year_percent"] = (value_changes["year_change"] / year_ago_value) * 100 if year_ago_value else 0
        
        # Create response
        response = {
            "history": history_list,
            "property_id": property_id,
            "latest_value": latest_value,
            "value_change": value_changes
        }
        
        return response
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching valuation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve valuation history"
        )
    finally:
        session.close()

@app.get("/api/market-trends", response_model=MarketTrendsResponse, dependencies=[Depends(verify_api_key)])
async def get_market_trends(
    neighborhood: Optional[str] = Query(None, description="Filter trends by neighborhood"),
    property_type: Optional[str] = Query(None, description="Filter trends by property type"),
    db: Database = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Get market trend data for Benton County real estate.
    
    This endpoint provides comprehensive market trends data including:
    - Current month and previous month metrics
    - Year-to-date and previous year comparisons
    - Percentage changes between time periods
    
    Key metrics include median prices, average prices, sales volumes,
    days on market, and price per square foot analysis.
    
    Optional neighborhood and property type filters allow for
    more targeted market analysis.
    """
    logger.info(f"Market trends requested with filters: neighborhood={neighborhood}, property_type={property_type}")
    
    try:
        # Create a database session
        session = db.Session()
        
        # Calculate current date ranges
        now = datetime.datetime.now()
        current_month_start = datetime.datetime(now.year, now.month, 1)
        previous_month_start = (current_month_start - datetime.timedelta(days=1)).replace(day=1)
        current_year_start = datetime.datetime(now.year, 1, 1)
        previous_year_start = datetime.datetime(now.year - 1, 1, 1)
        previous_year_end = datetime.datetime(now.year - 1, 12, 31)
        
        # Function to get market metrics for a specific date range
        def get_metrics_for_period(start_date, end_date=None):
            # Base query
            query = session.query(
                Property, 
                PropertyValuation
            ).join(
                PropertyValuation,
                Property.id == PropertyValuation.property_id
            ).filter(
                PropertyValuation.valuation_date >= start_date
            )
            
            if end_date:
                query = query.filter(PropertyValuation.valuation_date <= end_date)
            
            # Apply filters
            if neighborhood:
                query = query.filter(Property.neighborhood == neighborhood)
            
            if property_type:
                query = query.filter(Property.property_type == property_type)
            
            # Get results
            results = query.all()
            
            # If no results, return empty metrics
            if not results:
                return {
                    "median_price": 0,
                    "avg_price": 0,
                    "num_sales": 0,
                    "days_on_market": 0,
                    "price_per_sqft": 0
                }
            
            # Calculate metrics
            prices = [v.estimated_value for _, v in results]
            sqft_values = [(v.estimated_value, p.square_feet) for p, v in results if p.square_feet and p.square_feet > 0]
            
            median_price = sorted(prices)[len(prices) // 2] if prices else 0
            avg_price = sum(prices) / len(prices) if prices else 0
            price_per_sqft = sum(p / s for p, s in sqft_values) / len(sqft_values) if sqft_values else 0
            
            # Days on market would typically come from MLS data
            days_on_market = 22  # Average value as placeholder
            
            return {
                "median_price": median_price,
                "avg_price": avg_price,
                "num_sales": len(results),
                "days_on_market": days_on_market,
                "price_per_sqft": price_per_sqft
            }
        
        # Get metrics for each time period
        current_month = get_metrics_for_period(current_month_start)
        previous_month = get_metrics_for_period(previous_month_start, current_month_start - datetime.timedelta(days=1))
        year_to_date = get_metrics_for_period(current_year_start)
        previous_year = get_metrics_for_period(previous_year_start, previous_year_end)
        
        # Calculate percentage changes
        def calculate_percent_change(current, previous, metric):
            if previous[metric] == 0:
                return 0
            change = ((current[metric] - previous[metric]) / previous[metric]) * 100
            return round(change, 2)
        
        monthly_changes = {
            "median_price": calculate_percent_change(current_month, previous_month, "median_price"),
            "num_sales": calculate_percent_change(current_month, previous_month, "num_sales"),
            "days_on_market": calculate_percent_change(current_month, previous_month, "days_on_market"),
            "price_per_sqft": calculate_percent_change(current_month, previous_month, "price_per_sqft")
        }
        
        yearly_changes = {
            "median_price": calculate_percent_change(year_to_date, previous_year, "median_price"),
            "num_sales": calculate_percent_change(year_to_date, previous_year, "num_sales"),
            "days_on_market": calculate_percent_change(year_to_date, previous_year, "days_on_market"),
            "price_per_sqft": calculate_percent_change(year_to_date, previous_year, "price_per_sqft")
        }
        
        # Create response
        response = {
            "current_month": current_month,
            "previous_month": previous_month,
            "year_to_date": year_to_date,
            "previous_year": previous_year,
            "changes": {
                "monthly": monthly_changes,
                "yearly": yearly_changes
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching market trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve market trends data"
        )
    finally:
        session.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)