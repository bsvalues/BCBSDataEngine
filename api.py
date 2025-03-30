"""
FastAPI implementation for the BCBS_Values real estate valuation API.
This module provides HTTP endpoints for property valuation, ETL status, and agent status.
"""
import datetime
import json
import os
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Path, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field

# Import database and models
from db.database import Database
from db.models import PropertyValuation, Property, ValidationResult

# Import valuation engine
try:
    from src.valuation import estimate_property_value, train_basic_valuation_model, train_multiple_regression_model
    valuation_engine_available = True
except ImportError:
    # Log the error but don't crash - API can still work with other endpoints
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

# Define security for API key authentication
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Training data cache
TRAINING_DATA = None

# Get API key from environment variables
def get_api_key():
    """Get API key from environment"""
    return os.environ.get("BCBS_VALUES_API_KEY", "sample_test_key")

# Authentication dependency
async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Security dependency to verify API key.
    This adds authentication to protected endpoints.
    """
    # In production, use more secure methods like OAuth or JWT
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing. Add X-API-KEY header to your request.",
        )
    
    # Get expected API key
    expected_key = get_api_key()
    
    # Compare provided key with expected key
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key. Please use a valid key.",
        )
    
    return api_key

# Define reference points for GIS-enhanced valuation
REF_POINTS = {
    'downtown_richland': {
        'lat': 46.2804, 
        'lon': -119.2752, 
        'weight': 1.0  # Downtown Richland
    },
    'downtown_kennewick': {
        'lat': 46.2112, 
        'lon': -119.1367, 
        'weight': 0.9  # Downtown Kennewick
    },
    'downtown_pasco': {
        'lat': 46.2395, 
        'lon': -119.1005, 
        'weight': 0.8  # Downtown Pasco
    }
}

# Define neighborhood ratings for location quality adjustments
NEIGHBORHOOD_RATINGS = {
    'Richland': 1.15,       # Premium location
    'West Richland': 1.05,  # Above average
    'Kennewick': 1.0,       # Average
    'Pasco': 0.95,          # Slightly below average
    'Benton City': 0.9,     # Below average
    'Prosser': 0.85,        # Further below average
    
    # Common neighborhoods
    'Meadow Springs': 1.2,  # Premium Richland neighborhood
    'Horn Rapids': 1.1,     # Above average Richland neighborhood
    'Queensgate': 1.15,     # Premium West Richland neighborhood
    'Southridge': 1.05,     # Above average Kennewick neighborhood
    
    # Default for unknown locations
    'Unknown': 1.0
}

# Initialize FastAPI application
app = FastAPI(
    title="BCBS_Values API",
    description="Real estate valuation API for Benton County",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

class ETLStatus(BaseModel):
    """ETL process status response model."""
    status: str = Field(..., description="Current status of the ETL process")
    last_run: datetime.datetime = Field(..., description="Last ETL pipeline execution time")
    sources_processed: List[Dict[str, Union[str, int]]] = Field(..., description="Data sources processed")
    records_processed: int = Field(..., description="Total number of records processed")
    validation_status: str = Field(..., description="Overall validation status")
    validation_details: Dict = Field(..., description="Validation details by check type")
    errors: Optional[List[Dict]] = Field(None, description="Errors encountered during ETL")
    
class AgentStatus(BaseModel):
    """Agent status response model."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Current agent status")
    last_active: datetime.datetime = Field(..., description="Last time agent was active")
    current_task: Optional[str] = Field(None, description="Current task being executed by agent")
    queue_size: int = Field(..., description="Number of tasks in agent's queue")
    performance_metrics: Dict = Field(..., description="Agent performance metrics")

class AgentStatusList(BaseModel):
    """Agent status list response model."""
    agents: List[AgentStatus] = Field(..., description="List of agent statuses")
    system_status: str = Field(..., description="Overall system status")
    active_agents: int = Field(..., description="Number of active agents")
    tasks_in_progress: int = Field(..., description="Number of tasks currently in progress")
    tasks_completed_today: int = Field(..., description="Number of tasks completed today")

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

@app.get("/api/valuations", response_model=List[PropertyValue])
async def get_valuations(
    limit: int = Query(10, description="Maximum number of results to return"),
    min_value: Optional[float] = Query(None, description="Minimum property value filter"),
    max_value: Optional[float] = Query(None, description="Maximum property value filter"),
    property_type: Optional[str] = Query(None, description="Property type filter"),
    db: Database = Depends(get_db)
):
    """
    Get property valuations based on specified criteria.
    
    This endpoint connects to the database to:
    1. Query database for properties matching criteria
    2. Get the most recent property valuations for each property
    3. Filter results based on request parameters
    4. Return formatted valuation results
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
            
            # Build the response object
            valuations.append({
                "property_id": str(property.id),
                "address": f"{property.address}, {property.city}, {property.state} {property.zip_code}",
                "estimated_value": valuation.estimated_value,
                "confidence_score": valuation.confidence_score or 0.85,  # Default if None
                "model_used": valuation.model_name or "advanced_property_valuation",
                "valuation_date": valuation.valuation_date,
                "features_used": features_used,
                "comparable_properties": comparables
            })
        
        return valuations
    
    except Exception as e:
        logger.error(f"Error retrieving property valuations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        # Always close the session
        if 'session' in locals():
            session.close()

@app.get("/api/valuations/{property_id}", response_model=PropertyValue)
async def get_valuation_by_id(
    property_id: str = Path(..., description="Property ID to get valuation for"),
    db: Database = Depends(get_db)
):
    """
    Get valuation for a specific property by ID.
    
    This endpoint:
    1. Queries the database for the specific property
    2. Retrieves the latest valuation for that property
    3. Returns detailed valuation with confidence metrics
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
        
        # Build the response object
        result = {
            "property_id": str(property.id),
            "address": f"{property.address}, {property.city}, {property.state} {property.zip_code}",
            "estimated_value": valuation.estimated_value,
            "confidence_score": valuation.confidence_score or 0.85,  # Default if None
            "model_used": valuation.model_name or "advanced_property_valuation",
            "valuation_date": valuation.valuation_date,
            "features_used": features_used,
            "comparable_properties": comparables
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

@app.get("/api/etl-status", response_model=ETLStatus)
async def get_etl_status(db: Database = Depends(get_db)):
    """
    Get the current status of the ETL process.
    
    This endpoint:
    1. Queries the database for the most recent validation results
    2. Gets property counts by data source
    3. Returns formatted ETL status information
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
        
        # Build the ETL status response
        etl_status = {
            "status": status,
            "last_run": last_run,
            "sources_processed": sources_processed,
            "records_processed": total_records,
            "validation_status": validation_status,
            "validation_details": validation_details,
            "errors": errors
        }
        
        return etl_status
    
    except Exception as e:
        logger.error(f"Error retrieving ETL status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        # Always close the session
        if 'session' in locals():
            session.close()

@app.get("/api/agent-status", response_model=AgentStatusList)
async def get_agent_status():
    """
    Get the current status of the BCBS agent system.
    
    This endpoint will eventually:
    1. Query each agent for its current status
    2. Compile performance metrics across agents
    3. Return formatted agent status information
    """
    logger.info("Agent status request received")
    
    # Dummy data - in production this will query each agent's status
    agent_status = {
        "agents": [
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
        ],
        "system_status": "operational",
        "active_agents": 2,
        "tasks_in_progress": 2,
        "tasks_completed_today": 27
    }
    
    return agent_status


@app.post("/api/valuations", response_model=PropertyValue, dependencies=[Depends(verify_api_key)])
async def create_property_valuation(
    request: PropertyValuationRequest,
    db: Database = Depends(get_db)
):
    """
    Generate a property valuation based on provided property details.
    
    This endpoint:
    1. Takes property details as input
    2. Uses the valuation engine to estimate the property value
    3. Returns the valuation with confidence metrics
    
    Authentication: Requires API key header (X-API-KEY)
    """
    logger.info(f"Property valuation request received for {request.address}")
    
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
                
            # Build response object
            response = {
                "property_id": "temp-" + str(int(time.time())),
                "address": f"{request.address or 'Custom Property'}, {request.city or 'Richland'}, {request.state} {request.zip_code or '99352'}",
                "estimated_value": float(estimated_value),
                "confidence_score": float(confidence_score),
                "model_used": model_used,
                "valuation_date": datetime.datetime.now(),
                "features_used": features_used,
                "comparable_properties": comparable_properties
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
                    
                    # Create a new valuation record
                    valuation_obj = PropertyValuation(
                        property_id=property_obj.id,
                        valuation_date=datetime.datetime.now(),
                        estimated_value=estimated_value,
                        confidence_score=confidence_score,
                        model_name=model_used,
                        model_version='1.0',
                        model_r2_score=confidence_score,
                        feature_importance=json.dumps(feature_importance),
                        comparable_properties=json.dumps(comparable_properties) if comparable_properties else None
                    )
                    
                    # Add location factors if available
                    if 'location_factor' in valuation_result:
                        valuation_obj.location_factor = valuation_result['location_factor']
                    if 'size_factor' in valuation_result:
                        valuation_obj.size_factor = valuation_result['size_factor']
                        
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
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)