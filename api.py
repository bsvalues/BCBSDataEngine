"""
FastAPI implementation for the BCBS_Values real estate valuation API.
This module provides HTTP endpoints for property valuation, ETL status, and agent status.
"""
import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Path, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import database and models
from db.database import Database
from db.models import PropertyValuation, Property, ValidationResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Run the application when called directly (development mode)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)