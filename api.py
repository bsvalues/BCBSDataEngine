"""
Enhanced FastAPI server for the BCBS Valuation Platform.

This module provides a FastAPI application with endpoints for property valuations,
ETL pipeline status monitoring, and agent monitoring with comprehensive security features.
"""

import logging
import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from enum import Enum

# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, Header, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Try importing optional packages
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT is not installed. JWT authentication will be limited.")
    
    # Create a mock JWT module to avoid breaking code
    class MockJWT:
        class ExpiredSignatureError(Exception): pass
        class InvalidTokenError(Exception): pass
        def encode(self, *args, **kwargs): return "mock.token.invalid"
        def decode(self, *args, **kwargs): 
            raise self.InvalidTokenError("JWT module not installed")
    jwt = MockJWT()

# Import application modules
try:
    from app import app as flask_app, db
    from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    logging.error(f"Database module import error: {str(e)}")
    
    # Create temporary mock classes for when DB is not available
    class MockModel:
        id = None
        query = None
        
        @classmethod
        def get(cls, id):
            return None
            
    class Property(MockModel): pass
    class PropertyValuation(MockModel): pass
    class PropertyFeature(MockModel): pass
    class ETLJob(MockModel): pass
    class Agent(MockModel): pass
    class AgentLog(MockModel): pass

# Import valuation components with fallbacks
try:
    from src.valuation import perform_valuation
    VALUATION_AVAILABLE = True
except ImportError as e:
    VALUATION_AVAILABLE = False
    logging.error(f"Valuation module import error: {str(e)}")
    
    # Create minimal mocks for essential functionality
    def perform_valuation(property_obj, valuation_method='enhanced_regression'):
        """Mock implementation when the actual valuation module is not available"""
        return {
            "estimated_value": 250000, 
            "confidence_score": 0.75,
            "valuation_method": valuation_method,
            "performance_metrics": {
                "error": "Valuation module not available"
            }
        }

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BCBS Valuation API",
    description="Advanced API for property valuations, ETL monitoring, and agent status",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you'd restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Secret key for JWT tokens
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
TOKEN_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Initialize valuation module
# Note: perform_valuation is now imported directly and doesn't need instantiation

# ==========================================
# Pydantic Models for Request/Response
# ==========================================

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    scope: str

class PropertyBase(BaseModel):
    property_id: str
    address: str
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[int] = None
    lot_size: Optional[float] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    neighborhood: Optional[str] = None

class PropertyValuationBase(BaseModel):
    property_id: str
    address: str
    estimated_value: float
    valuation_date: str
    confidence_score: float
    valuation_method: str
    model_metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True

class ValuationRequest(BaseModel):
    property_id: Optional[str] = None
    address: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    include_metrics: bool = False
    
    @validator('property_id', 'address')
    def validate_identifiers(cls, v, values, **kwargs):
        # Ensure either property_id or address is provided
        if 'property_id' not in values and 'address' not in values:
            raise ValueError("Either property_id or address must be provided")
        return v

class ValuationResponse(BaseModel):
    valuations: List[PropertyValuationBase]
    page: int
    limit: int
    total: int
    pages: int
    metadata: Dict[str, Any]
    
    class Config:
        orm_mode = True

class EtlJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ETLJobBase(BaseModel):
    id: int
    job_type: str
    status: EtlJobStatus
    start_time: str
    end_time: Optional[str] = None
    progress: float
    records_processed: int
    records_total: int
    source: str
    message: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[int] = None
    
    class Config:
        orm_mode = True

class ETLStatusResponse(BaseModel):
    jobs: List[ETLJobBase]
    stats: Dict[str, Any]
    health: Dict[str, Any]
    timeframe: Optional[str] = None
    timestamp: str
    
    class Config:
        orm_mode = True

class AgentStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"

class AgentBase(BaseModel):
    id: int
    name: str
    agent_type: str
    status: AgentStatus
    is_active: bool
    version: str
    success_rate: float
    last_active: Optional[str] = None
    queue_size: Optional[int] = None
    avg_processing_time: Optional[float] = None
    error_count: Optional[int] = None
    
    class Config:
        orm_mode = True

class AgentLogBase(BaseModel):
    id: int
    agent_id: int
    log_level: str
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True

class AgentStatusResponse(BaseModel):
    agents: List[AgentBase]
    count: int
    active_count: int
    timestamp: str
    metrics: Dict[str, Any]
    health: Dict[str, Any]
    
    class Config:
        orm_mode = True

class AgentDetailResponse(BaseModel):
    agent: AgentBase
    logs: List[AgentLogBase]
    metrics: Dict[str, Any]
    
    class Config:
        orm_mode = True

# ==========================================
# Authentication and Authorization
# ==========================================

async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Verify API key authentication.
    
    This function validates the provided API key against the configured key.
    In a production system, this would check against a database of valid keys.
    
    Args:
        api_key: The API key from the X-API-Key header
        
    Returns:
        Dict with user information if valid, None otherwise
    """
    if not api_key:
        return None
    
    # Get the valid API key from environment variable
    valid_api_key = os.environ.get('BCBS_VALUES_API_KEY', 'sample_test_key')
    
    if api_key != valid_api_key:
        return None
        
    return {"client_id": "api_key_user", "client_type": "api_client"}

async def get_token(auth: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    """
    Verify JWT token authentication.
    
    This function validates the provided JWT token, checking signature and expiration.
    
    Args:
        auth: The authorization credentials containing the bearer token
        
    Returns:
        Dict with token claims if valid, None otherwise
    """
    if not auth or not JWT_AVAILABLE:
        return None
    
    try:
        payload = jwt.decode(auth.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def get_current_user(
    api_key_user: Optional[Dict] = Depends(get_api_key),
    token_user: Optional[Dict] = Depends(get_token)
):
    """
    Get the current authenticated user from either API key or JWT token.
    
    This function combines both authentication methods, allowing clients to use
    either an API key or a JWT token for authentication.
    
    Args:
        api_key_user: User info from API key auth
        token_user: User info from JWT token auth
        
    Returns:
        Dict with user information
        
    Raises:
        HTTPException: If neither authentication method provides a valid user
    """
    if api_key_user:
        return api_key_user
    if token_user:
        return token_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )

# ==========================================
# API Endpoints
# ==========================================

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint returning API health status
    """
    return {
        "status": "operational",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/token", response_model=TokenResponse)
async def generate_token(username: str = Query(...), password: str = Query(...)):
    """
    Generate a JWT token for API authentication.
    
    This endpoint allows clients to obtain a JWT token for authenticating future requests.
    In a production system, this would validate user credentials against a database.
    
    Args:
        username: The username for authentication
        password: The password for authentication
        
    Returns:
        TokenResponse with JWT token information
    """
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication is not available"
        )
        
    # In a real system, you would validate username/password against a database
    # For demo purposes, accept any username with "demo" password
    if password != "demo":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create token payload
    expires = datetime.utcnow() + timedelta(seconds=TOKEN_EXPIRY)
    payload = {
        "sub": username,
        "exp": expires,
        "client_id": username,
        "client_type": "user"
    }
    
    # Generate token
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_EXPIRY,
        "scope": "api"
    }

@app.get("/api/valuations", response_model=ValuationResponse)
async def get_valuations(
    property_id: Optional[str] = Query(None, description="Filter by property ID"),
    address: Optional[str] = Query(None, description="Filter by property address"),
    neighborhood: Optional[str] = Query(None, description="Filter by neighborhood"),
    property_type: Optional[str] = Query(None, description="Filter by property type"),
    min_value: Optional[float] = Query(None, description="Minimum estimated value"),
    max_value: Optional[float] = Query(None, description="Maximum estimated value"),
    valuation_date_start: Optional[str] = Query(None, description="Start date for valuation date range"),
    valuation_date_end: Optional[str] = Query(None, description="End date for valuation date range"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence score"),
    method: Optional[str] = Query(None, description="Valuation method"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    user: Dict = Depends(get_current_user)
):
    """
    Get property valuations with advanced filtering and pagination.
    
    This endpoint retrieves property valuations from the database with comprehensive
    filtering options and pagination. It supports filtering by property attributes,
    valuation parameters, date ranges, and more.
    
    The results include the valuations along with metadata about pagination and
    the query executed.
    
    Security:
        Requires API key or JWT token authentication
    
    Args:
        property_id: Filter by property ID
        address: Filter by property address (partial match)
        neighborhood: Filter by neighborhood
        property_type: Filter by property type
        min_value: Minimum estimated value
        max_value: Maximum estimated value
        valuation_date_start: Start date for valuation date range (ISO format)
        valuation_date_end: End date for valuation date range (ISO format)
        min_confidence: Minimum confidence score
        method: Valuation method used
        page: Page number for pagination
        limit: Items per page for pagination
        
    Returns:
        ValuationResponse with paginated valuations and metadata
    """
    if not DB_AVAILABLE:
        # Return example response if database is not available
        logger.warning("Database not available, returning example response")
        return {
            "valuations": [
                {
                    "property_id": "PROP123",
                    "address": "123 Main St, Anytown, WA",
                    "estimated_value": 350000.0,
                    "valuation_date": datetime.utcnow().isoformat(),
                    "confidence_score": 0.92,
                    "valuation_method": "advanced_regression",
                    "model_metrics": {
                        "r_squared": 0.87,
                        "mae": 15200.0,
                        "mse": 22.3
                    }
                }
            ],
            "page": page,
            "limit": limit,
            "total": 1,
            "pages": 1,
            "metadata": {
                "filters_applied": {
                    "property_id": property_id,
                    "neighborhood": neighborhood
                },
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    try:
        # Start with base query
        query = db.session.query(PropertyValuation)
        
        # Apply filters
        if property_id:
            query = query.filter(PropertyValuation.property_id == property_id)
            
        if address:
            query = query.join(Property).filter(Property.address.ilike(f"%{address}%"))
            
        if neighborhood:
            query = query.join(Property).filter(Property.neighborhood == neighborhood)
            
        if property_type:
            query = query.join(Property).filter(Property.property_type == property_type)
            
        if min_value:
            query = query.filter(PropertyValuation.estimated_value >= min_value)
            
        if max_value:
            query = query.filter(PropertyValuation.estimated_value <= max_value)
            
        if valuation_date_start:
            start_date = datetime.fromisoformat(valuation_date_start.replace('Z', '+00:00'))
            query = query.filter(PropertyValuation.valuation_date >= start_date)
            
        if valuation_date_end:
            end_date = datetime.fromisoformat(valuation_date_end.replace('Z', '+00:00'))
            query = query.filter(PropertyValuation.valuation_date <= end_date)
            
        if min_confidence:
            query = query.filter(PropertyValuation.confidence_score >= min_confidence)
            
        if method:
            query = query.filter(PropertyValuation.valuation_method == method)
        
        # Get total count for pagination
        total = query.count()
        pages = (total + limit - 1) // limit
        
        # Apply pagination
        valuations = query.order_by(PropertyValuation.valuation_date.desc()) \
                       .offset((page - 1) * limit) \
                       .limit(limit) \
                       .all()
        
        # Format response
        result = {
            "valuations": [
                {
                    "property_id": v.property_id,
                    "address": v.property.address if hasattr(v, 'property') else "Unknown",
                    "estimated_value": v.estimated_value,
                    "valuation_date": v.valuation_date.isoformat(),
                    "confidence_score": v.confidence_score,
                    "valuation_method": v.valuation_method,
                    "model_metrics": v.model_metrics if hasattr(v, 'model_metrics') else None
                }
                for v in valuations
            ],
            "page": page,
            "limit": limit,
            "total": total,
            "pages": pages,
            "metadata": {
                "filters_applied": {
                    "property_id": property_id,
                    "address": address,
                    "neighborhood": neighborhood,
                    "property_type": property_type,
                    "value_range": f"{min_value}-{max_value}" if min_value or max_value else None,
                    "date_range": f"{valuation_date_start}-{valuation_date_end}" if valuation_date_start or valuation_date_end else None,
                    "min_confidence": min_confidence,
                    "method": method
                },
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving valuations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve valuations: {str(e)}"
        )

@app.post("/api/valuations", response_model=PropertyValuationBase)
async def create_valuation(
    request: ValuationRequest,
    user: Dict = Depends(get_current_user)
):
    """
    Create a new property valuation using the enhanced valuation engine.
    
    This endpoint generates a new property valuation using the integrated
    valuation engine from src/valuation.py. It can use either a property ID
    to retrieve property data from the database, or accept property features
    directly in the request.
    
    Security:
        Requires API key or JWT token authentication
    
    Args:
        request: ValuationRequest with property information
        
    Returns:
        PropertyValuationBase with the valuation result
    """
    try:
        property_data = None
        
        # If property_id is provided, retrieve property data from database
        if request.property_id and DB_AVAILABLE:
            property_data = db.session.query(Property).filter(Property.property_id == request.property_id).first()
            
            if not property_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Property with ID {request.property_id} not found"
                )
        
        # If no property data from DB, use request features
        if not property_data:
            # In a real implementation, this would create a Property object from the features
            property_data = {
                "property_id": request.property_id or "UNKNOWN",
                "address": request.address or "Unknown Address",
                "features": request.features or {}
            }
        
        # Use valuation engine to generate valuation
        if VALUATION_AVAILABLE:
            # Create a Property-like object for the valuation function
            # This adapts the property data to be compatible with perform_valuation
            class PropertyAdapter:
                def __init__(self, data):
                    if isinstance(data, dict):
                        for key, value in data.items():
                            setattr(self, key, value)
                        # Extract features if present
                        if 'features' in data:
                            for feat_key, feat_value in data['features'].items():
                                setattr(self, feat_key, feat_value)
                    else:
                        # For database objects, copy attributes
                        for key in ['property_id', 'address', 'city', 'state', 'zip_code', 'latitude', 
                                   'longitude', 'bedrooms', 'bathrooms', 'square_feet', 'lot_size', 
                                   'year_built', 'property_type', 'neighborhood']:
                            if hasattr(data, key):
                                setattr(self, key, getattr(data, key))
            
            # Create adapter for the valuation function
            property_obj = PropertyAdapter(property_data)
            
            # Use the direct perform_valuation function from src.valuation
            valuation_method = request.features.get('valuation_method', 'enhanced_regression') if request.features else 'enhanced_regression'
            valuation_result = perform_valuation(property_obj, valuation_method=valuation_method)
            
            # Format metrics to be compatible with the expected output
            if request.include_metrics and 'performance_metrics' in valuation_result:
                metrics = valuation_result.get('performance_metrics', {})
            else:
                metrics = {}
                
            # Create compatible result format
            valuation_result = {
                "value": valuation_result.get("estimated_value", 0.0),
                "confidence": valuation_result.get("confidence_score", 0.0),
                "method": valuation_result.get("valuation_method", "enhanced_regression"),
                "metrics": metrics
            }
        else:
            # Mock valuation if engine not available
            valuation_result = {
                "value": 350000.0,
                "confidence": 0.85,
                "method": "mock_valuation",
                "metrics": {
                    "r_squared": 0.82,
                    "mae": 18500.0
                }
            }
        
        # Format the response
        result = {
            "property_id": getattr(property_data, "property_id", property_data.get("property_id", "UNKNOWN")),
            "address": getattr(property_data, "address", property_data.get("address", "Unknown")),
            "estimated_value": valuation_result.get("value", 0.0),
            "valuation_date": datetime.utcnow().isoformat(),
            "confidence_score": valuation_result.get("confidence", 0.0),
            "valuation_method": valuation_result.get("method", "unknown"),
            "model_metrics": valuation_result.get("metrics", {}) if request.include_metrics else None
        }
        
        # If database is available, store the valuation
        if DB_AVAILABLE:
            new_valuation = PropertyValuation(
                property_id=result["property_id"],
                estimated_value=result["estimated_value"],
                valuation_date=datetime.fromisoformat(result["valuation_date"].replace('Z', '+00:00')),
                confidence_score=result["confidence_score"],
                valuation_method=result["valuation_method"],
                model_metrics=result["model_metrics"]
            )
            
            db.session.add(new_valuation)
            db.session.commit()
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating valuation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create valuation: {str(e)}"
        )

@app.get("/api/etl-status", response_model=ETLStatusResponse)
async def get_etl_status(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[EtlJobStatus] = Query(None, description="Filter by job status"),
    timeframe: Optional[str] = Query(None, description="Timeframe for jobs (today, yesterday, this_week, last_week)"),
    source: Optional[str] = Query(None, description="Filter by data source"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of jobs to return"),
    user: Dict = Depends(get_current_user)
):
    """
    Get ETL pipeline status with comprehensive metrics and filtering.
    
    This endpoint provides detailed information about ETL (Extract, Transform, Load) jobs,
    including their current status, progress, and performance metrics. It supports filtering
    by job type, status, timeframe, and data source.
    
    The response includes individual job details as well as aggregated statistics and
    health information about the overall ETL pipeline.
    
    Security:
        Requires API key or JWT token authentication
    
    Args:
        job_type: Filter by job type (extract, transform, load, etc.)
        status: Filter by job status (pending, running, completed, failed, cancelled)
        timeframe: Timeframe for jobs (today, yesterday, this_week, last_week)
        source: Filter by data source (e.g., county_data, mls, etc.)
        limit: Maximum number of jobs to return
        
    Returns:
        ETLStatusResponse with job details, statistics, and health information
    """
    if not DB_AVAILABLE:
        # Return example response if database is not available
        logger.warning("Database not available, returning example response")
        now = datetime.utcnow()
        
        return {
            "jobs": [
                {
                    "id": 1,
                    "job_type": "extract",
                    "status": EtlJobStatus.COMPLETED,
                    "start_time": (now - timedelta(hours=2)).isoformat(),
                    "end_time": (now - timedelta(hours=1, minutes=45)).isoformat(),
                    "progress": 1.0,
                    "records_processed": 500,
                    "records_total": 500,
                    "source": "county_data",
                    "message": "Successfully extracted 500 properties",
                    "error": None,
                    "duration_seconds": 900
                },
                {
                    "id": 2,
                    "job_type": "transform",
                    "status": EtlJobStatus.COMPLETED,
                    "start_time": (now - timedelta(hours=1, minutes=40)).isoformat(),
                    "end_time": (now - timedelta(hours=1, minutes=20)).isoformat(),
                    "progress": 1.0,
                    "records_processed": 500,
                    "records_total": 500,
                    "source": "county_data",
                    "message": "Successfully transformed 500 properties",
                    "error": None,
                    "duration_seconds": 1200
                },
                {
                    "id": 3,
                    "job_type": "load",
                    "status": EtlJobStatus.RUNNING,
                    "start_time": (now - timedelta(minutes=15)).isoformat(),
                    "end_time": None,
                    "progress": 0.6,
                    "records_processed": 300,
                    "records_total": 500,
                    "source": "county_data",
                    "message": "Loading property data into database",
                    "error": None,
                    "duration_seconds": None
                }
            ],
            "stats": {
                "total_jobs": 3,
                "completed_jobs": 2,
                "failed_jobs": 0,
                "running_jobs": 1,
                "pending_jobs": 0,
                "total_records_processed": 1300,
                "average_progress": 0.87
            },
            "health": {
                "status": "healthy",
                "pipeline_active": True,
                "last_successful_job": (now - timedelta(hours=1, minutes=20)).isoformat()
            },
            "timeframe": timeframe,
            "timestamp": now.isoformat()
        }
    
    try:
        # Start with base query
        query = db.session.query(ETLJob)
        
        # Apply filters
        if job_type:
            query = query.filter(ETLJob.job_type == job_type)
            
        if status:
            query = query.filter(ETLJob.status == status)
            
        if source:
            query = query.filter(ETLJob.source == source)
            
        # Apply timeframe filter
        if timeframe:
            now = datetime.utcnow()
            
            if timeframe == "today":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(ETLJob.start_time >= start_date)
                
            elif timeframe == "yesterday":
                start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(ETLJob.start_time >= start_date, ETLJob.start_time < end_date)
                
            elif timeframe == "this_week":
                # Start of week (Monday)
                start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(ETLJob.start_time >= start_date)
                
            elif timeframe == "last_week":
                # Start of last week (Monday)
                start_date = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
                # End of last week (Sunday)
                end_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(ETLJob.start_time >= start_date, ETLJob.start_time < end_date)
        
        # Get all matching jobs for statistics
        all_jobs = query.all()
        
        # Calculate statistics
        completed_jobs = sum(1 for job in all_jobs if job.status == EtlJobStatus.COMPLETED)
        failed_jobs = sum(1 for job in all_jobs if job.status == EtlJobStatus.FAILED)
        running_jobs = sum(1 for job in all_jobs if job.status == EtlJobStatus.RUNNING)
        pending_jobs = sum(1 for job in all_jobs if job.status == EtlJobStatus.PENDING)
        
        total_records_processed = sum(job.records_processed for job in all_jobs if job.records_processed)
        
        # Calculate average progress (only for jobs with progress)
        progress_jobs = [job for job in all_jobs if hasattr(job, 'progress') and job.progress is not None]
        average_progress = sum(job.progress for job in progress_jobs) / len(progress_jobs) if progress_jobs else 0
        
        # Find last successful job
        last_successful_job = next(
            (job.end_time for job in sorted(all_jobs, key=lambda j: j.end_time or datetime.min, reverse=True)
             if job.status == EtlJobStatus.COMPLETED and job.end_time),
            None
        )
        
        # Apply limit for returned jobs
        jobs = query.order_by(ETLJob.start_time.desc()).limit(limit).all()
        
        # Format jobs for response
        formatted_jobs = []
        for job in jobs:
            duration = None
            if job.start_time and job.end_time:
                duration = int((job.end_time - job.start_time).total_seconds())
                
            formatted_jobs.append({
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "start_time": job.start_time.isoformat(),
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "progress": getattr(job, 'progress', 1.0 if job.status == EtlJobStatus.COMPLETED else 0.0),
                "records_processed": job.records_processed or 0,
                "records_total": getattr(job, 'records_total', job.records_processed or 0),
                "source": job.source,
                "message": getattr(job, 'message', None),
                "error": getattr(job, 'error', None),
                "duration_seconds": duration
            })
        
        # Determine overall health status
        health_status = "healthy"
        if failed_jobs > 0:
            health_status = "warning"
            
        if failed_jobs > running_jobs + completed_jobs:
            health_status = "error"
            
        # Format final response
        result = {
            "jobs": formatted_jobs,
            "stats": {
                "total_jobs": len(all_jobs),
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "running_jobs": running_jobs,
                "pending_jobs": pending_jobs,
                "total_records_processed": total_records_processed,
                "average_progress": average_progress
            },
            "health": {
                "status": health_status,
                "pipeline_active": running_jobs > 0,
                "last_successful_job": last_successful_job.isoformat() if last_successful_job else None
            },
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving ETL status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ETL status: {str(e)}"
        )

@app.get("/api/agent-status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    active_only: bool = Query(False, description="Return only active agents"),
    user: Dict = Depends(get_current_user)
):
    """
    Get detailed status information for BS Army agents.
    
    This endpoint provides comprehensive information about the status and performance
    of all agents in the BS Army. It includes individual agent details as well as
    aggregated statistics and health information for the entire agent fleet.
    
    Security:
        Requires API key or JWT token authentication
    
    Args:
        agent_type: Filter by agent type (regression, ensemble, gis, etc.)
        status: Filter by agent status (idle, processing, error, disabled, maintenance)
        active_only: If true, return only active agents
        
    Returns:
        AgentStatusResponse with agent details, statistics, and health information
    """
    if not DB_AVAILABLE:
        # Return example response if database is not available
        logger.warning("Database not available, returning example response")
        now = datetime.utcnow()
        
        example_agents = [
            {
                "id": 1,
                "name": "Advanced Regression Agent",
                "agent_type": "regression",
                "status": AgentStatus.PROCESSING,
                "is_active": True,
                "version": "2.1.0",
                "success_rate": 0.95,
                "last_active": now.isoformat(),
                "queue_size": 5,
                "avg_processing_time": 1.2,
                "error_count": 2
            },
            {
                "id": 2,
                "name": "GIS Feature Agent",
                "agent_type": "gis",
                "status": AgentStatus.IDLE,
                "is_active": True,
                "version": "1.8.5",
                "success_rate": 0.98,
                "last_active": (now - timedelta(minutes=15)).isoformat(),
                "queue_size": 0,
                "avg_processing_time": 3.5,
                "error_count": 0
            },
            {
                "id": 3,
                "name": "Neural Ensemble Agent",
                "agent_type": "ensemble",
                "status": AgentStatus.ERROR,
                "is_active": False,
                "version": "3.0.1",
                "success_rate": 0.75,
                "last_active": (now - timedelta(hours=2)).isoformat(),
                "queue_size": 20,
                "avg_processing_time": 5.0,
                "error_count": 8
            }
        ]
        
        # Apply filters to example data
        filtered_agents = example_agents
        if agent_type:
            filtered_agents = [a for a in filtered_agents if a["agent_type"] == agent_type]
            
        if status:
            filtered_agents = [a for a in filtered_agents if a["status"] == status]
            
        if active_only:
            filtered_agents = [a for a in filtered_agents if a["is_active"]]
        
        return {
            "agents": filtered_agents,
            "count": len(filtered_agents),
            "active_count": sum(1 for a in filtered_agents if a["is_active"]),
            "timestamp": now.isoformat(),
            "metrics": {
                "avg_success_rate": sum(a["success_rate"] for a in filtered_agents) / len(filtered_agents) if filtered_agents else 0,
                "total_queue_size": sum(a["queue_size"] for a in filtered_agents),
                "avg_processing_time": sum(a["avg_processing_time"] for a in filtered_agents) / len(filtered_agents) if filtered_agents else 0,
                "total_error_count": sum(a["error_count"] for a in filtered_agents)
            },
            "health": {
                "status": "warning" if any(a["status"] == AgentStatus.ERROR for a in filtered_agents) else "healthy",
                "agents_active": any(a["is_active"] for a in filtered_agents),
                "error_ratio": sum(a["error_count"] for a in filtered_agents) / max(1, sum(a["queue_size"] for a in filtered_agents) + sum(a["error_count"] for a in filtered_agents))
            }
        }
    
    try:
        # Start with base query
        query = db.session.query(Agent)
        
        # Apply filters
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type)
            
        if status:
            query = query.filter(Agent.status == status)
            
        if active_only:
            query = query.filter(Agent.is_active == True)
        
        # Get all agents
        agents = query.all()
        
        # Format agents for response
        formatted_agents = []
        total_success_rate = 0
        total_queue_size = 0
        total_processing_time = 0
        total_error_count = 0
        
        for agent in agents:
            # Get agent metrics
            agent_metrics = {
                "queue_size": getattr(agent, 'queue_size', 0),
                "avg_processing_time": getattr(agent, 'avg_processing_time', 0),
                "error_count": getattr(agent, 'error_count', 0)
            }
            
            # Accumulate totals for overall metrics
            total_success_rate += getattr(agent, 'success_rate', 0)
            total_queue_size += agent_metrics["queue_size"]
            total_processing_time += agent_metrics["avg_processing_time"]
            total_error_count += agent_metrics["error_count"]
            
            formatted_agents.append({
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "is_active": agent.is_active,
                "version": getattr(agent, 'version', '1.0.0'),
                "success_rate": getattr(agent, 'success_rate', 0),
                "last_active": agent.last_active.isoformat() if hasattr(agent, 'last_active') and agent.last_active else None,
                "queue_size": agent_metrics["queue_size"],
                "avg_processing_time": agent_metrics["avg_processing_time"],
                "error_count": agent_metrics["error_count"]
            })
        
        # Calculate average metrics
        avg_success_rate = total_success_rate / len(agents) if agents else 0
        avg_processing_time = total_processing_time / len(agents) if agents else 0
        
        # Calculate error ratio (errors / (queue + errors))
        total_tasks = total_queue_size + total_error_count
        error_ratio = total_error_count / total_tasks if total_tasks > 0 else 0
        
        # Determine overall health status
        health_status = "healthy"
        if error_ratio > 0.1:  # More than 10% error rate
            health_status = "warning"
            
        if error_ratio > 0.3:  # More than 30% error rate
            health_status = "error"
            
        if not any(agent.is_active for agent in agents):
            health_status = "critical"  # No active agents
        
        # Format final response
        result = {
            "agents": formatted_agents,
            "count": len(agents),
            "active_count": sum(1 for agent in agents if agent.is_active),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "avg_success_rate": avg_success_rate,
                "total_queue_size": total_queue_size,
                "avg_processing_time": avg_processing_time,
                "total_error_count": total_error_count
            },
            "health": {
                "status": health_status,
                "agents_active": any(agent.is_active for agent in agents),
                "error_ratio": error_ratio
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving agent status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent status: {str(e)}"
        )

@app.get("/api/agent-status/{agent_id}", response_model=AgentDetailResponse)
async def get_agent_detail(
    agent_id: int,
    log_level: Optional[str] = Query(None, description="Filter logs by level"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of logs to return"),
    user: Dict = Depends(get_current_user)
):
    """
    Get detailed information and logs for a specific agent.
    
    This endpoint provides comprehensive information about a single agent,
    including its status, performance metrics, and recent logs. It supports
    filtering logs by log level and limiting the number of logs returned.
    
    Security:
        Requires API key or JWT token authentication
    
    Args:
        agent_id: The ID of the agent to retrieve
        log_level: Filter logs by level (info, warning, error, debug)
        limit: Maximum number of logs to return
        
    Returns:
        AgentDetailResponse with agent details, logs, and metrics
    """
    if not DB_AVAILABLE:
        # Return example response if database is not available
        logger.warning("Database not available, returning example response")
        now = datetime.utcnow()
        
        # Example agent types and their corresponding metrics
        agent_types = {
            1: {"type": "regression", "name": "Advanced Regression Agent"},
            2: {"type": "gis", "name": "GIS Feature Agent"},
            3: {"type": "ensemble", "name": "Neural Ensemble Agent"}
        }
        
        # Get agent type info or use default
        agent_info = agent_types.get(agent_id, {"type": "unknown", "name": f"Agent {agent_id}"})
        
        # Generate example logs
        example_logs = []
        log_levels = ["info", "debug", "warning", "error"]
        
        for i in range(min(limit, 20)):
            log_level_idx = min(i % 4, 3)  # Cycle through log levels, but ensure more infos than errors
            timestamp = now - timedelta(minutes=i * 5)
            
            example_logs.append({
                "id": i + 1,
                "agent_id": agent_id,
                "log_level": log_levels[log_level_idx],
                "message": f"Example {log_levels[log_level_idx]} message for {agent_info['name']}",
                "timestamp": timestamp.isoformat(),
                "details": {
                    "request_id": f"req-{i:03d}",
                    "duration_ms": 120 + (i * 10),
                    "memory_usage_mb": 85 + (i % 10)
                }
            })
        
        # Filter logs by level if specified
        if log_level:
            example_logs = [log for log in example_logs if log["log_level"] == log_level]
        
        return {
            "agent": {
                "id": agent_id,
                "name": agent_info["name"],
                "agent_type": agent_info["type"],
                "status": AgentStatus.PROCESSING,
                "is_active": True,
                "version": "2.1.0",
                "success_rate": 0.92,
                "last_active": now.isoformat(),
                "queue_size": 3,
                "avg_processing_time": 2.5,
                "error_count": 1
            },
            "logs": example_logs,
            "metrics": {
                "processed_last_hour": 45,
                "avg_response_time_ms": 180,
                "memory_usage_mb": 120,
                "uptime_hours": 28,
                "error_rate_1h": 0.02,
                "cpu_usage_percent": 35
            }
        }
    
    try:
        # Get agent by ID
        agent = db.session.query(Agent).get(agent_id)
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Query agent logs
        logs_query = db.session.query(AgentLog).filter(AgentLog.agent_id == agent_id)
        
        # Apply log level filter if specified
        if log_level:
            logs_query = logs_query.filter(AgentLog.log_level == log_level)
        
        # Get logs with limit
        logs = logs_query.order_by(AgentLog.timestamp.desc()).limit(limit).all()
        
        # Format agent for response
        formatted_agent = {
            "id": agent.id,
            "name": agent.name,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "is_active": agent.is_active,
            "version": getattr(agent, 'version', '1.0.0'),
            "success_rate": getattr(agent, 'success_rate', 0),
            "last_active": agent.last_active.isoformat() if hasattr(agent, 'last_active') and agent.last_active else None,
            "queue_size": getattr(agent, 'queue_size', 0),
            "avg_processing_time": getattr(agent, 'avg_processing_time', 0),
            "error_count": getattr(agent, 'error_count', 0)
        }
        
        # Format logs for response
        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                "id": log.id,
                "agent_id": log.agent_id,
                "log_level": log.log_level,
                "message": log.message,
                "timestamp": log.timestamp.isoformat(),
                "details": log.details if hasattr(log, 'details') else None
            })
        
        # Calculate agent metrics
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        
        # Get logs from the last hour
        recent_logs = db.session.query(AgentLog).filter(
            AgentLog.agent_id == agent_id,
            AgentLog.timestamp >= one_hour_ago
        ).all()
        
        # Count processed items in the last hour
        processed_last_hour = len(recent_logs)
        
        # Count errors in the last hour
        error_logs = [log for log in recent_logs if log.log_level == 'error']
        error_rate_1h = len(error_logs) / processed_last_hour if processed_last_hour > 0 else 0
        
        # Calculate average response time from log details if available
        response_times = []
        for log in recent_logs:
            if hasattr(log, 'details') and log.details:
                details = log.details
                if isinstance(details, dict) and 'duration_ms' in details:
                    response_times.append(details['duration_ms'])
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Format metrics
        metrics = {
            "processed_last_hour": processed_last_hour,
            "avg_response_time_ms": avg_response_time,
            "error_rate_1h": error_rate_1h,
            # Additional metrics that might be available directly from the agent
            "memory_usage_mb": getattr(agent, 'memory_usage', 0),
            "uptime_hours": getattr(agent, 'uptime_hours', 0),
            "cpu_usage_percent": getattr(agent, 'cpu_usage', 0)
        }
        
        # Format final response
        result = {
            "agent": formatted_agent,
            "logs": formatted_logs,
            "metrics": metrics
        }
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent details: {str(e)}"
        )

# This conditional is for running the app directly with Python
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the API server")
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=5000, 
        reload=True
    )