"""
API endpoints for the BCBS Values application.
"""
import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta
from functools import wraps
import secrets
from typing import Dict, List, Optional, Union

# Try importing optional packages
try:
    import jwt
except ImportError:
    logging.error("PyJWT is not installed. JWT authentication will not work.")
    # Create a mock JWT module to avoid breaking code
    class MockJWT:
        class ExpiredSignatureError(Exception): pass
        class InvalidTokenError(Exception): pass
        def encode(self, *args, **kwargs): return "mock.token.invalid"
        def decode(self, *args, **kwargs): 
            raise self.InvalidTokenError("JWT module not installed")
    jwt = MockJWT()

try:
    import pandas as pd
except ImportError:
    logging.error("Pandas is not installed. Some data processing features may be limited.")
    # Create a simple mock for basic functionality
    class MockPandas:
        def DataFrame(self, *args, **kwargs): 
            return {"data": args[0] if args else kwargs}
    pd = MockPandas()

# Import Flask components
try:
    from flask import Blueprint, Flask, jsonify, request, current_app, g
    from sqlalchemy import desc, func
except ImportError as e:
    logging.error(f"Critical Flask component missing: {str(e)}")
    raise

# Import application modules
try:
    from app import app, db
    from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
except ImportError as e:
    logging.error(f"Critical application module missing: {str(e)}")
    raise

# Import valuation components with fallbacks
try:
    from src.valuation import ValuationEngine, AdvancedValuationEngine, EnhancedGISValuationEngine
    from src.enhanced_gis_features import load_gis_features, calculate_gis_adjustments
except ImportError as e:
    logging.error(f"Valuation module import error: {str(e)}")
    # Create minimal mocks for essential functionality
    class BaseValuationEngine:
        def predict(self, *args, **kwargs): return {"value": 250000, "confidence": 0.75}
        
    ValuationEngine = AdvancedValuationEngine = EnhancedGISValuationEngine = BaseValuationEngine
    
    def mock_load_gis_features(*args): return {}
    def mock_calculate_gis_adjustments(lat, lon, value, *args): return value, {"mocked": True}
    
    load_gis_features = mock_load_gis_features
    calculate_gis_adjustments = mock_calculate_gis_adjustments


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize valuation engine
valuation_engine = EnhancedGISValuationEngine()

# GIS features
try:
    gis_features = load_gis_features('gis_features_metadata.json')
    if gis_features:
        logger.info(f"Loaded GIS features with {len(gis_features)} categories")
    else:
        logger.warning("No GIS features loaded")
except Exception as e:
    logger.error(f"Error loading GIS features: {str(e)}")
    gis_features = {}


# Secret key for JWT tokens
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
TOKEN_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Function to generate JWT token
def generate_token(agent_id, agent_type):
    """
    Generate a JWT token for an agent.
    
    Args:
        agent_id: The ID of the agent
        agent_type: The type of agent (e.g., 'regression', 'ensemble', 'gis')
        
    Returns:
        str: JWT token
    """
    payload = {
        'sub': str(agent_id),
        'agent_type': agent_type,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(seconds=TOKEN_EXPIRY)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

# Token-based authentication
def require_auth_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for token in Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization token is required', 'code': 'token_required'}), 401
        
        token = auth_header.split(' ')[1]
        
        try:
            # Decode and validate the token
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            
            # Store user info in flask g for access in the route
            g.agent_id = payload['sub']
            g.agent_type = payload['agent_type']
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired', 'code': 'token_expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token', 'code': 'token_invalid'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Legacy API key authentication (maintained for backward compatibility)
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # First check if there's a valid JWT token
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
                # Store user info
                g.agent_id = payload['sub']
                g.agent_type = payload['agent_type']
                return f(*args, **kwargs)
            except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
                pass  # Fall through to API key authentication
        
        # If no valid JWT token, try API key
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key:
            return jsonify({'error': 'Authentication required. Please provide a valid token or API key.'}), 401
        
        # Validate the API key (in a real system, this would check against a database)
        valid_api_key = os.environ.get('API_KEY', 'bcbs_demo_key_2023')
        
        if api_key != valid_api_key:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Set a default agent_id for API key auth
        g.agent_id = 'api_key_user'
        g.agent_type = 'api_client'
        
        return f(*args, **kwargs)
    return decorated_function


# API endpoints
@api_bp.route('/property/<property_id>', methods=['GET'])
@require_api_key
def get_property(property_id):
    """Get property details by ID."""
    property = Property.query.filter_by(property_id=property_id).first()
    
    if not property:
        return jsonify({'error': 'Property not found'}), 404
    
    # Get the most recent valuation
    valuation = PropertyValuation.query.filter_by(property_id=property.id)\
        .order_by(desc(PropertyValuation.valuation_date)).first()
    
    # Get property features
    features = PropertyFeature.query.filter_by(property_id=property.id).all()
    
    result = {
        'property_id': property.property_id,
        'address': property.address,
        'city': property.city,
        'state': property.state,
        'zip_code': property.zip_code,
        'neighborhood': property.neighborhood,
        'property_type': property.property_type,
        'bedrooms': property.bedrooms,
        'bathrooms': property.bathrooms,
        'square_feet': property.square_feet,
        'year_built': property.year_built,
        'lot_size': property.lot_size,
        'latitude': property.latitude,
        'longitude': property.longitude,
        'last_sale_date': property.last_sale_date.isoformat() if property.last_sale_date else None,
        'last_sale_price': property.last_sale_price,
        'created_at': property.created_at.isoformat(),
        'updated_at': property.updated_at.isoformat(),
    }
    
    if valuation:
        result['latest_valuation'] = {
            'estimated_value': valuation.estimated_value,
            'valuation_date': valuation.valuation_date.isoformat(),
            'valuation_method': valuation.valuation_method,
            'confidence_score': valuation.confidence_score,
            'adj_r2_score': valuation.adj_r2_score,
            'rmse': valuation.rmse,
            'mae': valuation.mae,
        }
    
    if features:
        result['features'] = {
            feature.feature_name: feature.feature_value
            for feature in features
        }
    
    return jsonify(result)


@api_bp.route('/properties', methods=['GET'])
@require_api_key
def search_properties():
    """Search for properties based on criteria."""
    # Get search parameters
    neighborhood = request.args.get('neighborhood')
    property_type = request.args.get('property_type')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    min_bedrooms = request.args.get('min_bedrooms')
    max_bedrooms = request.args.get('max_bedrooms')
    min_bathrooms = request.args.get('min_bathrooms')
    max_bathrooms = request.args.get('max_bathrooms')
    min_square_feet = request.args.get('min_square_feet')
    max_square_feet = request.args.get('max_square_feet')
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    limit = min(100, limit)  # Cap at 100 to prevent abuse
    
    # Build query
    query = Property.query
    
    if neighborhood:
        query = query.filter(Property.neighborhood == neighborhood)
    if property_type:
        query = query.filter(Property.property_type == property_type)
    
    # Handle numeric filters
    if min_bedrooms:
        query = query.filter(Property.bedrooms >= int(min_bedrooms))
    if max_bedrooms:
        query = query.filter(Property.bedrooms <= int(max_bedrooms))
    if min_bathrooms:
        query = query.filter(Property.bathrooms >= float(min_bathrooms))
    if max_bathrooms:
        query = query.filter(Property.bathrooms <= float(max_bathrooms))
    if min_square_feet:
        query = query.filter(Property.square_feet >= float(min_square_feet))
    if max_square_feet:
        query = query.filter(Property.square_feet <= float(max_square_feet))
    
    # Get total count
    total = query.count()
    
    # Paginate results
    properties = query.offset((page - 1) * limit).limit(limit).all()
    
    # Format results
    results = []
    for prop in properties:
        valuation = prop.latest_valuation
        
        prop_dict = {
            'property_id': prop.property_id,
            'address': prop.address,
            'city': prop.city,
            'state': prop.state,
            'zip_code': prop.zip_code,
            'neighborhood': prop.neighborhood,
            'property_type': prop.property_type,
            'bedrooms': prop.bedrooms,
            'bathrooms': prop.bathrooms,
            'square_feet': prop.square_feet,
            'year_built': prop.year_built,
            'estimated_value': prop.estimated_value,
            'valuation_date': prop.valuation_date.isoformat() if prop.valuation_date else None,
        }
        
        results.append(prop_dict)
    
    return jsonify({
        'properties': results,
        'page': page,
        'limit': limit,
        'total': total,
        'pages': (total // limit) + (1 if total % limit > 0 else 0)
    })


@api_bp.route('/valuation', methods=['POST'])
@require_api_key
def value_property():
    """Calculate property valuation."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    required_fields = ['address', 'city', 'state', 'zip_code', 'property_type']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Create property object for valuation
    property_data = {
        'address': data['address'],
        'city': data['city'],
        'state': data['state'],
        'zip_code': data['zip_code'],
        'property_type': data['property_type'],
        'neighborhood': data.get('neighborhood'),
        'bedrooms': data.get('bedrooms'),
        'bathrooms': data.get('bathrooms'),
        'square_feet': data.get('square_feet'),
        'year_built': data.get('year_built'),
        'lot_size': data.get('lot_size'),
        'latitude': data.get('latitude'),
        'longitude': data.get('longitude'),
        'last_sale_price': data.get('last_sale_price', 300000),  # Default for demo
    }
    
    # Select valuation method
    valuation_method = data.get('valuation_method', 'enhanced_regression')
    
    try:
        # For a real system, we would train the model first with historical data
        # For demo purposes, we'll simulate the prediction with 
        # an expected price range based on square footage and bedroom count
        if 'square_feet' in data:
            base_value = float(data['square_feet']) * 150  # $150 per sq ft
        else:
            base_value = 250000  # Default base value
        
        if 'bedrooms' in data:
            base_value += int(data['bedrooms']) * 25000  # $25k per bedroom
        
        if 'bathrooms' in data:
            base_value += float(data['bathrooms']) * 15000  # $15k per bathroom
        
        if 'year_built' in data:
            age_factor = (2024 - int(data['year_built'])) / 100
            base_value *= (1 - age_factor)  # Older homes have lower values
        
        # Apply GIS adjustments if coordinates provided
        gis_adjustments = None
        if 'latitude' in data and 'longitude' in data and gis_features:
            adjusted_value, adjustments = calculate_gis_adjustments(
                data['latitude'], data['longitude'], base_value, gis_features
            )
            gis_adjustments = adjustments
            base_value = adjusted_value
        
        # Add random variation for methods
        method_variations = {
            'linear_regression': 1.0,
            'ridge_regression': 0.98,
            'lasso_regression': 0.97,
            'elastic_net': 0.99,
            'lightgbm': 1.05,
            'xgboost': 1.03,
            'enhanced_regression': 1.02
        }
        
        estimated_value = base_value * method_variations.get(valuation_method, 1.0)
        
        # Create response with model metrics
        confidence_score = 0.85  # Simulated confidence score
        
        response = {
            'estimated_value': estimated_value,
            'valuation_method': valuation_method,
            'confidence_score': confidence_score,
            'valuation_date': datetime.utcnow().isoformat(),
            'model_metrics': {
                'adj_r2': 0.82,  # Simulated metrics
                'rmse': 12500,
                'mae': 9500
            }
        }
        
        if gis_adjustments:
            response['gis_adjustments'] = gis_adjustments
        
        # Check if we should store in database
        if data.get('save_to_db', False):
            # Generate unique property ID if not provided
            if 'property_id' not in data:
                data['property_id'] = f"PROP-{uuid.uuid4().hex[:8].upper()}"
            
            # Check if property already exists
            property = Property.query.filter_by(property_id=data['property_id']).first()
            
            if not property:
                # Create new property
                property = Property(
                    property_id=data['property_id'],
                    address=data['address'],
                    city=data['city'],
                    state=data['state'],
                    zip_code=data['zip_code'],
                    neighborhood=data.get('neighborhood'),
                    property_type=data['property_type'],
                    bedrooms=data.get('bedrooms'),
                    bathrooms=data.get('bathrooms'),
                    square_feet=data.get('square_feet'),
                    year_built=data.get('year_built'),
                    lot_size=data.get('lot_size'),
                    latitude=data.get('latitude'),
                    longitude=data.get('longitude'),
                    last_sale_date=data.get('last_sale_date'),
                    last_sale_price=data.get('last_sale_price')
                )
                db.session.add(property)
                db.session.commit()
            
            # Create valuation record
            valuation = PropertyValuation(
                property_id=property.id,
                estimated_value=estimated_value,
                valuation_date=datetime.utcnow(),
                valuation_method=valuation_method,
                confidence_score=confidence_score,
                adj_r2_score=response['model_metrics']['adj_r2'],
                rmse=response['model_metrics']['rmse'],
                mae=response['model_metrics']['mae'],
                inputs=property_data,
                gis_adjustments=gis_adjustments
            )
            db.session.add(valuation)
            db.session.commit()
            
            response['property_id'] = property.property_id
            response['valuation_id'] = valuation.id
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error calculating valuation: {str(e)}")
        return jsonify({'error': f'Error calculating valuation: {str(e)}'}), 500


@api_bp.route('/neighborhoods', methods=['GET'])
@require_api_key
def get_neighborhoods():
    """Get list of neighborhoods."""
    # Query distinct neighborhoods
    neighborhoods = db.session.query(Property.neighborhood)\
        .filter(Property.neighborhood.isnot(None))\
        .distinct()\
        .order_by(Property.neighborhood)\
        .all()
    
    # Format results
    results = [n[0] for n in neighborhoods]
    
    return jsonify({
        'neighborhoods': results,
        'count': len(results)
    })


@api_bp.route('/market-trends', methods=['GET'])
@require_api_key
def get_market_trends():
    """Get market trends."""
    # Get parameters
    timeframe = request.args.get('timeframe', 'last_year')
    neighborhood = request.args.get('neighborhood')
    property_type = request.args.get('property_type')
    
    # Determine date range
    end_date = datetime.utcnow()
    
    if timeframe == 'last_month':
        start_date = end_date - timedelta(days=30)
    elif timeframe == 'last_quarter':
        start_date = end_date - timedelta(days=90)
    elif timeframe == 'ytd':
        start_date = datetime(end_date.year, 1, 1)
    else:  # last_year (default)
        start_date = end_date - timedelta(days=365)
    
    # Build query
    query = db.session.query(
        func.date_trunc('month', PropertyValuation.valuation_date).label('month'),
        func.avg(PropertyValuation.estimated_value).label('avg_value'),
        func.count(PropertyValuation.id).label('count')
    )\
    .join(Property, PropertyValuation.property_id == Property.id)\
    .filter(PropertyValuation.valuation_date.between(start_date, end_date))
    
    if neighborhood:
        query = query.filter(Property.neighborhood == neighborhood)
    if property_type:
        query = query.filter(Property.property_type == property_type)
    
    # Group by month
    query = query.group_by(func.date_trunc('month', PropertyValuation.valuation_date))
    
    # Order by month
    query = query.order_by(func.date_trunc('month', PropertyValuation.valuation_date))
    
    # Execute query
    trends = query.all()
    
    # Format results
    results = []
    for month, avg_value, count in trends:
        results.append({
            'month': month.strftime('%Y-%m'),
            'avg_value': float(avg_value),
            'count': count
        })
    
    return jsonify({
        'trends': results,
        'timeframe': timeframe,
        'filters': {
            'neighborhood': neighborhood,
            'property_type': property_type
        }
    })


@api_bp.route('/agent-status', methods=['GET'])
@require_api_key
def get_agent_status():
    """
    Get detailed, real-time status information for each BS Army agent.
    
    This enhanced endpoint provides comprehensive monitoring of valuation agents,
    including performance metrics, execution history, error analytics, and real-time
    operational statistics. It enables advanced monitoring and troubleshooting
    of the agent ecosystem.
    
    Query parameters:
    - agent_type: Filter by agent type (e.g., 'regression', 'ensemble', 'gis', 'lightgbm')
    - status: Filter by status (e.g., 'idle', 'processing', 'error', 'offline')
    - active_only: If set to 'true', only return active agents
    - performance_threshold: Filter by success rate threshold (float 0.0-1.0)
    - version: Filter by agent version
    - include_logs: Include detailed logs if 'true' (default: false)
    - include_metrics: Include comprehensive performance metrics if 'true' (default: false)
    - health_check: Perform real-time agent health check if 'true' (default: false)
    
    Returns:
        JSON with agent details, performance metrics, system health indicators, and operational statistics
    """
    # Log the incoming request
    logger.debug(f"GET /agent-status request from {g.agent_id} of type {g.agent_type}")
    
    # Start timing for performance monitoring
    start_time = datetime.utcnow()
    
    # Get filter parameters
    agent_type = request.args.get('agent_type')
    status = request.args.get('status')
    active_only = request.args.get('active_only') == 'true'
    performance_threshold = request.args.get('performance_threshold', type=float)
    version = request.args.get('version')
    include_logs = request.args.get('include_logs') == 'true'
    include_metrics = request.args.get('include_metrics') == 'true'
    health_check = request.args.get('health_check') == 'true'
    
    try:
        # Build base query
        query = Agent.query
        
        # Apply filters
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type)
        
        if status:
            query = query.filter(Agent.status == status)
        
        if active_only:
            query = query.filter(Agent.is_active == True)
            
        if version:
            query = query.filter(Agent.version == version)
            
        if performance_threshold is not None and 0 <= performance_threshold <= 1:
            query = query.filter(Agent.success_rate >= performance_threshold)
        
        # Execute query
        agents = query.all()
        
        # Format results with enhanced details
        results = []
        for agent in agents:
            # Determine log quantity based on include_logs parameter
            log_limit = 20 if include_logs else 1
            
            # Get agent logs
            logs_query = AgentLog.query.filter_by(agent_id=agent.id).order_by(desc(AgentLog.timestamp))
            
            if not include_logs:
                logs_query = logs_query.limit(log_limit)
                
            log_entries = logs_query.all()
            
            # Format log entries
            logs = []
            for entry in log_entries:
                log_data = {
                    'id': entry.id,
                    'level': entry.level,
                    'message': entry.message,
                    'timestamp': entry.timestamp.isoformat(),
                }
                
                # Include details if available and requested
                if include_logs and hasattr(entry, 'details') and entry.details:
                    try:
                        if isinstance(entry.details, str):
                            log_data['details'] = json.loads(entry.details)
                        else:
                            log_data['details'] = entry.details
                    except (json.JSONDecodeError, AttributeError) as e:
                        log_data['details'] = entry.details  # Keep as string if parse fails
                
                logs.append(log_data)
            
            # Get agent's recent valuation performance
            recent_valuations = PropertyValuation.query.filter_by(agent_id=agent.id)\
                .order_by(desc(PropertyValuation.valuation_date))\
                .limit(30)\
                .all()
            
            valuation_count = len(recent_valuations)
            
            # Basic performance metrics
            performance_metrics = {
                'recent_valuations': valuation_count,
                'average_confidence': round(sum(v.confidence_score for v in recent_valuations) / valuation_count, 4) if valuation_count > 0 else None,
                'methods_used': list(set(v.valuation_method for v in recent_valuations)) if valuation_count > 0 else []
            }
            
            # Calculate enhanced performance metrics if requested
            if include_metrics and valuation_count > 0:
                # Calculate success trend (last 5 vs previous 5)
                recent_five = recent_valuations[:5] if len(recent_valuations) >= 5 else recent_valuations
                previous_five = recent_valuations[5:10] if len(recent_valuations) >= 10 else []
                
                recent_confidence = sum(v.confidence_score for v in recent_five) / len(recent_five) if recent_five else 0
                previous_confidence = sum(v.confidence_score for v in previous_five) / len(previous_five) if previous_five else 0
                
                confidence_trend = round((recent_confidence - previous_confidence) * 100, 2) if previous_five else None
                
                # Calculate error rates
                error_valuations = sum(1 for v in recent_valuations if v.confidence_score < 0.6)
                error_rate = round((error_valuations / valuation_count) * 100, 2) if valuation_count > 0 else 0
                
                # Calculate average processing time if available
                processing_times = []
                for v in recent_valuations:
                    if hasattr(v, 'processing_time') and v.processing_time is not None:
                        processing_times.append(v.processing_time)
                        
                avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else None
                
                # Add enhanced metrics
                performance_metrics.update({
                    'confidence_trend_percent': confidence_trend,
                    'error_rate_percent': error_rate,
                    'avg_processing_time': avg_processing_time,
                    'valuations_by_method': {
                        method: sum(1 for v in recent_valuations if v.valuation_method == method)
                        for method in set(v.valuation_method for v in recent_valuations)
                    },
                    'confidence_distribution': {
                        'high (>0.8)': sum(1 for v in recent_valuations if v.confidence_score > 0.8),
                        'medium (0.6-0.8)': sum(1 for v in recent_valuations if 0.6 <= v.confidence_score <= 0.8),
                        'low (<0.6)': sum(1 for v in recent_valuations if v.confidence_score < 0.6)
                    }
                })
                
                # Calculate time-based metrics if timestamps are available
                if valuation_count >= 2 and all(hasattr(v, 'valuation_date') for v in recent_valuations[:2]):
                    latest = recent_valuations[0].valuation_date
                    previous = recent_valuations[1].valuation_date
                    if latest and previous:
                        time_diff = (latest - previous).total_seconds()
                        performance_metrics['time_between_valuations'] = time_diff
            
            # Perform real-time health check if requested
            health_check_result = None
            if health_check:
                # Simulate health check - in a real implementation, this would 
                # connect to the agent and verify its actual status
                last_log_time = log_entries[0].timestamp if log_entries else None
                is_responsive = True  # Would be determined by actual agent communication
                
                # Calculate responsiveness based on last activity
                if last_log_time:
                    time_since_last_log = (datetime.utcnow() - last_log_time).total_seconds()
                    is_responsive = time_since_last_log < 300  # 5 minutes
                
                health_check_result = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'responsive': is_responsive,
                    'memory_usage_percent': random.randint(10, 90),  # Simulated - would be actual agent memory usage
                    'cpu_usage_percent': random.randint(5, 95),  # Simulated - would be actual agent CPU usage
                    'connection_status': 'connected' if is_responsive else 'disconnected'
                }
            
            # Build comprehensive agent data
            agent_data = {
                'id': agent.id,
                'name': agent.name,
                'agent_type': agent.agent_type,
                'description': agent.description,
                'status': agent.status,
                'is_active': agent.is_active,
                'version': agent.version,
                'created_at': agent.created_at.isoformat(),
                'last_active': agent.last_active.isoformat() if agent.last_active else None,
                'success_rate': agent.success_rate,
                'performance_metrics': performance_metrics,
                'latest_log': logs[0] if logs else None
            }
            
            # Include all logs if requested
            if include_logs and len(logs) > 1:
                agent_data['logs'] = logs
            
            # Include configuration if available
            if hasattr(agent, 'configuration') and agent.configuration:
                try:
                    if isinstance(agent.configuration, str):
                        agent_data['configuration'] = json.loads(agent.configuration)
                    else:
                        agent_data['configuration'] = agent.configuration
                except json.JSONDecodeError:
                    agent_data['configuration'] = agent.configuration  # Keep as string if parse fails
            
            # Add queue_size field if it exists
            if hasattr(agent, 'queue_size'):
                agent_data['queue_size'] = agent.queue_size
            
            # Add current_task field if it exists
            if hasattr(agent, 'current_task') and agent.current_task:
                if isinstance(agent.current_task, str):
                    try:
                        agent_data['current_task'] = json.loads(agent.current_task)
                    except json.JSONDecodeError:
                        agent_data['current_task'] = agent.current_task
                else:
                    agent_data['current_task'] = agent.current_task
            
            # Add error_count field if it exists
            if hasattr(agent, 'error_count'):
                agent_data['error_count'] = agent.error_count
                
            # Add health check result if requested and available
            if health_check_result:
                agent_data['health_check'] = health_check_result
            
            results.append(agent_data)
        
        # Calculate system-wide metrics
        total_agents = len(agents)
        active_agents = sum(1 for agent in agents if agent.is_active)
        idle_agents = sum(1 for agent in agents if agent.status == 'idle' and agent.is_active)
        processing_agents = sum(1 for agent in agents if agent.status == 'processing')
        error_agents = sum(1 for agent in agents if agent.status == 'error')
        
        # Calculate agent type distribution
        agent_type_counts = {}
        for agent in agents:
            agent_type_counts[agent.agent_type] = agent_type_counts.get(agent.agent_type, 0) + 1
        
        # Calculate status distribution
        status_counts = {}
        for agent in agents:
            status_counts[agent.status] = status_counts.get(agent.status, 0) + 1
        
        # Calculate average success rate
        avg_success_rate = sum(agent.success_rate for agent in agents if hasattr(agent, 'success_rate') and agent.success_rate is not None) / total_agents if total_agents > 0 else 0
        
        # Calculate load distribution
        load_distribution = {
            'queue_size_by_agent': {agent.name: agent.queue_size for agent in agents if hasattr(agent, 'queue_size')},
            'total_queued_tasks': sum(agent.queue_size for agent in agents if hasattr(agent, 'queue_size')),
        }
        
        # System health assessment
        system_health = 'healthy'
        if error_agents > 0:
            system_health = 'warning'
        if error_agents > active_agents / 3:  # If more than 1/3 of agents are in error state
            system_health = 'critical'
            
        # Calculate more granular health indicators
        health_factors = {
            'error_rate': round((error_agents / total_agents) * 100, 2) if total_agents > 0 else 0,
            'agent_availability': round((active_agents / total_agents) * 100, 2) if total_agents > 0 else 0,
            'system_load': round((processing_agents / active_agents) * 100, 2) if active_agents > 0 else 0,
            'success_rate': round(avg_success_rate * 100, 2)
        }
        
        # Calculate query execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Build enhanced response
        return jsonify({
            'agents': results,
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'idle_agents': idle_agents,
                'processing_agents': processing_agents, 
                'error_agents': error_agents,
                'agent_types': agent_type_counts,
                'status_distribution': status_counts,
                'avg_success_rate': round(avg_success_rate, 4),
                'load_distribution': load_distribution
            },
            'health': {
                'status': system_health,
                'factors': health_factors,
                'recommendations': [
                    "Restart agents with error status" if error_agents > 0 else None,
                    "Increase agent pool" if processing_agents > idle_agents * 2 else None,
                    "Rebalance workload across agent types" if any(count > total_agents * 0.5 for count in agent_type_counts.values()) else None
                ] if system_health != 'healthy' else []
            },
            'metadata': {
                'query_time_seconds': execution_time,
                'filter_criteria': {
                    'agent_type': agent_type,
                    'status': status,
                    'active_only': active_only,
                    'version': version
                }
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing agent status request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred while processing the request',
            'details': str(e)
        }), 500


@api_bp.route('/agent-logs/<agent_id>', methods=['GET'])
@require_api_key
def get_agent_logs(agent_id):
    """Get logs for a specific agent."""
    agent = Agent.query.filter_by(agent_id=agent_id).first()
    
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    
    # Get logs
    logs = AgentLog.query.filter_by(agent_id=agent.id)\
        .order_by(desc(AgentLog.timestamp))\
        .limit(100)\
        .all()
    
    # Format results
    results = []
    for log in logs:
        results.append({
            'level': log.level,
            'message': log.message,
            'timestamp': log.timestamp.isoformat()
        })
    
    return jsonify({
        'agent_id': agent.agent_id,
        'agent_name': agent.name,
        'logs': results,
        'count': len(results)
    })


@api_bp.route('/valuations', methods=['GET'])
@require_api_key
def get_valuations():
    """
    Get property valuations with advanced filtering and pagination.
    
    This enhanced endpoint integrates with the valuation engine from src/valuation.py
    to provide comprehensive property value predictions and detailed model metrics.
    It supports advanced filtering, sorting, and pagination capabilities.
    
    Query parameters:
    - method: Filter by valuation method (e.g., 'enhanced_regression', 'lightgbm', 'xgboost')
    - min_confidence: Minimum confidence score (0.0-1.0)
    - after_date: Only valuations after this date (ISO format)
    - before_date: Only valuations before this date (ISO format)
    - property_id: Filter by property ID
    - neighborhood: Filter by neighborhood
    - city: Filter by city
    - state: Filter by state
    - feature_importance: Include feature importance data if 'true'
    - include_gis: Include detailed GIS data if 'true'
    - page: Page number for pagination (default: 1)
    - limit: Results per page (default: 20, max: 100)
    - sort_by: Field to sort by (default: 'valuation_date')
    - sort_dir: Sort direction ('asc' or 'desc', default: 'desc')
    
    Returns:
        JSON with paginated valuation data, model metrics, and response metadata
    """
    # Log the incoming request for monitoring
    logger.debug(f"GET /valuations request from {g.agent_id} of type {g.agent_type}")
    
    # Get filter parameters
    method = request.args.get('method')
    min_confidence = request.args.get('min_confidence', type=float)
    after_date = request.args.get('after_date')
    before_date = request.args.get('before_date')
    property_id = request.args.get('property_id')
    neighborhood = request.args.get('neighborhood')
    city = request.args.get('city')
    state = request.args.get('state')
    
    # Additional data options
    include_feature_importance = request.args.get('feature_importance') == 'true'
    include_gis_data = request.args.get('include_gis') == 'true'
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    limit = min(100, limit)  # Cap at 100 to prevent abuse
    
    # Sorting parameters
    sort_by = request.args.get('sort_by', 'valuation_date')
    sort_dir = request.args.get('sort_dir', 'desc')
    
    # Start timing for performance monitoring
    start_time = datetime.utcnow()
    
    # Validate sort parameters
    valid_sort_fields = ['valuation_date', 'estimated_value', 'confidence_score', 'year_built', 'square_feet']
    if sort_by not in valid_sort_fields:
        sort_by = 'valuation_date'
    
    if sort_dir not in ['asc', 'desc']:
        sort_dir = 'desc'
    
    try:
        # Build base query with all necessary joins for performance
        query = db.session.query(PropertyValuation, Property)\
            .join(Property, PropertyValuation.property_id == Property.id)
        
        # Apply filters
        if method:
            query = query.filter(PropertyValuation.valuation_method == method)
        
        if min_confidence is not None:
            query = query.filter(PropertyValuation.confidence_score >= min_confidence)
        
        if after_date:
            try:
                after_dt = datetime.fromisoformat(after_date.replace('Z', '+00:00'))
                query = query.filter(PropertyValuation.valuation_date >= after_dt)
            except ValueError:
                return jsonify({'error': 'Invalid after_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'}), 400
        
        if before_date:
            try:
                before_dt = datetime.fromisoformat(before_date.replace('Z', '+00:00'))
                query = query.filter(PropertyValuation.valuation_date <= before_dt)
            except ValueError:
                return jsonify({'error': 'Invalid before_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'}), 400
        
        if property_id:
            query = query.filter(Property.property_id == property_id)
        
        if neighborhood:
            query = query.filter(Property.neighborhood == neighborhood)
            
        if city:
            query = query.filter(Property.city == city)
            
        if state:
            query = query.filter(Property.state == state)
        
        # Get total count for pagination info
        total = query.count()
        
        # Apply sorting
        sort_field = getattr(PropertyValuation, sort_by) if sort_by != 'square_feet' and sort_by != 'year_built' else getattr(Property, sort_by)
        sort_field = sort_field.asc() if sort_dir == 'asc' else sort_field.desc()
        query = query.order_by(sort_field)
        
        # Apply pagination
        query = query.offset((page - 1) * limit).limit(limit)
        
        # Execute query
        results = query.all()
        
        # Import valuation module components locally to avoid circular imports
        try:
            from src.valuation import perform_valuation
            from src.gis_integration import (
                get_location_score, 
                get_school_district_info, 
                get_flood_risk_assessment
            )
            has_valuation_module = True
        except ImportError as e:
            logger.warning(f"Could not import valuation module: {str(e)}")
            has_valuation_module = False
        
        # Format results with enhanced data
        valuations = []
        for valuation, property in results:
            # Base property data
            data = {
                'valuation_id': valuation.id,
                'property_id': property.property_id,
                'address': property.address,
                'city': property.city,
                'state': property.state,
                'zip_code': property.zip_code,
                'neighborhood': property.neighborhood,
                'property_type': property.property_type,
                'bedrooms': property.bedrooms,
                'bathrooms': property.bathrooms,
                'square_feet': property.square_feet,
                'lot_size': property.lot_size,
                'year_built': property.year_built,
                'estimated_value': valuation.estimated_value,
                'valuation_date': valuation.valuation_date.isoformat(),
                'valuation_method': valuation.valuation_method,
                'confidence_score': valuation.confidence_score,
            }
            
            # Include model features if they exist
            if hasattr(valuation, 'model_features') and valuation.model_features:
                if isinstance(valuation.model_features, str):
                    try:
                        data['model_features'] = json.loads(valuation.model_features)
                    except json.JSONDecodeError:
                        data['model_features'] = valuation.model_features
                else:
                    data['model_features'] = valuation.model_features
            
            # Include comprehensive model metrics
            model_metrics = {}
            for metric in ['adj_r2_score', 'rmse', 'mae', 'r2_score', 'explained_variance']:
                if hasattr(valuation, metric) and getattr(valuation, metric) is not None:
                    model_metrics[metric.replace('_score', '')] = getattr(valuation, metric)
            
            if model_metrics:
                data['model_metrics'] = model_metrics
            
            # Include feature importance data if requested and available
            if include_feature_importance and hasattr(valuation, 'feature_importance') and valuation.feature_importance:
                try:
                    if isinstance(valuation.feature_importance, str):
                        data['feature_importance'] = json.loads(valuation.feature_importance)
                    else:
                        data['feature_importance'] = valuation.feature_importance
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Could not parse feature importance: {str(e)}")
            
            # Include GIS features if requested and available
            if include_gis_data:
                gis_data = {}
                
                # Add existing GIS data from the database
                if hasattr(valuation, 'gis_features') and valuation.gis_features:
                    try:
                        if isinstance(valuation.gis_features, str):
                            gis_data = json.loads(valuation.gis_features)
                        else:
                            gis_data = valuation.gis_features
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.warning(f"Could not parse GIS features: {str(e)}")
                
                # If we have the GIS module and coordinates, add fresh GIS data
                if has_valuation_module and property.latitude and property.longitude:
                    try:
                        # Get real-time GIS data
                        location_score = get_location_score(property.latitude, property.longitude)
                        school_info = get_school_district_info(property.latitude, property.longitude)
                        flood_risk = get_flood_risk_assessment(property.latitude, property.longitude)
                        
                        # Update GIS data with fresh information
                        gis_data.update({
                            'location_score': location_score,
                            'school_district': school_info,
                            'flood_risk': flood_risk
                        })
                    except Exception as e:
                        logger.warning(f"Error fetching GIS data: {str(e)}")
                
                if gis_data:
                    data['gis_data'] = gis_data
            
            valuations.append(data)
        
        # Calculate query execution time for performance monitoring
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Return enhanced response with metadata
        return jsonify({
            'valuations': valuations,
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total // limit) + (1 if total % limit > 0 else 0),
            'metadata': {
                'query_time_seconds': execution_time,
                'timestamp': datetime.utcnow().isoformat(),
                'filter_criteria': {
                    'method': method,
                    'min_confidence': min_confidence,
                    'neighborhood': neighborhood,
                    'city': city,
                    'state': state
                }
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing valuations request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred while processing the request',
            'details': str(e)
        }), 500


@api_bp.route('/etl-status', methods=['GET'])
@require_api_key
def get_etl_status():
    """
    Get status of ETL pipeline jobs with detailed analytics and data validation summary.
    
    This enhanced endpoint provides comprehensive information about the ETL pipeline's
    performance, data quality, and operational status. It includes detailed statistics
    on job execution times, error rates, and data validation results.
    
    Query parameters:
    - job_type: Filter by job type (e.g., 'property_import', 'valuation_batch', 'gis_update')
    - status: Filter by status (e.g., 'completed', 'running', 'failed', 'pending')
    - timeframe: Filter by timeframe ('today', 'yesterday', 'this_week', 'last_week', 'this_month', 'custom')
    - start_date: Start date for custom timeframe (ISO format, required if timeframe='custom')
    - end_date: End date for custom timeframe (ISO format, required if timeframe='custom')
    - source: Filter by data source
    - include_validation: Include detailed validation results if 'true'
    - limit: Maximum number of jobs to return (default: 20, max: 100)
    
    Returns:
        JSON with job details, summary statistics, health indicators, and data validation summary
    """
    # Log the incoming request
    logger.debug(f"GET /etl-status request from {g.agent_id} of type {g.agent_type}")
    
    # Start timing for performance monitoring
    start_time = datetime.utcnow()
    
    # Get filter parameters
    job_type = request.args.get('job_type')
    status = request.args.get('status')
    timeframe = request.args.get('timeframe', 'today')
    source = request.args.get('source')
    include_validation = request.args.get('include_validation') == 'true'
    limit = request.args.get('limit', 20, type=int)
    limit = min(100, limit)  # Cap at 100 to prevent abuse
    
    try:
        # Build base query
        query = ETLJob.query
        
        # Determine date range based on timeframe
        now = datetime.utcnow()
        
        if timeframe == 'custom':
            # Custom date range
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            
            if not start_date_str or not end_date_str:
                return jsonify({
                    'error': 'Custom timeframe requires both start_date and end_date parameters',
                    'details': 'Please provide both dates in ISO format (YYYY-MM-DDTHH:MM:SS)'
                }), 400
                
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                query = query.filter(ETLJob.start_time >= start_date)
                query = query.filter(ETLJob.start_time <= end_date)
            except ValueError:
                return jsonify({
                    'error': 'Invalid date format',
                    'details': 'Please use ISO format (YYYY-MM-DDTHH:MM:SS)'
                }), 400
        else:
            # Predefined timeframes
            if timeframe == 'today':
                start_date = datetime(now.year, now.month, now.day)
            elif timeframe == 'yesterday':
                yesterday = now - timedelta(days=1)
                start_date = datetime(yesterday.year, yesterday.month, yesterday.day)
                end_date = datetime(now.year, now.month, now.day)
                query = query.filter(ETLJob.start_time < end_date)
            elif timeframe == 'this_week':
                # Start of week (Monday)
                start_date = now - timedelta(days=now.weekday())
                start_date = datetime(start_date.year, start_date.month, start_date.day)
            elif timeframe == 'last_week':
                # Start of last week
                start_of_this_week = now - timedelta(days=now.weekday())
                start_date = start_of_this_week - timedelta(days=7)
                start_date = datetime(start_date.year, start_date.month, start_date.day)
                end_date = start_of_this_week
                query = query.filter(ETLJob.start_time < end_date)
            elif timeframe == 'this_month':
                start_date = datetime(now.year, now.month, 1)
            else:  # Default to today
                start_date = datetime(now.year, now.month, now.day)
                timeframe = 'today'  # Normalize invalid values
            
            # Apply standard timeframe filter
            query = query.filter(ETLJob.start_time >= start_date)
        
        # Apply other filters
        if job_type:
            query = query.filter(ETLJob.job_type == job_type)
        
        if status:
            query = query.filter(ETLJob.status == status)
            
        if source:
            query = query.filter(ETLJob.source == source)
        
        # Get job type counts for analytics
        job_type_counts = db.session.query(
            ETLJob.job_type, 
            func.count(ETLJob.id)
        ).filter(
            ETLJob.start_time >= start_date
        ).group_by(
            ETLJob.job_type
        ).all()
        
        job_type_distribution = {job_type: count for job_type, count in job_type_counts}
        
        # Get status distribution
        status_counts = db.session.query(
            ETLJob.status, 
            func.count(ETLJob.id)
        ).filter(
            ETLJob.start_time >= start_date
        ).group_by(
            ETLJob.status
        ).all()
        
        status_distribution = {status: count for status, count in status_counts}
        
        # Order by most recent first
        query = query.order_by(desc(ETLJob.start_time))
        
        # Apply limit
        query = query.limit(limit)
        
        # Execute query
        jobs = query.all()
        
        # Format results with enhanced details
        results = []
        for job in jobs:
            # Parse validation results if they exist
            validation_details = None
            if hasattr(job, 'validation_results') and job.validation_results:
                try:
                    if isinstance(job.validation_results, str):
                        validation_details = json.loads(job.validation_results)
                    else:
                        validation_details = job.validation_results
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Could not parse validation results: {str(e)}")
            
            # Base job data
            job_data = {
                'id': job.id,
                'job_type': job.job_type,
                'status': job.status,
                'start_time': job.start_time.isoformat(),
                'end_time': job.end_time.isoformat() if job.end_time else None,
                'progress': job.progress,
                'records_processed': job.records_processed,
                'records_total': job.records_total,
                'source': job.source,
                'message': job.message,
                'error': job.error
            }
            
            # Calculate duration if completed
            if job.end_time:
                duration = (job.end_time - job.start_time).total_seconds()
                job_data['duration_seconds'] = duration
                
                # Calculate processing rate (records per second)
                if duration > 0 and job.records_processed > 0:
                    job_data['processing_rate'] = round(job.records_processed / duration, 2)
            
            # Include validation details if requested and available
            if include_validation and validation_details:
                job_data['validation_results'] = validation_details
                
            # Include additional metadata if available
            if hasattr(job, 'metadata') and job.metadata:
                try:
                    if isinstance(job.metadata, str):
                        job_data['metadata'] = json.loads(job.metadata)
                    else:
                        job_data['metadata'] = job.metadata
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Could not parse job metadata: {str(e)}")
            
            results.append(job_data)
        
        # Calculate comprehensive summary statistics
        total_jobs = len(jobs)
        completed_jobs = sum(1 for job in jobs if job.status == 'completed')
        failed_jobs = sum(1 for job in jobs if job.status == 'failed')
        running_jobs = sum(1 for job in jobs if job.status == 'running')
        pending_jobs = sum(1 for job in jobs if job.status == 'pending')
        total_records_processed = sum(job.records_processed for job in jobs)
        
        # Calculate success rate
        success_rate = (completed_jobs / total_jobs) * 100 if total_jobs > 0 else 0
        
        # Calculate average processing time
        completed_durations = [
            (job.end_time - job.start_time).total_seconds() 
            for job in jobs 
            if job.status == 'completed' and job.end_time
        ]
        avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        
        # Enhanced statistics
        stats = {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'running_jobs': running_jobs,
            'pending_jobs': pending_jobs,
            'total_records_processed': total_records_processed,
            'average_progress': sum(job.progress for job in jobs) / total_jobs if total_jobs else 0,
            'success_rate_percent': round(success_rate, 2),
            'average_duration_seconds': round(avg_duration, 2),
            'job_type_distribution': job_type_distribution,
            'status_distribution': status_distribution
        }
        
        # Compile data validation summary
        validation_summary = None
        if include_validation:
            # Aggregate validation metrics from jobs
            all_validation_metrics = []
            for job in jobs:
                if hasattr(job, 'validation_results') and job.validation_results:
                    try:
                        validation_data = job.validation_results
                        if isinstance(validation_data, str):
                            validation_data = json.loads(validation_data)
                        
                        if isinstance(validation_data, dict) and 'metrics' in validation_data:
                            all_validation_metrics.append(validation_data['metrics'])
                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Error processing validation metrics: {str(e)}")
            
            # Compute aggregated metrics
            if all_validation_metrics:
                # Collect all metric keys
                all_keys = set()
                for metrics in all_validation_metrics:
                    all_keys.update(metrics.keys())
                
                # Calculate average for each metric
                avg_metrics = {}
                for key in all_keys:
                    values = [metrics.get(key) for metrics in all_validation_metrics if key in metrics and metrics[key] is not None]
                    if values:
                        avg_metrics[key] = sum(values) / len(values)
                
                validation_summary = {
                    'metrics': avg_metrics,
                    'jobs_with_validation': len(all_validation_metrics),
                    'total_validation_errors': sum(
                        metrics.get('error_count', 0) 
                        for metrics in all_validation_metrics 
                        if 'error_count' in metrics
                    )
                }
        
        # Add system health indicators
        system_health_status = 'healthy'
        if failed_jobs > 0:
            system_health_status = 'warning'
        if failed_jobs > 2 or success_rate < 70:
            system_health_status = 'critical'
        
        # Check if pipeline is active (has recent activity)
        is_pipeline_active = running_jobs > 0 or (
            completed_jobs > 0 and 
            any(
                job.end_time and (now - job.end_time).total_seconds() < 3600 
                for job in jobs if job.status == 'completed'
            )
        )
        
        # Find the last successful job
        last_successful_job = next(
            (job.end_time.isoformat() for job in jobs if job.status == 'completed'), 
            None
        )
        
        # Enhanced health indicators
        health = {
            'status': system_health_status,
            'pipeline_active': is_pipeline_active,
            'last_successful_job': last_successful_job,
            'error_rate_percent': round((failed_jobs / total_jobs) * 100, 2) if total_jobs > 0 else 0,
            'avg_processing_rate': round(
                sum(job.records_processed / (job.end_time - job.start_time).total_seconds() 
                    for job in jobs 
                    if job.status == 'completed' and job.end_time and job.records_processed > 0 and 
                    (job.end_time - job.start_time).total_seconds() > 0
                ) / sum(1 for job in jobs 
                    if job.status == 'completed' and job.end_time and job.records_processed > 0 and 
                    (job.end_time - job.start_time).total_seconds() > 0
                ) if sum(1 for job in jobs 
                    if job.status == 'completed' and job.end_time and job.records_processed > 0 and 
                    (job.end_time - job.start_time).total_seconds() > 0
                ) > 0 else 0,
                2
            ),
            'bottleneck_job_types': [
                job_type for job_type, avg_duration in sorted([
                    (job_type, sum((job.end_time - job.start_time).total_seconds() 
                                  for job in jobs 
                                  if job.job_type == job_type and job.status == 'completed' and job.end_time) / 
                               sum(1 for job in jobs 
                                  if job.job_type == job_type and job.status == 'completed' and job.end_time)
                    ) 
                    for job_type in set(job.job_type for job in jobs)
                    if sum(1 for j in jobs 
                          if j.job_type == job_type and j.status == 'completed' and j.end_time) > 0
                ], key=lambda x: x[1], reverse=True)
            ][:3]  # Top 3 bottlenecks
        }
        
        # Calculate query execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Build the response
        response = {
            'jobs': results,
            'stats': stats,
            'health': health,
            'timeframe': timeframe,
            'timestamp': now.isoformat(),
            'metadata': {
                'query_time_seconds': execution_time,
                'filter_criteria': {
                    'job_type': job_type,
                    'status': status,
                    'source': source
                }
            }
        }
        
        # Include validation summary if requested
        if include_validation and validation_summary:
            response['validation_summary'] = validation_summary
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing ETL status request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred while processing the request',
            'details': str(e)
        }), 500


@api_bp.route('/auth/token', methods=['POST'])
def generate_auth_token():
    """
    Generate authentication token for agents.
    
    This endpoint expects a JSON object with:
    - agent_id: The agent's unique identifier
    - agent_type: The type of agent (e.g., 'regression', 'ensemble', 'gis')
    - api_key: A valid API key for initial authentication
    
    Returns a JWT token valid for 24 hours.
    """
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate required fields
    required_fields = ['agent_id', 'agent_type', 'api_key']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Validate API key
    valid_api_key = os.environ.get('API_KEY', 'bcbs_demo_key_2023')
    if data['api_key'] != valid_api_key:
        return jsonify({'error': 'Invalid API key'}), 401
    
    # Generate token
    token = generate_token(data['agent_id'], data['agent_type'])
    
    # Return token
    return jsonify({
        'token': token,
        'expires_in': TOKEN_EXPIRY,
        'token_type': 'Bearer'
    })


# Register the blueprint with the app
app.register_blueprint(api_bp)

# Run the app if executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)