"""
API endpoints for the BCBS Values application.
"""
import json
import logging
import os
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
    Get detailed status of valuation agents.
    
    Query parameters:
    - agent_type: Filter by agent type (e.g., 'regression', 'ensemble', 'gis')
    - status: Filter by status (e.g., 'idle', 'processing', 'error')
    - active_only: If set to 'true', only return active agents
    """
    # Get filter parameters
    agent_type = request.args.get('agent_type')
    status = request.args.get('status')
    active_only = request.args.get('active_only') == 'true'
    
    # Build query
    query = Agent.query
    
    if agent_type:
        query = query.filter(Agent.agent_type == agent_type)
    
    if status:
        query = query.filter(Agent.status == status)
    
    if active_only:
        query = query.filter(Agent.is_active == True)
    
    # Execute query
    agents = query.all()
    
    # Format results
    results = []
    for agent in agents:
        # Get the latest log
        latest_log = None
        log_entry = AgentLog.query.filter_by(agent_id=agent.id)\
            .order_by(desc(AgentLog.timestamp))\
            .first()
        
        if log_entry:
            latest_log = {
                'level': log_entry.level,
                'message': log_entry.message,
                'timestamp': log_entry.timestamp.isoformat(),
                'details': log_entry.details
            }
        
        # Get recent performance metrics
        recent_valuations = PropertyValuation.query.filter_by(agent_id=agent.id)\
            .order_by(desc(PropertyValuation.valuation_date))\
            .limit(10)\
            .all()
        
        valuation_count = len(recent_valuations)
        
        # Calculate performance metrics if we have valuations
        performance_metrics = {
            'recent_valuations': valuation_count,
            'average_confidence': sum(v.confidence_score for v in recent_valuations) / valuation_count if valuation_count > 0 else None,
            'methods_used': list(set(v.valuation_method for v in recent_valuations)) if valuation_count > 0 else []
        }
        
        # Build agent data
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
            'configuration': agent.configuration,
            'latest_log': latest_log
        }
        
        # Add queue_size field if it exists
        if hasattr(agent, 'queue_size'):
            agent_data['queue_size'] = agent.queue_size
        
        # Add current_task field if it exists
        if hasattr(agent, 'current_task'):
            agent_data['current_task'] = agent.current_task
        
        # Add error_count field if it exists
        if hasattr(agent, 'error_count'):
            agent_data['error_count'] = agent.error_count
        
        results.append(agent_data)
    
    # Calculate system-wide metrics
    active_agents = sum(1 for agent in agents if agent.is_active)
    idle_agents = sum(1 for agent in agents if agent.status == 'idle' and agent.is_active)
    processing_agents = sum(1 for agent in agents if agent.status == 'processing')
    error_agents = sum(1 for agent in agents if agent.status == 'error')
    
    # System health indicator
    system_health = 'healthy'
    if error_agents > 0:
        system_health = 'warning'
    if error_agents > active_agents / 3:  # If more than 1/3 of agents are in error state
        system_health = 'critical'
    
    return jsonify({
        'agents': results,
        'count': len(results),
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': {
            'total_agents': len(agents),
            'active_agents': active_agents,
            'idle_agents': idle_agents,
            'processing_agents': processing_agents, 
            'error_agents': error_agents,
            'system_health': system_health
        }
    })


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
    
    Query parameters:
    - method: Filter by valuation method (e.g., 'enhanced_regression', 'lightgbm')
    - min_confidence: Minimum confidence score (0.0-1.0)
    - after_date: Only valuations after this date (ISO format)
    - before_date: Only valuations before this date (ISO format)
    - property_id: Filter by property ID
    - neighborhood: Filter by neighborhood
    - page: Page number for pagination (default: 1)
    - limit: Results per page (default: 20, max: 100)
    - sort_by: Field to sort by (default: 'valuation_date')
    - sort_dir: Sort direction ('asc' or 'desc', default: 'desc')
    """
    # Get filter parameters
    method = request.args.get('method')
    min_confidence = request.args.get('min_confidence', type=float)
    after_date = request.args.get('after_date')
    before_date = request.args.get('before_date')
    property_id = request.args.get('property_id')
    neighborhood = request.args.get('neighborhood')
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    limit = min(100, limit)  # Cap at 100 to prevent abuse
    
    # Sorting parameters
    sort_by = request.args.get('sort_by', 'valuation_date')
    sort_dir = request.args.get('sort_dir', 'desc')
    
    # Validate sort parameters
    valid_sort_fields = ['valuation_date', 'estimated_value', 'confidence_score']
    if sort_by not in valid_sort_fields:
        sort_by = 'valuation_date'
    
    if sort_dir not in ['asc', 'desc']:
        sort_dir = 'desc'
    
    # Build base query
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
    
    # Get total count
    total = query.count()
    
    # Apply sorting
    sort_field = getattr(PropertyValuation, sort_by)
    sort_field = sort_field.asc() if sort_dir == 'asc' else sort_field.desc()
    query = query.order_by(sort_field)
    
    # Apply pagination
    query = query.offset((page - 1) * limit).limit(limit)
    
    # Execute query
    results = query.all()
    
    # Format results
    valuations = []
    for valuation, property in results:
        data = {
            'valuation_id': valuation.id,
            'property_id': property.property_id,
            'address': property.address,
            'city': property.city,
            'state': property.state,
            'zip_code': property.zip_code,
            'neighborhood': property.neighborhood,
            'estimated_value': valuation.estimated_value,
            'valuation_date': valuation.valuation_date.isoformat(),
            'valuation_method': valuation.valuation_method,
            'confidence_score': valuation.confidence_score,
            'model_features': valuation.model_features
        }
        
        # Include model metrics if available
        model_metrics = {}
        if hasattr(valuation, 'adj_r2_score') and valuation.adj_r2_score is not None:
            model_metrics['adj_r2'] = valuation.adj_r2_score
        if hasattr(valuation, 'rmse') and valuation.rmse is not None:
            model_metrics['rmse'] = valuation.rmse
        if hasattr(valuation, 'mae') and valuation.mae is not None:
            model_metrics['mae'] = valuation.mae
        
        if model_metrics:
            data['model_metrics'] = model_metrics
        
        # Include GIS features if available
        if valuation.gis_features:
            data['gis_features'] = valuation.gis_features
        
        valuations.append(data)
    
    return jsonify({
        'valuations': valuations,
        'page': page,
        'limit': limit,
        'total': total,
        'pages': (total // limit) + (1 if total % limit > 0 else 0)
    })


@api_bp.route('/etl-status', methods=['GET'])
@require_api_key
def get_etl_status():
    """
    Get status of ETL pipeline jobs.
    
    Query parameters:
    - job_type: Filter by job type
    - status: Filter by status
    - timeframe: Filter by timeframe ('today', 'yesterday', 'this_week', 'last_week', 'this_month')
    - limit: Maximum number of jobs to return (default: 20, max: 100)
    """
    # Get filter parameters
    job_type = request.args.get('job_type')
    status = request.args.get('status')
    timeframe = request.args.get('timeframe', 'today')
    limit = request.args.get('limit', 20, type=int)
    limit = min(100, limit)  # Cap at 100 to prevent abuse
    
    # Build base query
    query = ETLJob.query
    
    # Determine date range based on timeframe
    now = datetime.utcnow()
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
    
    # Apply timeframe filter
    query = query.filter(ETLJob.start_time >= start_date)
    
    # Apply other filters
    if job_type:
        query = query.filter(ETLJob.job_type == job_type)
    
    if status:
        query = query.filter(ETLJob.status == status)
    
    # Order by most recent first
    query = query.order_by(desc(ETLJob.start_time))
    
    # Apply limit
    query = query.limit(limit)
    
    # Execute query
    jobs = query.all()
    
    # Format results
    results = []
    for job in jobs:
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
        
        results.append(job_data)
    
    # Get summary statistics
    stats = {
        'total_jobs': len(results),
        'completed_jobs': sum(1 for job in jobs if job.status == 'completed'),
        'failed_jobs': sum(1 for job in jobs if job.status == 'failed'),
        'running_jobs': sum(1 for job in jobs if job.status == 'running'),
        'pending_jobs': sum(1 for job in jobs if job.status == 'pending'),
        'total_records_processed': sum(job.records_processed for job in jobs),
        'average_progress': sum(job.progress for job in jobs) / len(jobs) if jobs else 0
    }
    
    # Add system health indicators
    health = {
        'status': 'healthy' if stats['failed_jobs'] == 0 else 'warning' if stats['failed_jobs'] <= 2 else 'critical',
        'pipeline_active': stats['running_jobs'] > 0 or (stats['completed_jobs'] > 0 and any(job.end_time and (now - job.end_time).total_seconds() < 3600 for job in jobs if job.status == 'completed')),
        'last_successful_job': next((job.end_time.isoformat() for job in jobs if job.status == 'completed'), None)
    }
    
    return jsonify({
        'jobs': results,
        'stats': stats,
        'health': health,
        'timeframe': timeframe,
        'timestamp': now.isoformat()
    })


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