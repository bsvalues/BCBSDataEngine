"""
API endpoints for the BCBS Values application.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Union

import pandas as pd
from flask import Blueprint, Flask, jsonify, request, current_app, g
from sqlalchemy import desc, func

from app import app, db
from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
from src.valuation import ValuationEngine, AdvancedValuationEngine, EnhancedGISValuationEngine
from src.enhanced_gis_features import load_gis_features, calculate_gis_adjustments


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


# API key authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key:
            return jsonify({'error': 'API key is required'}), 401
        
        # In a real system, we would validate the API key against a database
        # For now, we'll use a simple check against an environment variable
        valid_api_key = os.environ.get('API_KEY', 'bcbs_demo_key_2023')
        
        if api_key != valid_api_key:
            return jsonify({'error': 'Invalid API key'}), 401
        
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
    """Get status of valuation agents."""
    agents = Agent.query.all()
    
    # Format results
    results = []
    for agent in agents:
        latest_log = None
        if agent.logs:
            latest_log = {
                'level': agent.logs[0].level,
                'message': agent.logs[0].message,
                'timestamp': agent.logs[0].timestamp.isoformat()
            }
        
        agent_data = {
            'agent_id': agent.agent_id,
            'agent_name': agent.agent_name,
            'agent_type': agent.agent_type,
            'status': agent.status,
            'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            'current_task': agent.current_task,
            'queue_size': agent.queue_size,
            'success_rate': agent.success_rate,
            'error_count': agent.error_count,
            'latest_log': latest_log
        }
        results.append(agent_data)
    
    return jsonify({
        'agents': results,
        'count': len(results),
        'timestamp': datetime.utcnow().isoformat()
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
        'agent_name': agent.agent_name,
        'logs': results,
        'count': len(results)
    })


# Register the blueprint with the app
app.register_blueprint(api_bp)

# Run the app if executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)