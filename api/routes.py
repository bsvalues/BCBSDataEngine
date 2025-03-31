"""
API routes for the BCBS Values application.

This module defines API endpoints for our web application.
"""

import logging
from datetime import datetime
from functools import wraps

from flask import Blueprint, request, jsonify, current_app, make_response
from app import app, db
from models import Property, ValuationRecord, ETLJobStatus, AgentStatus

# Set up logging
logger = logging.getLogger(__name__)

# API key verification
def require_api_key(f):
    """
    Decorator to require API key for access.
    
    Args:
        f: The function to protect
    
    Returns:
        function: Decorated function
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get API key from environment
        api_key = current_app.config.get('API_KEY')
        
        # Check if API key is provided in request header
        provided_key = request.headers.get('X-API-Key')
        
        if not api_key or provided_key != api_key:
            return jsonify({'error': 'Unauthorized access. Valid API key required.'}), 401
            
        return f(*args, **kwargs)
    return decorated


# API status endpoint
@app.route('/api/status')
def api_status():
    """Return API status."""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


# Stats endpoint
@app.route('/api/stats')
def stats():
    """Return quick statistics about properties and valuations."""
    try:
        # Get property and valuation statistics
        total_properties = db.session.query(Property).count()
        
        # Calculate average valuation
        avg_valuation_result = db.session.query(
            db.func.avg(Property.estimated_value)
        ).scalar()
        avg_valuation = float(avg_valuation_result) if avg_valuation_result else 0
        
        # Get recent valuations (last 30 days)
        recent_valuations = db.session.query(ValuationRecord).filter(
            ValuationRecord.created_at >= datetime.utcnow() - db.func.interval('30 days')
        ).count()
        
        # Get unique neighborhoods
        neighborhoods = db.session.query(
            db.func.count(db.func.distinct(Property.neighborhood))
        ).scalar() or 0
        
        return jsonify({
            'total_properties': total_properties,
            'avg_valuation': avg_valuation,
            'recent_valuations': recent_valuations,
            'neighborhoods': neighborhoods
        })
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'details': str(e)
        }), 500


# Properties endpoint
@app.route('/api/properties')
def get_properties():
    """
    Get all properties with filtering options.
    
    Query Parameters:
        page (int): Page number for pagination
        per_page (int): Items per page
        neighborhood (str): Filter by neighborhood
        property_type (str): Filter by property type
        min_price (float): Minimum price/valuation
        max_price (float): Maximum price/valuation
    """
    try:
        # Pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Build query with filters
        query = db.session.query(Property)
        
        # Apply filters if provided
        if request.args.get('neighborhood'):
            query = query.filter(Property.neighborhood == request.args.get('neighborhood'))
            
        if request.args.get('property_type'):
            query = query.filter(Property.property_type == request.args.get('property_type'))
            
        if request.args.get('min_price', type=float):
            query = query.filter(Property.estimated_value >= request.args.get('min_price', type=float))
            
        if request.args.get('max_price', type=float):
            query = query.filter(Property.estimated_value <= request.args.get('max_price', type=float))
            
        # Get total count for pagination
        total = query.count()
        
        # Apply pagination
        properties = query.order_by(Property.estimated_value.desc()) \
                          .offset((page - 1) * per_page) \
                          .limit(per_page) \
                          .all()
        
        # Format properties for JSON response
        properties_data = [{
            'id': p.id,
            'property_id': p.property_id,
            'address': p.address,
            'city': p.city,
            'state': p.state,
            'zip_code': p.zip_code,
            'latitude': p.latitude,
            'longitude': p.longitude,
            'bedrooms': p.bedrooms,
            'bathrooms': p.bathrooms,
            'square_feet': p.square_feet,
            'year_built': p.year_built,
            'property_type': p.property_type,
            'neighborhood': p.neighborhood,
            'estimated_value': p.estimated_value,
            'valuation_date': p.valuation_date.isoformat() if p.valuation_date else None,
            'last_sale_price': p.last_sale_price,
            'last_sale_date': p.last_sale_date.isoformat() if p.last_sale_date else None
        } for p in properties]
        
        return jsonify({
            'properties': properties_data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error retrieving properties: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve properties',
            'details': str(e)
        }), 500


# Property detail endpoint
@app.route('/api/properties/<string:property_id>')
def get_property(property_id):
    """
    Get details for a specific property.
    
    Args:
        property_id (str): The property ID to retrieve
    """
    try:
        # First try to find by ID
        try:
            property_int_id = int(property_id)
            property = Property.query.get(property_int_id)
        except ValueError:
            # If not an integer, try to find by property_id field
            property = Property.query.filter_by(property_id=property_id).first()
        
        if not property:
            return jsonify({'error': 'Property not found'}), 404
            
        # Get the property's latest valuation
        latest_valuation = ValuationRecord.query.filter_by(property_id=property.id) \
                                           .order_by(ValuationRecord.valuation_date.desc()) \
                                           .first()
        
        # Format property for JSON response
        property_data = {
            'id': property.id,
            'property_id': property.property_id,
            'address': property.address,
            'city': property.city,
            'state': property.state,
            'zip_code': property.zip_code,
            'latitude': property.latitude,
            'longitude': property.longitude,
            'bedrooms': property.bedrooms,
            'bathrooms': property.bathrooms,
            'square_feet': property.square_feet,
            'lot_size': property.lot_size,
            'year_built': property.year_built,
            'property_type': property.property_type,
            'neighborhood': property.neighborhood,
            'estimated_value': property.estimated_value,
            'valuation_date': property.valuation_date.isoformat() if property.valuation_date else None,
            'last_sale_price': property.last_sale_price,
            'last_sale_date': property.last_sale_date.isoformat() if property.last_sale_date else None,
            'latest_valuation': {
                'estimated_value': latest_valuation.estimated_value if latest_valuation else None,
                'valuation_date': latest_valuation.valuation_date.isoformat() if latest_valuation else None,
                'valuation_method': latest_valuation.valuation_method if latest_valuation else None,
                'confidence_score': latest_valuation.confidence_score if latest_valuation else None
            } if latest_valuation else None
        }
        
        return jsonify(property_data)
    except Exception as e:
        logger.error(f"Error retrieving property {property_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve property',
            'details': str(e)
        }), 500


# Property valuations endpoint
@app.route('/api/properties/<string:property_id>/valuations')
def get_property_valuations(property_id):
    """
    Get valuation history for a specific property.
    
    Args:
        property_id (str): The property ID to retrieve valuations for
    """
    try:
        # First try to find by ID
        try:
            property_int_id = int(property_id)
            property = Property.query.get(property_int_id)
        except ValueError:
            # If not an integer, try to find by property_id field
            property = Property.query.filter_by(property_id=property_id).first()
        
        if not property:
            return jsonify({'error': 'Property not found'}), 404
            
        # Get the property's valuation history
        valuations = ValuationRecord.query.filter_by(property_id=property.id) \
                                    .order_by(ValuationRecord.valuation_date.desc()) \
                                    .all()
        
        # Format valuations for JSON response
        valuations_data = [{
            'id': v.id,
            'valuation_date': v.valuation_date.isoformat(),
            'estimated_value': v.estimated_value,
            'valuation_method': v.valuation_method,
            'confidence_score': v.confidence_score,
            'created_at': v.created_at.isoformat()
        } for v in valuations]
        
        return jsonify({
            'property_id': property.id,
            'valuations': valuations_data
        })
    except Exception as e:
        logger.error(f"Error retrieving valuations for property {property_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve valuations',
            'details': str(e)
        }), 500


# Property search endpoint
@app.route('/api/properties/search')
def search_properties():
    """
    Search for properties by text query.
    
    Query Parameters:
        q (str): Search query (will search address, city, neighborhood)
    """
    try:
        query = request.args.get('q', '')
        
        if not query:
            return jsonify({
                'properties': [],
                'count': 0
            })
            
        # Create search filter for address, city, neighborhood
        search_filter = db.or_(
            Property.address.ilike(f'%{query}%'),
            Property.city.ilike(f'%{query}%'),
            Property.neighborhood.ilike(f'%{query}%'),
            Property.property_id.ilike(f'%{query}%')
        )
        
        # Get properties matching the search
        properties = Property.query.filter(search_filter).limit(20).all()
        
        # Format properties for JSON response
        properties_data = [{
            'id': p.id,
            'property_id': p.property_id,
            'address': p.address,
            'city': p.city,
            'state': p.state,
            'zip_code': p.zip_code,
            'bedrooms': p.bedrooms,
            'bathrooms': p.bathrooms,
            'square_feet': p.square_feet,
            'property_type': p.property_type,
            'neighborhood': p.neighborhood,
            'estimated_value': p.estimated_value
        } for p in properties]
        
        return jsonify({
            'properties': properties_data,
            'count': len(properties_data)
        })
    except Exception as e:
        logger.error(f"Error searching properties with query '{query}': {str(e)}")
        return jsonify({
            'error': 'Failed to search properties',
            'details': str(e)
        }), 500


# ETL status endpoint
@app.route('/api/etl-status')
@require_api_key
def get_etl_status():
    """Get the status of ETL jobs."""
    try:
        # Get the latest ETL job statuses
        etl_jobs = ETLJobStatus.query.order_by(ETLJobStatus.start_time.desc()).limit(10).all()
        
        # Format jobs for JSON response
        jobs_data = [{
            'id': job.id,
            'job_name': job.job_name,
            'start_time': job.start_time.isoformat(),
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'status': job.status,
            'records_processed': job.records_processed,
            'error_message': job.error_message
        } for job in etl_jobs]
        
        return jsonify({
            'etl_jobs': jobs_data,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving ETL job status: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve ETL job status',
            'details': str(e)
        }), 500


# Agent status endpoint
@app.route('/api/agent-status')
@require_api_key
def get_agent_status():
    """Get the status of agents."""
    try:
        # Get all agent statuses
        agents = AgentStatus.query.all()
        
        # Format agents for JSON response
        agents_data = [{
            'id': agent.id,
            'agent_id': agent.agent_id,
            'agent_name': agent.agent_name,
            'status': agent.status,
            'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            'current_task': agent.current_task,
            'queue_size': agent.queue_size,
            'success_rate': agent.success_rate,
            'error_count': agent.error_count
        } for agent in agents]
        
        return jsonify({
            'agents': agents_data,
            'count': len(agents_data),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving agent status: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve agent status',
            'details': str(e)
        }), 500


# Neighborhoods endpoint
@app.route('/api/neighborhoods')
def get_neighborhoods():
    """Get a list of unique neighborhoods."""
    try:
        # Get distinct neighborhoods
        neighborhoods = db.session.query(Property.neighborhood) \
                               .filter(Property.neighborhood.isnot(None)) \
                               .distinct() \
                               .order_by(Property.neighborhood) \
                               .all()
        
        # Extract neighborhood names from result tuples
        neighborhood_list = [n[0] for n in neighborhoods if n[0]]
        
        return jsonify({
            'neighborhoods': neighborhood_list,
            'count': len(neighborhood_list)
        })
    except Exception as e:
        logger.error(f"Error retrieving neighborhoods: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve neighborhoods',
            'details': str(e)
        }), 500


# Property types endpoint
@app.route('/api/property-types')
def get_property_types():
    """Get a list of unique property types."""
    try:
        # Get distinct property types
        property_types = db.session.query(Property.property_type) \
                                .filter(Property.property_type.isnot(None)) \
                                .distinct() \
                                .order_by(Property.property_type) \
                                .all()
        
        # Extract property type names from result tuples
        property_type_list = [pt[0] for pt in property_types if pt[0]]
        
        return jsonify({
            'property_types': property_type_list,
            'count': len(property_type_list)
        })
    except Exception as e:
        logger.error(f"Error retrieving property types: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve property types',
            'details': str(e)
        }), 500