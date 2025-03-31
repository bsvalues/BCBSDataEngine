"""
Routes for the BCBS Values application.

This module defines the routes for the application.
"""
from flask import render_template, jsonify, request, url_for, redirect, flash
from app import app, db
from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
from replit_auth import replit_user, require_login


@app.route('/')
def index():
    """Homepage route."""
    user = replit_user()
    return render_template('index.html', user=user)


@app.route('/dashboard')
@require_login
def dashboard():
    """Dashboard route - shows the property valuation dashboard."""
    user = replit_user()
    return render_template('dashboard.html', user=user)


@app.route('/profile')
@require_login
def profile():
    """User profile route."""
    user = replit_user()
    return render_template('profile.html', user=user)


# API routes
@app.route('/api/properties')
def api_properties():
    """API route to get properties with optional filtering."""
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 100)  # Limit to 100 for performance
    sort_by = request.args.get('sort_by', 'id')
    sort_dir = request.args.get('sort_dir', 'asc')
    
    # Filters
    search = request.args.get('search', '')
    neighborhood = request.args.get('neighborhood', '')
    property_type = request.args.get('property_type', '')
    min_price = request.args.get('min_price', None, type=float)
    max_price = request.args.get('max_price', None, type=float)
    bedrooms = request.args.get('bedrooms', None, type=int)
    updated_since = request.args.get('updated_since', None)
    
    # Build query
    query = Property.query
    
    # Apply filters
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Property.address.ilike(search_term)) |
            (Property.property_id.ilike(search_term)) |
            (Property.neighborhood.ilike(search_term))
        )
    
    if neighborhood:
        query = query.filter(Property.neighborhood == neighborhood)
    
    if property_type:
        query = query.filter(Property.property_type == property_type)
    
    if min_price:
        # Join to PropertyValuation to filter by price
        query = query.join(PropertyValuation).filter(PropertyValuation.estimated_value >= min_price)
    
    if max_price:
        # Join to PropertyValuation to filter by price if not already joined
        if not min_price:
            query = query.join(PropertyValuation)
        query = query.filter(PropertyValuation.estimated_value <= max_price)
    
    if bedrooms:
        query = query.filter(Property.bedrooms >= bedrooms)
    
    if updated_since:
        query = query.filter(Property.updated_at >= updated_since)
    
    # Apply sorting
    if sort_by == 'estimated_value':
        # Need to join with PropertyValuation for this sort
        if not (min_price or max_price):
            query = query.join(PropertyValuation)
        if sort_dir == 'desc':
            query = query.order_by(PropertyValuation.estimated_value.desc())
        else:
            query = query.order_by(PropertyValuation.estimated_value.asc())
    else:
        # Handle other sort columns
        sort_column = getattr(Property, sort_by, Property.id)
        if sort_dir == 'desc':
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
    
    # Paginate results
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Format the response
    result = {
        'properties': [],
        'total': paginated.total,
        'pages': paginated.pages,
        'page': page,
        'per_page': per_page
    }
    
    for prop in paginated.items:
        # Get the estimated value from the most recent valuation
        latest_valuation = prop.latest_valuation
        
        prop_data = {
            'id': prop.id,
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
            'lot_size': prop.lot_size,
            'estimated_value': prop.estimated_value,
            'valuation_date': prop.valuation_date.isoformat() if prop.valuation_date else None,
            'updated_at': prop.updated_at.isoformat()
        }
        result['properties'].append(prop_data)
    
    return jsonify(result)


@app.route('/api/properties/<int:property_id>')
def api_property_detail(property_id):
    """API route to get details for a specific property."""
    property = Property.query.get_or_404(property_id)
    
    # Get features
    features = {}
    for feature in property.features:
        features[feature.feature_name] = feature.feature_value
    
    # Format the response
    result = {
        'id': property.id,
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
        'estimated_value': property.estimated_value,
        'valuation_date': property.valuation_date.isoformat() if property.valuation_date else None,
        'latest_valuation': {
            'valuation_method': property.latest_valuation.valuation_method if property.latest_valuation else None,
            'confidence_score': property.latest_valuation.confidence_score if property.latest_valuation else None,
            'inputs': property.latest_valuation.inputs if property.latest_valuation else None
        } if property.latest_valuation else None,
        'features': features,
        'last_sale_date': property.last_sale_date.isoformat() if property.last_sale_date else None,
        'last_sale_price': property.last_sale_price,
        'created_at': property.created_at.isoformat(),
        'updated_at': property.updated_at.isoformat()
    }
    
    return jsonify(result)


@app.route('/api/properties/<int:property_id>/valuations')
def api_property_valuations(property_id):
    """API route to get valuation history for a specific property."""
    property = Property.query.get_or_404(property_id)
    
    # Format the response
    result = {
        'property_id': property.id,
        'valuations': []
    }
    
    for valuation in property.valuations:
        valuation_data = {
            'id': valuation.id,
            'estimated_value': valuation.estimated_value,
            'valuation_date': valuation.valuation_date.isoformat(),
            'valuation_method': valuation.valuation_method,
            'confidence_score': valuation.confidence_score,
            'created_at': valuation.created_at.isoformat()
        }
        result['valuations'].append(valuation_data)
    
    return jsonify(result)


@app.route('/api/neighborhoods')
def api_neighborhoods():
    """API route to get a list of neighborhoods."""
    neighborhoods = db.session.query(Property.neighborhood).distinct().filter(Property.neighborhood != None).all()
    neighborhoods = [n[0] for n in neighborhoods if n[0]]  # Extract from tuples and remove empty values
    
    return jsonify({'neighborhoods': neighborhoods})


@app.route('/api/property-types')
def api_property_types():
    """API route to get a list of property types."""
    property_types = db.session.query(Property.property_type).distinct().filter(Property.property_type != None).all()
    property_types = [pt[0] for pt in property_types if pt[0]]  # Extract from tuples and remove empty values
    
    return jsonify({'property_types': property_types})


@app.route('/api/etl-status')
def api_etl_status():
    """API route to get ETL job status."""
    # Get recent ETL jobs, ordered by start time descending
    jobs = ETLJob.query.order_by(ETLJob.start_time.desc()).limit(50).all()
    
    # Format the response
    result = {
        'etl_jobs': []
    }
    
    for job in jobs:
        job_data = {
            'id': job.id,
            'job_name': job.job_name,
            'job_type': job.job_type,
            'source': job.source,
            'start_time': job.start_time.isoformat(),
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'status': job.status,
            'records_processed': job.records_processed,
            'records_failed': job.records_failed,
            'error_message': job.error_message
        }
        result['etl_jobs'].append(job_data)
    
    return jsonify(result)


@app.route('/api/agent-status')
def api_agent_status():
    """API route to get agent status."""
    agents = Agent.query.all()
    
    # Format the response
    result = {
        'agents': []
    }
    
    for agent in agents:
        agent_data = {
            'id': agent.id,
            'agent_id': agent.agent_id,
            'agent_name': agent.agent_name,
            'agent_type': agent.agent_type,
            'status': agent.status,
            'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            'current_task': agent.current_task,
            'queue_size': agent.queue_size,
            'success_rate': agent.success_rate,
            'error_count': agent.error_count,
            'updated_at': agent.updated_at.isoformat()
        }
        result['agents'].append(agent_data)
    
    return jsonify(result)


@app.route('/api/agent-status/<string:agent_id>')
def api_agent_detail(agent_id):
    """API route to get details for a specific agent."""
    agent = Agent.query.filter_by(agent_id=agent_id).first_or_404()
    
    # Get recent logs
    logs = agent.logs[:50]  # Limit to 50 most recent logs
    
    # Format the response
    result = {
        'agent': {
            'id': agent.id,
            'agent_id': agent.agent_id,
            'agent_name': agent.agent_name,
            'agent_type': agent.agent_type,
            'status': agent.status,
            'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            'current_task': agent.current_task,
            'queue_size': agent.queue_size,
            'success_rate': agent.success_rate,
            'error_count': agent.error_count,
            'created_at': agent.created_at.isoformat(),
            'updated_at': agent.updated_at.isoformat()
        },
        'logs': []
    }
    
    for log in logs:
        log_data = {
            'id': log.id,
            'level': log.level,
            'message': log.message,
            'timestamp': log.timestamp.isoformat()
        }
        result['logs'].append(log_data)
    
    return jsonify(result)


@app.route('/api/agent-status/<string:agent_id>/restart', methods=['POST'])
def api_agent_restart(agent_id):
    """API route to restart an agent."""
    agent = Agent.query.filter_by(agent_id=agent_id).first_or_404()
    
    # Update agent status
    agent.status = 'idle'
    agent.current_task = None
    agent.queue_size = 0
    
    # Add a log entry
    log = AgentLog(
        agent_id=agent.id,
        level='info',
        message=f'Agent restarted via API call'
    )
    db.session.add(log)
    
    # Save changes
    db.session.commit()
    
    return jsonify({'success': True, 'message': f'Agent {agent.agent_name} restarted successfully'})


@app.route('/api/properties/export')
def api_export_properties():
    """API route to export property data in various formats."""
    format = request.args.get('format', 'csv')
    
    # This is a placeholder function - in a real implementation, 
    # we would generate the appropriate file format and send it as an attachment
    
    # For now, we'll just return a JSON response indicating that the export is not implemented
    return jsonify({
        'success': False,
        'message': f'Export to {format} is not implemented yet'
    }), 501


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(403)
def forbidden(e):
    """Handle 403 errors."""
    return render_template('403.html'), 403


@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500