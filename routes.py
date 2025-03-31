"""
BCBS Values - Route definitions
"""

import os
import logging
from datetime import datetime
from flask import render_template, redirect, url_for, flash, request, abort, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user

from app import db
from models import User, Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
from forms import PropertyValuationForm, PropertySearchForm

logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all routes with the Flask application."""
    
    @app.route('/')
    def index():
        """Render the home page."""
        return render_template('index.html')
    
    @app.route('/valuation', methods=['GET', 'POST'])
    def valuation():
        """Render the property valuation form and handle submissions."""
        form = PropertyValuationForm()
        
        if form.validate_on_submit():
            # Create a new property object
            property_obj = Property(
                address=form.address.data,
                city=form.city.data,
                state=form.state.data,
                zip_code=form.zip_code.data,
                neighborhood=form.neighborhood.data,
                property_type=form.property_type.data,
                bedrooms=form.bedrooms.data,
                bathrooms=form.bathrooms.data,
                square_feet=form.square_feet.data,
                year_built=form.year_built.data,
                lot_size=form.lot_size.data,
                latitude=form.latitude.data,
                longitude=form.longitude.data,
                user_id=current_user.id if not current_user.is_anonymous else None
            )
            
            db.session.add(property_obj)
            db.session.commit()
            
            # Now perform valuation
            try:
                # Import here to avoid circular imports
                from src.valuation import perform_valuation
                
                result = perform_valuation(
                    property_obj, 
                    valuation_method=form.valuation_method.data
                )
                
                # Create a valuation record
                valuation_record = PropertyValuation(
                    property_id=property_obj.id,
                    user_id=current_user.id if not current_user.is_anonymous else None,
                    estimated_value=result['estimated_value'],
                    valuation_method=form.valuation_method.data,
                    confidence_score=result.get('confidence_score', 0.7),
                    model_features=result.get('model_features'),
                    comparable_properties=result.get('comparable_properties'),
                    market_trends=result.get('market_trends'),
                    gis_features=result.get('gis_features')
                )
                
                db.session.add(valuation_record)
                db.session.commit()
                
                return redirect(url_for('valuation_result', valuation_id=valuation_record.id))
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Valuation error: {e}")
                flash(f"Error during valuation: {str(e)}", "danger")
                return render_template('valuation_form.html', form=form)
        
        return render_template('valuation_form.html', form=form)
    
    @app.route('/valuation/result/<int:valuation_id>')
    def valuation_result(valuation_id):
        """Display property valuation results."""
        valuation = PropertyValuation.query.get_or_404(valuation_id)
        property_obj = Property.query.get_or_404(valuation.property_id)
        
        return render_template(
            'valuation_result.html',
            valuation=valuation,
            property=property_obj,
            result=valuation
        )
    
    @app.route('/property/<int:property_id>')
    def property_detail(property_id):
        """Display detailed property information."""
        property_obj = Property.query.get_or_404(property_id)
        
        # Get the latest valuation for this property
        latest_valuation = PropertyValuation.query \
            .filter_by(property_id=property_id) \
            .order_by(PropertyValuation.valuation_date.desc()) \
            .first()
        
        # Get property features
        features = PropertyFeature.query.filter_by(property_id=property_id).all()
        
        return render_template(
            'property_detail.html',
            property=property_obj,
            valuation=latest_valuation,
            features=features
        )
    
    @app.route('/properties', methods=['GET', 'POST'])
    def properties():
        """Display a searchable and filterable list of properties."""
        form = PropertySearchForm()
        
        # Populate neighborhood choices
        neighborhoods = db.session.query(Property.neighborhood) \
            .filter(Property.neighborhood.isnot(None)) \
            .distinct() \
            .order_by(Property.neighborhood)
        
        form.neighborhood.choices = [('', 'All')] + [(n[0], n[0]) for n in neighborhoods]
        
        # Get query parameters
        query = request.args.get('search_query', '')
        neighborhood = request.args.get('neighborhood', '')
        property_type = request.args.get('property_type', '')
        min_bedrooms = request.args.get('min_bedrooms', '')
        min_price = request.args.get('min_price', '')
        max_price = request.args.get('max_price', '')
        
        # Build query
        properties_query = Property.query
        
        if query:
            properties_query = properties_query.filter(
                Property.address.ilike(f'%{query}%') | 
                Property.city.ilike(f'%{query}%') |
                Property.neighborhood.ilike(f'%{query}%')
            )
        
        if neighborhood:
            properties_query = properties_query.filter(Property.neighborhood == neighborhood)
        
        if property_type:
            properties_query = properties_query.filter(Property.property_type == property_type)
        
        if min_bedrooms and min_bedrooms.isdigit():
            properties_query = properties_query.filter(Property.bedrooms >= int(min_bedrooms))
        
        # For price filtering, we need to join with the PropertyValuation table
        if min_price or max_price:
            # Subquery to get the latest valuation for each property
            from sqlalchemy import func
            latest_valuations = db.session.query(
                PropertyValuation.property_id,
                func.max(PropertyValuation.valuation_date).label('max_date')
            ).group_by(PropertyValuation.property_id).subquery()
            
            properties_query = properties_query.join(
                latest_valuations,
                Property.id == latest_valuations.c.property_id
            ).join(
                PropertyValuation,
                db.and_(
                    PropertyValuation.property_id == latest_valuations.c.property_id,
                    PropertyValuation.valuation_date == latest_valuations.c.max_date
                )
            )
            
            if min_price and min_price.replace('.', '').isdigit():
                properties_query = properties_query.filter(PropertyValuation.estimated_value >= float(min_price))
            
            if max_price and max_price.replace('.', '').isdigit():
                properties_query = properties_query.filter(PropertyValuation.estimated_value <= float(max_price))
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        properties_paginated = properties_query.order_by(Property.updated_at.desc()).paginate(
            page=page, 
            per_page=per_page,
            error_out=False
        )
        
        # Get the latest valuation for each property
        property_valuations = {}
        for prop in properties_paginated.items:
            latest_val = PropertyValuation.query \
                .filter_by(property_id=prop.id) \
                .order_by(PropertyValuation.valuation_date.desc()) \
                .first()
            
            if latest_val:
                property_valuations[prop.id] = latest_val
        
        return render_template(
            'properties.html',
            properties=properties_paginated,
            valuations=property_valuations,
            form=form,
            query=query,
            neighborhood=neighborhood,
            property_type=property_type,
            min_bedrooms=min_bedrooms,
            min_price=min_price,
            max_price=max_price
        )
    
    @app.route('/dashboard')
    def dashboard():
        """Display property valuation dashboard."""
        # Get the most recent properties with valuations
        recent_properties = db.session.query(
            Property, PropertyValuation
        ).join(
            PropertyValuation, Property.id == PropertyValuation.property_id
        ).order_by(
            PropertyValuation.valuation_date.desc()
        ).limit(5).all()
        
        # Get ETL job status
        etl_jobs = ETLJob.query.order_by(ETLJob.start_time.desc()).limit(3).all()
        
        # Get agent status
        agents = Agent.query.all()
        
        return render_template(
            'dashboard.html',
            recent_properties=recent_properties,
            etl_jobs=etl_jobs,
            agents=agents
        )
    
    @app.route('/what-if-analysis/<int:property_id>')
    def what_if_analysis(property_id):
        """Display what-if analysis for property valuation."""
        property_obj = Property.query.get_or_404(property_id)
        
        # Get the latest valuation for this property
        latest_valuation = PropertyValuation.query \
            .filter_by(property_id=property_id) \
            .order_by(PropertyValuation.valuation_date.desc()) \
            .first()
        
        if not latest_valuation:
            flash("No valuation record found for this property.", "warning")
            return redirect(url_for('property_detail', property_id=property_id))
        
        return render_template(
            'what_if_analysis.html',
            property=property_obj,
            valuation=latest_valuation
        )
    
    # API routes
    @app.route('/api/property/<int:property_id>')
    def api_property(property_id):
        """API endpoint to get property details."""
        property_obj = Property.query.get_or_404(property_id)
        
        # Convert to dictionary
        data = {
            'id': property_obj.id,
            'address': property_obj.address,
            'city': property_obj.city,
            'state': property_obj.state,
            'zip_code': property_obj.zip_code,
            'neighborhood': property_obj.neighborhood,
            'property_type': property_obj.property_type,
            'bedrooms': property_obj.bedrooms,
            'bathrooms': property_obj.bathrooms,
            'square_feet': property_obj.square_feet,
            'year_built': property_obj.year_built,
            'lot_size': property_obj.lot_size,
            'latitude': property_obj.latitude,
            'longitude': property_obj.longitude,
            'last_sale_price': property_obj.last_sale_price,
            'last_sale_date': property_obj.last_sale_date.isoformat() if property_obj.last_sale_date else None,
            'created_at': property_obj.created_at.isoformat(),
            'updated_at': property_obj.updated_at.isoformat()
        }
        
        return jsonify(data)
    
    @app.route('/api/property/<int:property_id>/valuation')
    def api_property_valuation(property_id):
        """API endpoint to get property valuation details."""
        property_obj = Property.query.get_or_404(property_id)
        
        # Get the latest valuation
        valuation = PropertyValuation.query \
            .filter_by(property_id=property_id) \
            .order_by(PropertyValuation.valuation_date.desc()) \
            .first()
        
        if not valuation:
            return jsonify({'error': 'No valuation found for this property'}), 404
        
        # Convert to dictionary
        data = {
            'id': valuation.id,
            'property_id': valuation.property_id,
            'estimated_value': valuation.estimated_value,
            'valuation_date': valuation.valuation_date.isoformat(),
            'valuation_method': valuation.valuation_method,
            'confidence_score': valuation.confidence_score,
            'model_features': valuation.model_features,
            'comparable_properties': valuation.comparable_properties,
            'market_trends': valuation.market_trends,
            'gis_features': valuation.gis_features
        }
        
        return jsonify(data)
    
    @app.route('/api/what-if-analysis', methods=['POST'])
    def api_what_if_analysis():
        """API endpoint to perform what-if analysis."""
        data = request.json
        
        if not data or 'property_id' not in data:
            return jsonify({'error': 'Missing property_id in request data'}), 400
        
        property_id = data.get('property_id')
        property_obj = Property.query.get_or_404(property_id)
        
        # Get the parameters for the what-if analysis
        params = data.get('params', {})
        
        try:
            # Import here to avoid circular imports
            from src.valuation import perform_what_if_analysis
            
            result = perform_what_if_analysis(property_obj, params)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"What-if analysis error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/neighborhoods')
    def api_neighborhoods():
        """API endpoint to get a list of neighborhoods."""
        neighborhoods = db.session.query(Property.neighborhood) \
            .filter(Property.neighborhood.isnot(None)) \
            .distinct() \
            .order_by(Property.neighborhood) \
            .all()
        
        return jsonify([n[0] for n in neighborhoods])
    
    @app.route('/api/market-trends')
    def api_market_trends():
        """API endpoint to get market trends."""
        # This would typically involve aggregating data from the database
        # For demonstration purposes, we'll return some sample data
        try:
            from src.valuation import get_market_trends
            trends = get_market_trends()
            return jsonify(trends)
        except Exception as e:
            logger.error(f"Error fetching market trends: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/agent-status')
    def api_agent_status():
        """API endpoint to get agent status."""
        agents = Agent.query.all()
        
        data = []
        for agent in agents:
            # Get the latest log entry
            latest_log = AgentLog.query \
                .filter_by(agent_id=agent.id) \
                .order_by(AgentLog.timestamp.desc()) \
                .first()
                
            agent_data = {
                'id': agent.id,
                'name': agent.name,
                'type': agent.agent_type,
                'status': agent.status,
                'is_active': agent.is_active,
                'last_active': agent.last_active.isoformat(),
                'version': agent.version,
                'success_rate': agent.success_rate,
                'latest_log': {
                    'timestamp': latest_log.timestamp.isoformat() if latest_log else None,
                    'level': latest_log.level if latest_log else None,
                    'message': latest_log.message if latest_log else None
                } if latest_log else None
            }
            
            data.append(agent_data)
        
        return jsonify(data)
    
    @app.route('/api/agent-logs/<int:agent_id>')
    def api_agent_logs(agent_id):
        """API endpoint to get logs for a specific agent."""
        agent = Agent.query.get_or_404(agent_id)
        
        logs = AgentLog.query \
            .filter_by(agent_id=agent_id) \
            .order_by(AgentLog.timestamp.desc()) \
            .limit(100) \
            .all()
            
        data = []
        for log in logs:
            log_data = {
                'id': log.id,
                'timestamp': log.timestamp.isoformat(),
                'level': log.level,
                'message': log.message,
                'details': log.details
            }
            
            data.append(log_data)
            
        return jsonify(data)