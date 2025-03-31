"""
Routes for BCBS Property Valuation application.
Includes routes for user authentication, property management, valuations,
API endpoints, and dashboard views.
"""
import os
import logging
import json
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import render_template, flash, redirect, url_for, request, jsonify, abort, g
from flask_login import login_user, logout_user, current_user, login_required
from sqlalchemy import or_, and_, desc, func

from app import app, db
from models import User, Property, PropertyValuation, ApiKey, EtlStatus, DataSource, Agent, AgentLog
from forms import (
    LoginForm, RegistrationForm, PasswordResetRequestForm, PasswordResetForm,
    ApiKeyForm, PropertySearchForm, PropertyForm, ValuationForm, BatchValuationForm
)

# Configure logging
logger = logging.getLogger(__name__)

# Helper Functions
def require_api_key(view_function):
    """Decorator to require API key for access to API endpoints."""
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        # Get API key from header or request args
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if api_key is None:
            return jsonify({"error": "API key is required"}), 401
        
        # Look up the API key
        key = ApiKey.query.filter_by(key=api_key, is_active=True).first()
        
        if key is None:
            return jsonify({"error": "Invalid or inactive API key"}), 401
        
        # Update last used timestamp
        key.last_used = datetime.utcnow()
        db.session.commit()
        
        # Set the user for this request
        g.api_user = key.user
        
        return view_function(*args, **kwargs)
    return decorated_function

def format_currency(value):
    """Format a number as currency."""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format a number as percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"

# Route Handlers

# Home and Authentication Routes
@app.route('/')
def index():
    """Home page route."""
    return render_template('index.html', title='Home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user is None or not check_password_hash(user.password_hash, form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('dashboard')
            
        flash('Login successful', 'success')
        return redirect(next_page)
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    """User logout route."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', title='Register', form=form)

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password_request():
    """Password reset request route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        if user:
            # In a real application, send a password reset email here
            flash('Check your email for instructions to reset your password.', 'info')
        else:
            # Don't reveal that the user doesn't exist
            flash('Check your email for instructions to reset your password.', 'info')
        
        return redirect(url_for('login'))
    
    return render_template('reset_password_request.html', title='Reset Password', form=form)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Password reset with token route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    # In a real application, validate the token here
    # For demo purposes, just show the form
    
    form = PasswordResetForm()
    if form.validate_on_submit():
        # Update the user's password
        flash('Your password has been reset successfully.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', title='Reset Password', form=form)

# User Profile and Settings Routes
@app.route('/profile')
@login_required
def profile():
    """User profile route."""
    return render_template('profile.html', title='Profile')

@app.route('/api-keys', methods=['GET', 'POST'])
@login_required
def api_keys():
    """API key management route."""
    form = ApiKeyForm()
    
    if form.validate_on_submit():
        api_key = ApiKey(
            key=ApiKey.generate_key(),
            name=form.name.data,
            user_id=current_user.id,
            is_active=True
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        flash(f'API Key "{form.name.data}" created successfully.', 'success')
        return redirect(url_for('api_keys'))
    
    # Get all API keys for the current user
    keys = ApiKey.query.filter_by(user_id=current_user.id).order_by(ApiKey.created_at.desc()).all()
    
    return render_template('api_keys.html', title='API Keys', form=form, keys=keys)

@app.route('/api-keys/<int:key_id>/deactivate', methods=['POST'])
@login_required
def deactivate_api_key(key_id):
    """Deactivate an API key."""
    key = ApiKey.query.get_or_404(key_id)
    
    # Ensure the current user owns this key
    if key.user_id != current_user.id:
        flash('You do not have permission to deactivate this API key.', 'danger')
        return redirect(url_for('api_keys'))
    
    key.is_active = False
    db.session.commit()
    
    flash(f'API Key "{key.name}" has been deactivated.', 'info')
    return redirect(url_for('api_keys'))

# Dashboard Routes
@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard route."""
    # Get latest property valuations
    recent_valuations = (PropertyValuation.query
                          .join(Property)
                          .order_by(PropertyValuation.created_at.desc())
                          .limit(5)
                          .all())
    
    # Get agent status summary
    agent_status = db.session.query(
        Agent.status,
        func.count(Agent.id).label('count')
    ).group_by(Agent.status).all()
    
    agent_status_dict = {status: count for status, count in agent_status}
    
    # Get ETL status
    etl_status = EtlStatus.query.order_by(EtlStatus.last_update.desc()).first()
    
    # Get property count by type
    property_types = db.session.query(
        Property.property_type,
        func.count(Property.id).label('count')
    ).group_by(Property.property_type).all()
    
    property_type_dict = {pt: count for pt, count in property_types}
    
    # Get valuation method distribution
    valuation_methods = db.session.query(
        PropertyValuation.valuation_method,
        func.count(PropertyValuation.id).label('count')
    ).group_by(PropertyValuation.valuation_method).all()
    
    valuation_method_dict = {vm: count for vm, count in valuation_methods}
    
    # Get property count by neighborhood
    neighborhoods = db.session.query(
        Property.neighborhood,
        func.count(Property.id).label('count')
    ).filter(Property.neighborhood != None).group_by(Property.neighborhood).all()
    
    neighborhood_dict = {n: count for n, count in neighborhoods if n}
    
    # Prepare dashboard stats
    stats = {
        'total_properties': Property.query.count(),
        'total_valuations': PropertyValuation.query.count(),
        'active_agents': agent_status_dict.get('active', 0),
        'etl_completeness': etl_status.completeness if etl_status else 0,
        'property_types': property_type_dict,
        'valuation_methods': valuation_method_dict,
        'neighborhoods': neighborhood_dict,
        'agent_status': agent_status_dict
    }
    
    return render_template('dashboard.html', title='Dashboard', stats=stats, recent_valuations=recent_valuations)

# Property Routes
@app.route('/properties')
@login_required
def properties():
    """Property listing route."""
    form = PropertySearchForm()
    
    # Get search parameters from request args
    query = request.args.get('query', '')
    property_type = request.args.get('property_type', '')
    neighborhood = request.args.get('neighborhood', '')
    min_price = request.args.get('min_price', None)
    max_price = request.args.get('max_price', None)
    min_bedrooms = request.args.get('min_bedrooms', None)
    min_bathrooms = request.args.get('min_bathrooms', None)
    min_area = request.args.get('min_area', None)
    
    # Set form values from request args
    form.query.data = query
    form.property_type.data = property_type
    form.neighborhood.data = neighborhood
    form.min_price.data = min_price
    form.max_price.data = max_price
    form.min_bedrooms.data = min_bedrooms
    form.min_bathrooms.data = min_bathrooms
    form.min_area.data = min_area
    
    # Build the query
    property_query = Property.query
    
    # Apply filters
    if query:
        property_query = property_query.filter(
            or_(
                Property.property_id.ilike(f'%{query}%'),
                Property.address.ilike(f'%{query}%'),
                Property.neighborhood.ilike(f'%{query}%'),
                Property.city.ilike(f'%{query}%'),
                Property.zip_code.ilike(f'%{query}%')
            )
        )
    
    if property_type:
        property_query = property_query.filter(Property.property_type == property_type)
    
    if neighborhood:
        property_query = property_query.filter(Property.neighborhood == neighborhood)
    
    if min_bedrooms:
        property_query = property_query.filter(Property.bedrooms >= int(min_bedrooms))
    
    if min_bathrooms:
        property_query = property_query.filter(Property.bathrooms >= float(min_bathrooms))
    
    if min_area:
        property_query = property_query.filter(Property.living_area >= float(min_area))
    
    # Filter by price range using the latest valuation
    if min_price or max_price:
        # This is a complex query, we'll use a subquery to get the latest valuation for each property
        latest_valuations = db.session.query(
            PropertyValuation.property_id,
            func.max(PropertyValuation.valuation_date).label('latest_date')
        ).group_by(PropertyValuation.property_id).subquery()
        
        price_query = db.session.query(
            PropertyValuation.property_id,
            PropertyValuation.estimated_value
        ).join(
            latest_valuations,
            and_(
                PropertyValuation.property_id == latest_valuations.c.property_id,
                PropertyValuation.valuation_date == latest_valuations.c.latest_date
            )
        )
        
        if min_price:
            price_query = price_query.filter(PropertyValuation.estimated_value >= float(min_price))
        
        if max_price:
            price_query = price_query.filter(PropertyValuation.estimated_value <= float(max_price))
        
        # Get property IDs that match the price criteria
        property_ids = [p.property_id for p in price_query.all()]
        
        # Filter the main query by these property IDs
        property_query = property_query.filter(Property.id.in_(property_ids))
    
    # Paginate the results
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    pagination = property_query.order_by(Property.updated_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    properties = pagination.items
    
    # Get the latest valuation for each property
    property_ids = [p.id for p in properties]
    latest_valuations = {}
    
    if property_ids:
        # Subquery to get the latest valuation for each property
        latest_valuation_dates = db.session.query(
            PropertyValuation.property_id,
            func.max(PropertyValuation.valuation_date).label('latest_date')
        ).filter(PropertyValuation.property_id.in_(property_ids)).group_by(PropertyValuation.property_id).subquery()
        
        # Query to get the valuation objects
        valuations = db.session.query(PropertyValuation).join(
            latest_valuation_dates,
            and_(
                PropertyValuation.property_id == latest_valuation_dates.c.property_id,
                PropertyValuation.valuation_date == latest_valuation_dates.c.latest_date
            )
        ).all()
        
        # Create a dictionary of property_id to valuation
        latest_valuations = {v.property_id: v for v in valuations}
    
    # Get unique neighborhoods for the filter dropdown
    neighborhoods = db.session.query(Property.neighborhood).filter(
        Property.neighborhood != None
    ).distinct().order_by(Property.neighborhood).all()
    
    neighborhoods = [n[0] for n in neighborhoods if n[0]]  # Extract from tuples and remove None
    
    return render_template(
        'properties.html',
        title='Properties',
        properties=properties,
        pagination=pagination,
        form=form,
        latest_valuations=latest_valuations,
        neighborhoods=neighborhoods,
        format_currency=format_currency
    )

@app.route('/properties/<int:property_id>')
@login_required
def property_detail(property_id):
    """Property detail route."""
    property_obj = Property.query.get_or_404(property_id)
    
    # Get all valuations for this property, ordered by date
    valuations = PropertyValuation.query.filter_by(
        property_id=property_id
    ).order_by(PropertyValuation.valuation_date.desc()).all()
    
    # Create valuation form for new valuations
    valuation_form = ValuationForm()
    valuation_form.property_id.data = property_id
    
    return render_template(
        'property_detail.html',
        title=f'Property: {property_obj.address}',
        property=property_obj,
        valuations=valuations,
        valuation_form=valuation_form,
        format_currency=format_currency
    )

@app.route('/properties/new', methods=['GET', 'POST'])
@login_required
def new_property():
    """Create new property route."""
    form = PropertyForm()
    
    if form.validate_on_submit():
        property_obj = Property(
            property_id=form.property_id.data,
            address=form.address.data,
            city=form.city.data,
            state=form.state.data,
            zip_code=form.zip_code.data,
            neighborhood=form.neighborhood.data,
            property_type=form.property_type.data,
            year_built=form.year_built.data,
            bedrooms=form.bedrooms.data,
            bathrooms=form.bathrooms.data,
            living_area=form.living_area.data,
            land_area=form.land_area.data,
            latitude=form.latitude.data,
            longitude=form.longitude.data
        )
        
        db.session.add(property_obj)
        db.session.commit()
        
        flash(f'Property "{form.address.data}" created successfully.', 'success')
        return redirect(url_for('property_detail', property_id=property_obj.id))
    
    return render_template('property_form.html', title='New Property', form=form, is_new=True)

@app.route('/properties/<int:property_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_property(property_id):
    """Edit property route."""
    property_obj = Property.query.get_or_404(property_id)
    form = PropertyForm(obj=property_obj)
    
    # Store original property_id for validation
    form.original_property_id = property_obj.property_id
    
    if form.validate_on_submit():
        property_obj.property_id = form.property_id.data
        property_obj.address = form.address.data
        property_obj.city = form.city.data
        property_obj.state = form.state.data
        property_obj.zip_code = form.zip_code.data
        property_obj.neighborhood = form.neighborhood.data
        property_obj.property_type = form.property_type.data
        property_obj.year_built = form.year_built.data
        property_obj.bedrooms = form.bedrooms.data
        property_obj.bathrooms = form.bathrooms.data
        property_obj.living_area = form.living_area.data
        property_obj.land_area = form.land_area.data
        property_obj.latitude = form.latitude.data
        property_obj.longitude = form.longitude.data
        property_obj.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        flash(f'Property "{form.address.data}" updated successfully.', 'success')
        return redirect(url_for('property_detail', property_id=property_obj.id))
    
    return render_template('property_form.html', title='Edit Property', form=form, is_new=False)

# Valuation Routes
@app.route('/valuations/new', methods=['POST'])
@login_required
def new_valuation():
    """Create new valuation route."""
    form = ValuationForm()
    
    if form.validate_on_submit():
        property_id = form.property_id.data
        property_obj = Property.query.get_or_404(property_id)
        
        # For this demo, we'll generate a simple valuation
        # In a real app, this would call valuation algorithms
        
        # Base value calculation
        if property_obj.property_type == "land":
            # Land is valued by acre
            base_value = 100000 + (property_obj.land_area * 250000)
        else:
            # Calculate base value from square footage
            sq_ft_value = {
                "single_family": 300,
                "condo": 350,
                "townhouse": 325,
                "multi_family": 250
            }.get(property_obj.property_type, 300)
            
            base_value = property_obj.living_area * sq_ft_value
            
            # Adjust for bedrooms and bathrooms
            if property_obj.bedrooms:
                base_value += property_obj.bedrooms * 15000
            
            if property_obj.bathrooms:
                base_value += property_obj.bathrooms * 25000
            
            # Adjust for age if year_built is available
            if property_obj.year_built:
                age = datetime.now().year - property_obj.year_built
                age_factor = max(0.7, 1 - (age / 100))
                base_value *= age_factor
        
        # Adjust value based on method
        method = form.valuation_method.data
        if method == "enhanced_regression":
            value_adjustment = 1.05
            confidence = 0.92
        elif method == "lightgbm":
            value_adjustment = 1.03
            confidence = 0.90
        elif method == "xgboost":
            value_adjustment = 1.04
            confidence = 0.91
        elif method == "linear_regression":
            value_adjustment = 0.98
            confidence = 0.82
        elif method == "ridge_regression":
            value_adjustment = 0.99
            confidence = 0.84
        elif method == "lasso_regression":
            value_adjustment = 0.97
            confidence = 0.83
        else:  # elastic_net
            value_adjustment = 1.00
            confidence = 0.85
        
        # Create the valuation
        valuation = PropertyValuation(
            property_id=property_id,
            estimated_value=round(base_value * value_adjustment, 2),
            confidence_score=confidence,
            valuation_date=form.valuation_date.data,
            valuation_method=method
        )
        
        db.session.add(valuation)
        db.session.commit()
        
        flash(f'New valuation created: {format_currency(valuation.estimated_value)}', 'success')
        return redirect(url_for('property_detail', property_id=property_id))
    
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'Error in {getattr(form, field).label.text}: {error}', 'danger')
    
    return redirect(url_for('properties'))

@app.route('/batch-valuation', methods=['GET', 'POST'])
@login_required
def batch_valuation():
    """Batch valuation route."""
    form = BatchValuationForm()
    
    if form.validate_on_submit():
        # Get properties matching the criteria
        property_query = Property.query
        
        if form.property_type.data != 'all':
            property_query = property_query.filter(Property.property_type == form.property_type.data)
        
        if form.neighborhood.data:
            property_query = property_query.filter(Property.neighborhood == form.neighborhood.data)
        
        properties = property_query.all()
        
        if not properties:
            flash('No properties found matching the selected criteria.', 'warning')
            return redirect(url_for('batch_valuation'))
        
        # For each property, create a valuation
        valuations_created = 0
        method = form.valuation_method.data
        
        for property_obj in properties:
            # Check if a recent valuation (last 30 days) exists with this method
            recent_valuation = PropertyValuation.query.filter(
                PropertyValuation.property_id == property_obj.id,
                PropertyValuation.valuation_method == method,
                PropertyValuation.valuation_date >= (datetime.utcnow() - timedelta(days=30))
            ).first()
            
            if recent_valuation:
                continue  # Skip if recent valuation exists
            
            # Similar logic to single valuation, simplified for batch processing
            if property_obj.property_type == "land":
                base_value = 100000 + (property_obj.land_area * 250000 if property_obj.land_area else 0)
            else:
                sq_ft_value = {
                    "single_family": 300,
                    "condo": 350,
                    "townhouse": 325,
                    "multi_family": 250
                }.get(property_obj.property_type, 300)
                
                base_value = property_obj.living_area * sq_ft_value if property_obj.living_area else 200000
                
                if property_obj.bedrooms:
                    base_value += property_obj.bedrooms * 15000
                
                if property_obj.bathrooms:
                    base_value += property_obj.bathrooms * 25000
                
                if property_obj.year_built:
                    age = datetime.now().year - property_obj.year_built
                    age_factor = max(0.7, 1 - (age / 100))
                    base_value *= age_factor
            
            # Adjust based on method
            if method == "enhanced_regression":
                value_adjustment = 1.05
                confidence = 0.92
            elif method == "lightgbm":
                value_adjustment = 1.03
                confidence = 0.90
            elif method == "xgboost":
                value_adjustment = 1.04
                confidence = 0.91
            elif method == "linear_regression":
                value_adjustment = 0.98
                confidence = 0.82
            elif method == "ridge_regression":
                value_adjustment = 0.99
                confidence = 0.84
            elif method == "lasso_regression":
                value_adjustment = 0.97
                confidence = 0.83
            else:  # elastic_net
                value_adjustment = 1.00
                confidence = 0.85
            
            # Create the valuation
            valuation = PropertyValuation(
                property_id=property_obj.id,
                estimated_value=round(base_value * value_adjustment, 2),
                confidence_score=confidence,
                valuation_date=datetime.utcnow(),
                valuation_method=method
            )
            
            db.session.add(valuation)
            valuations_created += 1
        
        db.session.commit()
        
        flash(f'Batch valuation completed. Created {valuations_created} new valuations.', 'success')
        return redirect(url_for('dashboard'))
    
    # Get unique neighborhoods for the filter dropdown
    neighborhoods = db.session.query(Property.neighborhood).filter(
        Property.neighborhood != None
    ).distinct().order_by(Property.neighborhood).all()
    
    neighborhoods = [n[0] for n in neighborhoods if n[0]]  # Extract from tuples and remove None
    
    return render_template('batch_valuation.html', title='Batch Valuation', form=form, neighborhoods=neighborhoods)

# ETL and Agent Routes
@app.route('/etl-status')
@login_required
def etl_status():
    """ETL status route."""
    status = EtlStatus.query.order_by(EtlStatus.last_update.desc()).first()
    sources = []
    
    if status:
        sources = DataSource.query.filter_by(etl_status_id=status.id).all()
    
    return render_template('etl_status.html', title='ETL Status', status=status, sources=sources)

@app.route('/agents')
@login_required
def agents():
    """Agent status route."""
    agents = Agent.query.order_by(Agent.agent_type, Agent.agent_id).all()
    
    # Group agents by type
    agents_by_type = {}
    for agent in agents:
        if agent.agent_type not in agents_by_type:
            agents_by_type[agent.agent_type] = []
        agents_by_type[agent.agent_type].append(agent)
    
    return render_template('agents.html', title='Agent Status', agents_by_type=agents_by_type)

@app.route('/agents/<int:agent_id>/logs')
@login_required
def agent_logs(agent_id):
    """Agent logs route."""
    agent = Agent.query.get_or_404(agent_id)
    
    # Get logs with pagination
    page = request.args.get('page', 1, type=int)
    logs = AgentLog.query.filter_by(agent_id=agent_id).order_by(
        AgentLog.timestamp.desc()
    ).paginate(page=page, per_page=20, error_out=False)
    
    return render_template('agent_logs.html', title=f'Logs: {agent.agent_id}', agent=agent, logs=logs)

# API Routes

@app.route('/api/properties', methods=['GET'])
@require_api_key
def api_properties():
    """API endpoint for properties."""
    # Parse query parameters
    query = request.args.get('query', '')
    property_type = request.args.get('type', '')
    neighborhood = request.args.get('neighborhood', '')
    min_price = request.args.get('min_price', None)
    max_price = request.args.get('max_price', None)
    min_bedrooms = request.args.get('min_bedrooms', None)
    min_bathrooms = request.args.get('min_bathrooms', None)
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Build the query
    property_query = Property.query
    
    # Apply filters
    if query:
        property_query = property_query.filter(
            or_(
                Property.property_id.ilike(f'%{query}%'),
                Property.address.ilike(f'%{query}%'),
                Property.neighborhood.ilike(f'%{query}%'),
                Property.city.ilike(f'%{query}%'),
                Property.zip_code.ilike(f'%{query}%')
            )
        )
    
    if property_type:
        property_query = property_query.filter(Property.property_type == property_type)
    
    if neighborhood:
        property_query = property_query.filter(Property.neighborhood == neighborhood)
    
    if min_bedrooms:
        property_query = property_query.filter(Property.bedrooms >= int(min_bedrooms))
    
    if min_bathrooms:
        property_query = property_query.filter(Property.bathrooms >= float(min_bathrooms))
    
    # Filter by price range using the latest valuation
    if min_price or max_price:
        # This is a complex query, we'll use a subquery to get the latest valuation for each property
        latest_valuations = db.session.query(
            PropertyValuation.property_id,
            func.max(PropertyValuation.valuation_date).label('latest_date')
        ).group_by(PropertyValuation.property_id).subquery()
        
        price_query = db.session.query(
            PropertyValuation.property_id,
            PropertyValuation.estimated_value
        ).join(
            latest_valuations,
            and_(
                PropertyValuation.property_id == latest_valuations.c.property_id,
                PropertyValuation.valuation_date == latest_valuations.c.latest_date
            )
        )
        
        if min_price:
            price_query = price_query.filter(PropertyValuation.estimated_value >= float(min_price))
        
        if max_price:
            price_query = price_query.filter(PropertyValuation.estimated_value <= float(max_price))
        
        # Get property IDs that match the price criteria
        property_ids = [p.property_id for p in price_query.all()]
        
        # Filter the main query by these property IDs
        if property_ids:
            property_query = property_query.filter(Property.id.in_(property_ids))
        else:
            # If no properties match the price criteria, return an empty list
            return jsonify({
                "total": 0,
                "offset": offset,
                "limit": limit,
                "properties": []
            })
    
    # Count total matching properties
    total = property_query.count()
    
    # Apply pagination
    properties = property_query.order_by(Property.updated_at.desc()).offset(offset).limit(limit).all()
    
    # Get the latest valuation for each property
    property_ids = [p.id for p in properties]
    latest_valuations = {}
    
    if property_ids:
        # Subquery to get the latest valuation for each property
        latest_valuation_dates = db.session.query(
            PropertyValuation.property_id,
            func.max(PropertyValuation.valuation_date).label('latest_date')
        ).filter(PropertyValuation.property_id.in_(property_ids)).group_by(PropertyValuation.property_id).subquery()
        
        # Query to get the valuation objects
        valuations = db.session.query(PropertyValuation).join(
            latest_valuation_dates,
            and_(
                PropertyValuation.property_id == latest_valuation_dates.c.property_id,
                PropertyValuation.valuation_date == latest_valuation_dates.c.latest_date
            )
        ).all()
        
        # Create a dictionary of property_id to valuation
        latest_valuations = {v.property_id: v for v in valuations}
    
    # Format the response
    result = {
        "total": total,
        "offset": offset,
        "limit": limit,
        "properties": []
    }
    
    for property_obj in properties:
        property_data = {
            "id": property_obj.id,
            "property_id": property_obj.property_id,
            "address": property_obj.address,
            "city": property_obj.city,
            "state": property_obj.state,
            "zip_code": property_obj.zip_code,
            "neighborhood": property_obj.neighborhood,
            "property_type": property_obj.property_type,
            "year_built": property_obj.year_built,
            "bedrooms": property_obj.bedrooms,
            "bathrooms": property_obj.bathrooms,
            "living_area": property_obj.living_area,
            "land_area": property_obj.land_area,
            "latitude": property_obj.latitude,
            "longitude": property_obj.longitude,
            "created_at": property_obj.created_at.isoformat(),
            "updated_at": property_obj.updated_at.isoformat()
        }
        
        # Add latest valuation if available
        if property_obj.id in latest_valuations:
            valuation = latest_valuations[property_obj.id]
            property_data["latest_valuation"] = {
                "estimated_value": valuation.estimated_value,
                "confidence_score": valuation.confidence_score,
                "valuation_date": valuation.valuation_date.isoformat(),
                "valuation_method": valuation.valuation_method
            }
        
        result["properties"].append(property_data)
    
    return jsonify(result)

@app.route('/api/properties/<int:property_id>', methods=['GET'])
@require_api_key
def api_property_detail(property_id):
    """API endpoint for property details."""
    property_obj = Property.query.get_or_404(property_id)
    
    # Get all valuations for this property, ordered by date
    valuations = PropertyValuation.query.filter_by(
        property_id=property_id
    ).order_by(PropertyValuation.valuation_date.desc()).all()
    
    # Format the response
    result = {
        "id": property_obj.id,
        "property_id": property_obj.property_id,
        "address": property_obj.address,
        "city": property_obj.city,
        "state": property_obj.state,
        "zip_code": property_obj.zip_code,
        "neighborhood": property_obj.neighborhood,
        "property_type": property_obj.property_type,
        "year_built": property_obj.year_built,
        "bedrooms": property_obj.bedrooms,
        "bathrooms": property_obj.bathrooms,
        "living_area": property_obj.living_area,
        "land_area": property_obj.land_area,
        "latitude": property_obj.latitude,
        "longitude": property_obj.longitude,
        "created_at": property_obj.created_at.isoformat(),
        "updated_at": property_obj.updated_at.isoformat(),
        "valuations": []
    }
    
    for valuation in valuations:
        result["valuations"].append({
            "id": valuation.id,
            "estimated_value": valuation.estimated_value,
            "confidence_score": valuation.confidence_score,
            "valuation_date": valuation.valuation_date.isoformat(),
            "valuation_method": valuation.valuation_method,
            "created_at": valuation.created_at.isoformat()
        })
    
    return jsonify(result)

@app.route('/api/valuations', methods=['GET'])
@require_api_key
def api_valuations():
    """API endpoint for property valuations."""
    # Parse query parameters
    property_id = request.args.get('property_id', None, type=int)
    method = request.args.get('method', None)
    min_date = request.args.get('min_date', None)
    max_date = request.args.get('max_date', None)
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Build the query
    valuation_query = PropertyValuation.query
    
    # Apply filters
    if property_id:
        valuation_query = valuation_query.filter(PropertyValuation.property_id == property_id)
    
    if method:
        valuation_query = valuation_query.filter(PropertyValuation.valuation_method == method)
    
    if min_date:
        try:
            min_date = datetime.fromisoformat(min_date)
            valuation_query = valuation_query.filter(PropertyValuation.valuation_date >= min_date)
        except ValueError:
            return jsonify({"error": "Invalid min_date format. Use ISO format (YYYY-MM-DD)."}), 400
    
    if max_date:
        try:
            max_date = datetime.fromisoformat(max_date)
            valuation_query = valuation_query.filter(PropertyValuation.valuation_date <= max_date)
        except ValueError:
            return jsonify({"error": "Invalid max_date format. Use ISO format (YYYY-MM-DD)."}), 400
    
    # Count total matching valuations
    total = valuation_query.count()
    
    # Apply pagination
    valuations = valuation_query.order_by(PropertyValuation.valuation_date.desc()).offset(offset).limit(limit).all()
    
    # Get property details for each valuation
    property_ids = {v.property_id for v in valuations}
    properties = {p.id: p for p in Property.query.filter(Property.id.in_(property_ids)).all()} if property_ids else {}
    
    # Format the response
    result = {
        "total": total,
        "offset": offset,
        "limit": limit,
        "valuations": []
    }
    
    for valuation in valuations:
        valuation_data = {
            "id": valuation.id,
            "property_id": valuation.property_id,
            "estimated_value": valuation.estimated_value,
            "confidence_score": valuation.confidence_score,
            "valuation_date": valuation.valuation_date.isoformat(),
            "valuation_method": valuation.valuation_method,
            "created_at": valuation.created_at.isoformat()
        }
        
        # Add property details if available
        if valuation.property_id in properties:
            property_obj = properties[valuation.property_id]
            valuation_data["property"] = {
                "property_id": property_obj.property_id,
                "address": property_obj.address,
                "city": property_obj.city,
                "state": property_obj.state,
                "zip_code": property_obj.zip_code,
                "neighborhood": property_obj.neighborhood,
                "property_type": property_obj.property_type
            }
        
        result["valuations"].append(valuation_data)
    
    return jsonify(result)

@app.route('/api/etl-status', methods=['GET'])
@require_api_key
def api_etl_status():
    """API endpoint for ETL status."""
    status = EtlStatus.query.order_by(EtlStatus.last_update.desc()).first()
    
    if not status:
        return jsonify({
            "status": "unknown",
            "message": "No ETL status records found."
        }), 404
    
    # Get data sources
    sources = DataSource.query.filter_by(etl_status_id=status.id).all()
    
    # Format the response
    result = {
        "id": status.id,
        "status": status.status,
        "progress": status.progress,
        "last_update": status.last_update.isoformat(),
        "records_processed": status.records_processed,
        "success_rate": status.success_rate,
        "average_processing_time": status.average_processing_time,
        "completeness": status.completeness,
        "accuracy": status.accuracy,
        "timeliness": status.timeliness,
        "sources": []
    }
    
    for source in sources:
        result["sources"].append({
            "id": source.id,
            "name": source.name,
            "status": source.status,
            "records": source.records,
            "created_at": source.created_at.isoformat(),
            "updated_at": source.updated_at.isoformat()
        })
    
    return jsonify(result)

@app.route('/api/agent-status', methods=['GET'])
@require_api_key
def api_agent_status():
    """API endpoint for agent status."""
    # Parse query parameters
    agent_type = request.args.get('agent_type', None)
    status = request.args.get('status', None)
    
    # Build the query
    agent_query = Agent.query
    
    # Apply filters
    if agent_type:
        agent_query = agent_query.filter(Agent.agent_type == agent_type)
    
    if status:
        agent_query = agent_query.filter(Agent.status == status)
    
    # Get all agents matching the criteria
    agents = agent_query.order_by(Agent.agent_type, Agent.agent_id).all()
    
    # Format the response
    result = {
        "count": len(agents),
        "agents": []
    }
    
    for agent in agents:
        result["agents"].append({
            "id": agent.id,
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "last_active": agent.last_active.isoformat(),
            "queue_size": agent.queue_size,
            "total_processed": agent.total_processed,
            "success_rate": agent.success_rate,
            "average_processing_time": agent.average_processing_time,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat()
        })
    
    return jsonify(result)

@app.route('/api/agent-logs/<int:agent_id>', methods=['GET'])
@require_api_key
def api_agent_logs(agent_id):
    """API endpoint for agent logs."""
    agent = Agent.query.get_or_404(agent_id)
    
    # Parse query parameters
    level = request.args.get('level', None)
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Build the query
    log_query = AgentLog.query.filter_by(agent_id=agent_id)
    
    # Apply filters
    if level:
        log_query = log_query.filter(AgentLog.level == level)
    
    # Count total matching logs
    total = log_query.count()
    
    # Apply pagination
    logs = log_query.order_by(AgentLog.timestamp.desc()).offset(offset).limit(limit).all()
    
    # Format the response
    result = {
        "agent_id": agent.agent_id,
        "agent_type": agent.agent_type,
        "status": agent.status,
        "total": total,
        "offset": offset,
        "limit": limit,
        "logs": []
    }
    
    for log in logs:
        result["logs"].append({
            "id": log.id,
            "level": log.level,
            "message": log.message,
            "timestamp": log.timestamp.isoformat()
        })
    
    return jsonify(result)

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    """404 error handler."""
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler."""
    db.session.rollback()  # Roll back any failed database transaction
    logger.error(f"Internal Server Error: {str(error)}")
    return render_template('errors/500.html'), 500