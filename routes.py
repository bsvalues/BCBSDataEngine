import os
import json
import secrets
import logging
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, abort, current_app, g
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.urls import url_parse
import jwt

from app import db
from models import User, Property, Valuation, Agent, AgentLog, ETLJob, ApiKey, PropertyImage, MarketTrend
from forms import (
    LoginForm, RegistrationForm, PropertyForm, ValuationForm, AgentForm, 
    ApiKeyForm, SearchForm, ProfileForm, ChangePasswordForm, ContactForm
)


# Configure logging
logger = logging.getLogger(__name__)

# Create blueprints
main_bp = Blueprint('main', __name__)
auth_bp = Blueprint('auth', __name__)
api_bp = Blueprint('api', __name__)
admin_bp = Blueprint('admin', __name__)
error_bp = Blueprint('errors', __name__)


# Utility functions and decorators
def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin privileges required to access this resource.", "danger")
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function


def api_key_required(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        api_key = None
        
        if auth_header:
            try:
                auth_type, api_key = auth_header.split(' ', 1)
                if auth_type.lower() != 'apikey':
                    return jsonify({"error": "Invalid authorization header format. Use 'ApiKey YOUR_API_KEY'"}), 401
            except ValueError:
                return jsonify({"error": "Invalid authorization header format. Use 'ApiKey YOUR_API_KEY'"}), 401
        else:
            api_key = request.args.get('api_key')
        
        if not api_key:
            return jsonify({"error": "API key is required"}), 401
        
        api_key_obj = ApiKey.query.filter_by(key=api_key, is_active=True).first()
        
        if not api_key_obj:
            return jsonify({"error": "Invalid or inactive API key"}), 401
        
        # Update last used timestamp
        api_key_obj.last_used = datetime.utcnow()
        db.session.commit()
        
        # Store the user associated with this API key for later use
        g.api_user = api_key_obj.user
        
        return f(*args, **kwargs)
    return decorated_function


def jwt_token_required(f):
    """Decorator to require JWT token authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        token = None
        
        if auth_header:
            try:
                auth_type, token = auth_header.split(' ', 1)
                if auth_type.lower() != 'bearer':
                    return jsonify({"error": "Invalid authorization header format. Use 'Bearer YOUR_TOKEN'"}), 401
            except ValueError:
                return jsonify({"error": "Invalid authorization header format. Use 'Bearer YOUR_TOKEN'"}), 401
        else:
            token = request.args.get('token')
        
        if not token:
            return jsonify({"error": "JWT token is required"}), 401
        
        try:
            # Decode the JWT token
            secret_key = current_app.config['SECRET_KEY']
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            
            # Store agent information for later use
            g.agent_id = payload.get('agent_id')
            g.agent_type = payload.get('agent_type')
            
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


# Main routes
@main_bp.route('/')
def index():
    """Home page route"""
    return render_template('index.html')


@main_bp.route('/about')
def about():
    """About page route"""
    return render_template('about.html')


@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page route"""
    form = ContactForm()
    if form.validate_on_submit():
        # Process the contact form (would typically send an email)
        flash("Your message has been sent! We'll get back to you soon.", "success")
        return redirect(url_for('main.contact'))
    return render_template('contact.html', form=form)


@main_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard route"""
    # Get property count
    property_count = Property.query.count()
    
    # Get valuation count
    valuation_count = Valuation.query.count()
    
    # Get active agents
    active_agents = Agent.query.all()
    agent_count = len(active_agents)
    
    # Get recent valuations
    recent_valuations = Valuation.query.order_by(Valuation.valuation_date.desc()).limit(5).all()
    
    # Get ETL jobs
    etl_jobs = ETLJob.query.order_by(ETLJob.start_time.desc()).limit(3).all()
    
    return render_template('dashboard.html', 
                            property_count=property_count,
                            valuation_count=valuation_count,
                            agent_count=agent_count,
                            active_agents=active_agents,
                            recent_valuations=recent_valuations,
                            etl_jobs=etl_jobs)


@main_bp.route('/properties')
@login_required
def properties():
    """Properties listing route"""
    search_form = SearchForm()
    page = request.args.get('page', 1, type=int)
    query = Property.query
    
    # Apply search filters if provided
    if request.args.get('query'):
        search_term = f"%{request.args.get('query')}%"
        query = query.filter(Property.address.ilike(search_term) | 
                             Property.neighborhood.ilike(search_term) |
                             Property.city.ilike(search_term))
    
    if request.args.get('property_type'):
        query = query.filter(Property.property_type == request.args.get('property_type'))
    
    if request.args.get('city'):
        query = query.filter(Property.city.ilike(f"%{request.args.get('city')}%"))
    
    if request.args.get('min_price'):
        # We'll use the last_sold_price as a proxy for price
        query = query.filter(Property.last_sold_price >= float(request.args.get('min_price')))
    
    if request.args.get('max_price'):
        # We'll use the last_sold_price as a proxy for price
        query = query.filter(Property.last_sold_price <= float(request.args.get('max_price')))
    
    if request.args.get('min_beds'):
        query = query.filter(Property.bedrooms >= int(request.args.get('min_beds')))
    
    if request.args.get('min_baths'):
        query = query.filter(Property.bathrooms >= float(request.args.get('min_baths')))
    
    if request.args.get('min_sqft'):
        query = query.filter(Property.square_feet >= int(request.args.get('min_sqft')))
    
    # Paginate the results
    properties = query.paginate(page=page, per_page=12)
    
    return render_template('properties.html', properties=properties, form=search_form)


@main_bp.route('/property/<int:property_id>')
@login_required
def property_detail(property_id):
    """Property detail route"""
    property = Property.query.get_or_404(property_id)
    
    # Get the most recent valuation for this property
    latest_valuation = Valuation.query.filter_by(property_id=property_id).order_by(Valuation.valuation_date.desc()).first()
    
    # Get valuation history for this property
    valuation_history = Valuation.query.filter_by(property_id=property_id).order_by(Valuation.valuation_date.desc()).all()
    
    # Get property images
    images = PropertyImage.query.filter_by(property_id=property_id).all()
    
    return render_template('property_detail.html', 
                           property=property, 
                           latest_valuation=latest_valuation,
                           valuation_history=valuation_history,
                           images=images)


@main_bp.route('/valuations')
@login_required
def valuations():
    """Valuations listing route"""
    page = request.args.get('page', 1, type=int)
    query = Valuation.query
    
    # Apply filters if provided
    if request.args.get('method'):
        query = query.filter(Valuation.valuation_method == request.args.get('method'))
    
    if request.args.get('min_confidence'):
        query = query.filter(Valuation.confidence_score >= float(request.args.get('min_confidence')))
    
    if request.args.get('property_id'):
        query = query.filter(Valuation.property_id == int(request.args.get('property_id')))
    
    # Paginate the results
    valuations = query.order_by(Valuation.valuation_date.desc()).paginate(page=page, per_page=20)
    
    return render_template('valuations.html', valuations=valuations)


@main_bp.route('/agents')
@login_required
def agents():
    """Agents listing route"""
    agents = Agent.query.all()
    return render_template('agents.html', agents=agents)


@main_bp.route('/etl_jobs')
@login_required
def etl_jobs():
    """ETL jobs listing route"""
    page = request.args.get('page', 1, type=int)
    query = ETLJob.query
    
    # Apply filters if provided
    if request.args.get('job_type'):
        query = query.filter(ETLJob.job_type == request.args.get('job_type'))
    
    if request.args.get('status'):
        query = query.filter(ETLJob.status == request.args.get('status'))
    
    # Paginate the results
    jobs = query.order_by(ETLJob.start_time.desc()).paginate(page=page, per_page=20)
    
    return render_template('etl_jobs.html', jobs=jobs)


@main_bp.route('/market_trends')
@login_required
def market_trends():
    """Market trends analysis route"""
    trends = MarketTrend.query.order_by(MarketTrend.trend_date.desc()).all()
    
    # Group trends by neighborhood
    neighborhoods = {}
    for trend in trends:
        if trend.neighborhood not in neighborhoods:
            neighborhoods[trend.neighborhood] = []
        neighborhoods[trend.neighborhood].append(trend)
    
    return render_template('market_trends.html', neighborhoods=neighborhoods, trends=trends)


@main_bp.route('/what_if_analysis')
@login_required
def what_if_analysis():
    """What-if analysis tool route"""
    return render_template('what_if_analysis.html')


@main_bp.route('/reports')
@login_required
def reports():
    """Reports generation route"""
    return render_template('reports.html')


@main_bp.route('/api_docs')
def api_docs():
    """API documentation route"""
    return render_template('api_docs.html')


@main_bp.route('/privacy')
def privacy():
    """Privacy policy route"""
    return render_template('privacy.html')


@main_bp.route('/terms')
def terms():
    """Terms of service route"""
    return render_template('terms.html')


@main_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile route"""
    form = ProfileForm(original_username=current_user.username, original_email=current_user.email)
    
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('main.profile'))
    
    if request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
    
    return render_template('profile.html', form=form)


@main_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """User settings route"""
    password_form = ChangePasswordForm()
    api_key_form = ApiKeyForm()
    
    if password_form.validate_on_submit():
        if current_user.check_password(password_form.current_password.data):
            current_user.set_password(password_form.new_password.data)
            db.session.commit()
            flash('Your password has been updated.', 'success')
            return redirect(url_for('main.settings'))
        else:
            flash('Current password is incorrect.', 'danger')
    
    # Get user's API keys
    api_keys = ApiKey.query.filter_by(user_id=current_user.id).all()
    
    return render_template('settings.html', 
                           password_form=password_form, 
                           api_key_form=api_key_form,
                           api_keys=api_keys)


@main_bp.route('/settings/generate_api_key', methods=['POST'])
@login_required
def generate_api_key():
    """Generate a new API key for the current user"""
    form = ApiKeyForm()
    
    if form.validate_on_submit():
        # Generate a secure random API key
        key = secrets.token_hex(32)
        
        # Create a new API key record
        api_key = ApiKey(
            key=key,
            name=form.name.data,
            user_id=current_user.id,
            permissions=form.permissions.data,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),  # 1 year expiration
            is_active=True
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        flash(f'New API key generated: {key}. Please save this key as it will not be shown again.', 'success')
        return redirect(url_for('main.settings'))
    
    flash('Failed to generate API key. Please check the form and try again.', 'danger')
    return redirect(url_for('main.settings'))


@main_bp.route('/settings/revoke_api_key/<int:key_id>', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    """Revoke an API key"""
    api_key = ApiKey.query.get_or_404(key_id)
    
    # Ensure the API key belongs to the current user
    if api_key.user_id != current_user.id:
        flash('You do not have permission to revoke this API key.', 'danger')
        return redirect(url_for('main.settings'))
    
    api_key.is_active = False
    db.session.commit()
    
    flash('API key has been revoked.', 'success')
    return redirect(url_for('main.settings'))


# Authentication routes
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('auth.login'))
        
        # Update last login timestamp
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        login_user(user, remember=form.remember.data)
        
        # Redirect to the page the user was trying to access
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.dashboard')
        
        return redirect(next_page)
    
    return render_template('login.html', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            created_at=datetime.utcnow()
        )
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Congratulations, you are now registered! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """User logout route"""
    logout_user()
    return redirect(url_for('main.index'))


# API routes
@api_bp.route('/health', methods=['GET'])
def health():
    """API health check route"""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@api_bp.route('/properties', methods=['GET'])
@api_key_required
def get_properties():
    """API route to get properties"""
    page = request.args.get('page', 1, type=int)
    per_page = min(int(request.args.get('per_page', 20)), 100)  # Limit max per_page to 100
    
    query = Property.query
    
    # Apply filters if provided
    if request.args.get('property_type'):
        query = query.filter(Property.property_type == request.args.get('property_type'))
    
    if request.args.get('city'):
        query = query.filter(Property.city.ilike(f"%{request.args.get('city')}%"))
    
    if request.args.get('neighborhood'):
        query = query.filter(Property.neighborhood.ilike(f"%{request.args.get('neighborhood')}%"))
    
    # Paginate the results
    properties_page = query.paginate(page=page, per_page=per_page)
    
    result = {
        "total": properties_page.total,
        "page": properties_page.page,
        "per_page": per_page,
        "pages": properties_page.pages,
        "properties": []
    }
    
    for prop in properties_page.items:
        result["properties"].append({
            "id": prop.id,
            "address": prop.address,
            "city": prop.city,
            "state": prop.state,
            "zip_code": prop.zip_code,
            "property_type": prop.property_type,
            "bedrooms": prop.bedrooms,
            "bathrooms": prop.bathrooms,
            "square_feet": prop.square_feet,
            "year_built": prop.year_built,
            "latitude": prop.latitude,
            "longitude": prop.longitude,
            "neighborhood": prop.neighborhood
        })
    
    return jsonify(result)


@api_bp.route('/properties/<int:property_id>', methods=['GET'])
@api_key_required
def get_property(property_id):
    """API route to get a specific property"""
    property = Property.query.get_or_404(property_id)
    
    result = {
        "id": property.id,
        "address": property.address,
        "city": property.city,
        "state": property.state,
        "zip_code": property.zip_code,
        "property_type": property.property_type,
        "bedrooms": property.bedrooms,
        "bathrooms": property.bathrooms,
        "square_feet": property.square_feet,
        "lot_size": property.lot_size,
        "year_built": property.year_built,
        "last_sold_date": property.last_sold_date.isoformat() if property.last_sold_date else None,
        "last_sold_price": property.last_sold_price,
        "latitude": property.latitude,
        "longitude": property.longitude,
        "neighborhood": property.neighborhood,
        "description": property.description,
        "features": property.features,
        "created_at": property.created_at.isoformat(),
        "updated_at": property.updated_at.isoformat()
    }
    
    return jsonify(result)


@api_bp.route('/valuations', methods=['GET'])
@api_key_required
def get_valuations():
    """API route to get valuations"""
    page = request.args.get('page', 1, type=int)
    per_page = min(int(request.args.get('per_page', 20)), 100)  # Limit max per_page to 100
    
    query = Valuation.query
    
    # Apply filters if provided
    if request.args.get('method'):
        query = query.filter(Valuation.valuation_method == request.args.get('method'))
    
    if request.args.get('min_confidence'):
        query = query.filter(Valuation.confidence_score >= float(request.args.get('min_confidence')))
    
    if request.args.get('property_id'):
        query = query.filter(Valuation.property_id == int(request.args.get('property_id')))
    
    if request.args.get('after_date'):
        try:
            after_date = datetime.fromisoformat(request.args.get('after_date'))
            query = query.filter(Valuation.valuation_date >= after_date)
        except ValueError:
            return jsonify({"error": "Invalid after_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
    
    if request.args.get('before_date'):
        try:
            before_date = datetime.fromisoformat(request.args.get('before_date'))
            query = query.filter(Valuation.valuation_date <= before_date)
        except ValueError:
            return jsonify({"error": "Invalid before_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
    
    # Apply sorting
    sort_by = request.args.get('sort_by', 'valuation_date')
    sort_dir = request.args.get('sort_dir', 'desc')
    
    if sort_dir == 'desc':
        query = query.order_by(getattr(Valuation, sort_by).desc())
    else:
        query = query.order_by(getattr(Valuation, sort_by))
    
    # Paginate the results
    valuations_page = query.paginate(page=page, per_page=per_page)
    
    result = {
        "total": valuations_page.total,
        "page": valuations_page.page,
        "per_page": per_page,
        "pages": valuations_page.pages,
        "valuations": []
    }
    
    for val in valuations_page.items:
        result["valuations"].append({
            "id": val.id,
            "property_id": val.property_id,
            "agent_id": val.agent_id,
            "estimated_value": val.estimated_value,
            "confidence_score": val.confidence_score,
            "valuation_date": val.valuation_date.isoformat(),
            "valuation_method": val.valuation_method,
            "model_version": val.model_version,
            "adjusted_value": val.adjusted_value,
            "property_address": val.property.address if val.property else None
        })
    
    return jsonify(result)


@api_bp.route('/valuations/<int:valuation_id>', methods=['GET'])
@api_key_required
def get_valuation(valuation_id):
    """API route to get a specific valuation"""
    valuation = Valuation.query.get_or_404(valuation_id)
    
    result = {
        "id": valuation.id,
        "property_id": valuation.property_id,
        "user_id": valuation.user_id,
        "agent_id": valuation.agent_id,
        "estimated_value": valuation.estimated_value,
        "confidence_score": valuation.confidence_score,
        "valuation_date": valuation.valuation_date.isoformat(),
        "valuation_method": valuation.valuation_method,
        "model_version": valuation.model_version,
        "adjusted_value": valuation.adjusted_value,
        "adjustment_factors": valuation.adjustment_factors,
        "comparable_properties": valuation.comparable_properties,
        "metrics": valuation.metrics,
        "notes": valuation.notes,
        "property": {
            "id": valuation.property.id,
            "address": valuation.property.address,
            "city": valuation.property.city,
            "state": valuation.property.state,
            "property_type": valuation.property.property_type
        } if valuation.property else None,
        "agent": {
            "id": valuation.agent.id,
            "name": valuation.agent.name,
            "agent_type": valuation.agent.agent_type,
            "model_version": valuation.agent.model_version
        } if valuation.agent else None
    }
    
    return jsonify(result)


@api_bp.route('/valuation/request', methods=['POST'])
@api_key_required
def request_valuation():
    """API route to request a new property valuation"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ['property_id', 'valuation_method']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Check if the property exists
    property_obj = Property.query.get(data['property_id'])
    if not property_obj:
        return jsonify({"error": f"Property with ID {data['property_id']} not found"}), 404
    
    # Find an available agent for the requested method
    agent = Agent.query.filter_by(agent_type=data['valuation_method'], status='idle').first()
    
    try:
        # Import the valuation module
        from valuation.core import calculate_basic_valuation, save_valuation
        
        # Calculate the valuation
        valuation_result = calculate_basic_valuation(property_obj)
        
        # Save it to the database
        valuation_id = save_valuation(property_obj.id, valuation_result)
        
        # Get the saved valuation
        valuation = Valuation.query.get(valuation_id)
        
        # Return the valuation result
        return jsonify({
            "status": "completed",
            "valuation_id": valuation.id,
            "property_id": property_obj.id,
            "estimated_value": valuation.estimated_value,
            "confidence_score": valuation.confidence_score,
            "valuation_date": valuation.valuation_date.isoformat(),
            "valuation_method": valuation.valuation_method
        }), 200
        
    except ImportError:
        # Fallback to the old behavior if the valuation module isn't available
        logger.warning("Valuation module not available, falling back to queued valuation")
        
        # Create a new valuation record
        valuation = Valuation(
            property_id=data['property_id'],
            user_id=g.api_user.id,
            agent_id=agent.id if agent else None,
            estimated_value=0.0,  # Will be updated by the agent
            confidence_score=0.0,  # Will be updated by the agent
            valuation_date=datetime.utcnow(),
            valuation_method=data['valuation_method'],
            notes=data.get('notes')
        )
        
        db.session.add(valuation)
        db.session.commit()
        
        return jsonify({
            "status": "pending",
            "valuation_id": valuation.id,
            "message": "Valuation request received and queued for processing"
        }), 202
    except Exception as e:
        logger.error(f"Error performing valuation: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "An error occurred while performing the valuation"
        }), 500


@main_bp.route('/calculate-valuation/<int:property_id>', methods=['GET', 'POST'])
@login_required
def calculate_valuation(property_id):
    """Calculate valuation for a property and display the result"""
    # Get the property
    property_obj = Property.query.get_or_404(property_id)
    
    # Default method
    valuation_method = 'basic'
    
    # Check if form was submitted
    if request.method == 'POST':
        valuation_method = request.form.get('valuation_method', 'basic')
    
    try:
        # Import valuation module
        from valuation.core import calculate_basic_valuation, save_valuation
        
        # Calculate valuation
        valuation_result = calculate_basic_valuation(property_obj)
        
        # Save to database
        valuation_id = save_valuation(property_obj.id, valuation_result)
        
        # Get the saved valuation
        valuation = Valuation.query.get(valuation_id)
        
        flash(f"Valuation completed: ${valuation.estimated_value:,.2f} with {valuation.confidence_score:.2%} confidence", "success")
        
        # Redirect to property detail page
        return redirect(url_for('main.property_detail', property_id=property_id))
        
    except ImportError:
        flash("Valuation module is not available", "danger")
        return redirect(url_for('main.property_detail', property_id=property_id))
    except Exception as e:
        flash(f"Error calculating valuation: {str(e)}", "danger")
        return redirect(url_for('main.property_detail', property_id=property_id))


@api_bp.route('/agent-status', methods=['GET'])
@api_key_required
def get_agent_status():
    """API route to get status of valuation agents"""
    query = Agent.query
    
    # Apply filters if provided
    if request.args.get('agent_type'):
        query = query.filter(Agent.agent_type == request.args.get('agent_type'))
    
    if request.args.get('status'):
        query = query.filter(Agent.status == request.args.get('status'))
    
    if request.args.get('active_only') == 'true':
        query = query.filter(Agent.status != 'error')
    
    agents = query.all()
    
    result = {
        "total": len(agents),
        "agents": []
    }
    
    for agent in agents:
        result["agents"].append({
            "id": agent.id,
            "name": agent.name,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "model_version": agent.model_version,
            "last_active": agent.last_active.isoformat() if agent.last_active else None,
            "processing_count": agent.processing_count,
            "success_count": agent.success_count,
            "error_count": agent.error_count,
            "average_confidence": agent.average_confidence
        })
    
    return jsonify(result)


@api_bp.route('/agent-logs/<int:agent_id>', methods=['GET'])
@api_key_required
def get_agent_logs(agent_id):
    """API route to get logs for a specific agent"""
    agent = Agent.query.get_or_404(agent_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = min(int(request.args.get('per_page', 20)), 100)  # Limit max per_page to 100
    
    query = AgentLog.query.filter_by(agent_id=agent_id)
    
    # Apply filters if provided
    if request.args.get('log_level'):
        query = query.filter(AgentLog.log_level == request.args.get('log_level'))
    
    # Paginate the results
    logs_page = query.order_by(AgentLog.timestamp.desc()).paginate(page=page, per_page=per_page)
    
    result = {
        "agent": {
            "id": agent.id,
            "name": agent.name,
            "agent_type": agent.agent_type,
            "status": agent.status
        },
        "total": logs_page.total,
        "page": logs_page.page,
        "per_page": per_page,
        "pages": logs_page.pages,
        "logs": []
    }
    
    for log in logs_page.items:
        result["logs"].append({
            "id": log.id,
            "log_level": log.log_level,
            "message": log.message,
            "timestamp": log.timestamp.isoformat(),
            "details": log.details
        })
    
    return jsonify(result)


@api_bp.route('/etl-status', methods=['GET'])
@api_key_required
def get_etl_status():
    """API route to get status of ETL pipeline jobs"""
    query = ETLJob.query
    
    # Apply filters if provided
    if request.args.get('job_type'):
        query = query.filter(ETLJob.job_type == request.args.get('job_type'))
    
    if request.args.get('status'):
        query = query.filter(ETLJob.status == request.args.get('status'))
    
    if request.args.get('timeframe'):
        timeframe = request.args.get('timeframe')
        now = datetime.utcnow()
        
        if timeframe == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(ETLJob.start_time >= start_date)
        elif timeframe == 'yesterday':
            start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(ETLJob.start_time >= start_date, ETLJob.start_time < end_date)
        elif timeframe == 'this_week':
            # Start of week (Monday)
            start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(ETLJob.start_time >= start_date)
        elif timeframe == 'last_week':
            # Start of last week (Monday)
            start_date = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
            # End of last week (Sunday)
            end_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(ETLJob.start_time >= start_date, ETLJob.start_time < end_date)
        elif timeframe == 'this_month':
            # Start of month
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(ETLJob.start_time >= start_date)
    
    # Apply limit
    limit = min(int(request.args.get('limit', 20)), 100)  # Limit max results to 100
    
    etl_jobs = query.order_by(ETLJob.start_time.desc()).limit(limit).all()
    
    result = {
        "total": len(etl_jobs),
        "jobs": []
    }
    
    for job in etl_jobs:
        result["jobs"].append({
            "id": job.id,
            "job_type": job.job_type,
            "source": job.source,
            "status": job.status,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "records_processed": job.records_processed,
            "total_records": job.total_records,
            "success_count": job.success_count,
            "error_count": job.error_count,
            "error_details": job.error_details,
            "duration": (job.end_time - job.start_time).total_seconds() if job.end_time else None
        })
    
    return jsonify(result)


@api_bp.route('/token', methods=['POST'])
@api_key_required
def generate_token():
    """API route to generate a JWT token for agents"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ['agent_id', 'agent_type']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # In a real application, we would validate the agent's credentials
    # For now, we'll just check if the agent exists
    agent = Agent.query.get(data['agent_id'])
    if not agent:
        return jsonify({"error": f"Agent with ID {data['agent_id']} not found"}), 404
    
    # Generate a JWT token
    payload = {
        'agent_id': data['agent_id'],
        'agent_type': data['agent_type'],
        'exp': datetime.utcnow() + timedelta(days=1)  # 24 hour expiration
    }
    
    secret_key = current_app.config['SECRET_KEY']
    token = jwt.encode(payload, secret_key, algorithm='HS256')
    
    return jsonify({
        "token": token,
        "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat()
    })


@api_bp.route('/market-trends', methods=['GET'])
@api_key_required
def get_market_trends():
    """API route to get market trends data"""
    query = MarketTrend.query
    
    # Apply filters if provided
    if request.args.get('neighborhood'):
        query = query.filter(MarketTrend.neighborhood == request.args.get('neighborhood'))
    
    if request.args.get('city'):
        query = query.filter(MarketTrend.city == request.args.get('city'))
    
    if request.args.get('state'):
        query = query.filter(MarketTrend.state == request.args.get('state'))
    
    if request.args.get('property_type'):
        query = query.filter(MarketTrend.property_type == request.args.get('property_type'))
    
    # Get date range
    if request.args.get('start_date'):
        try:
            start_date = datetime.fromisoformat(request.args.get('start_date')).date()
            query = query.filter(MarketTrend.trend_date >= start_date)
        except ValueError:
            return jsonify({"error": "Invalid start_date format. Use ISO format (YYYY-MM-DD)"}), 400
    
    if request.args.get('end_date'):
        try:
            end_date = datetime.fromisoformat(request.args.get('end_date')).date()
            query = query.filter(MarketTrend.trend_date <= end_date)
        except ValueError:
            return jsonify({"error": "Invalid end_date format. Use ISO format (YYYY-MM-DD)"}), 400
    
    trends = query.order_by(MarketTrend.trend_date).all()
    
    result = {
        "total": len(trends),
        "trends": []
    }
    
    for trend in trends:
        result["trends"].append({
            "id": trend.id,
            "neighborhood": trend.neighborhood,
            "city": trend.city,
            "state": trend.state,
            "trend_date": trend.trend_date.isoformat(),
            "median_price": trend.median_price,
            "average_price": trend.average_price,
            "price_per_sqft": trend.price_per_sqft,
            "inventory_count": trend.inventory_count,
            "days_on_market": trend.days_on_market,
            "month_over_month": trend.month_over_month,
            "year_over_year": trend.year_over_year,
            "property_type": trend.property_type
        })
    
    return jsonify(result)


# Admin routes
@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    """Admin dashboard route"""
    user_count = User.query.count()
    property_count = Property.query.count()
    valuation_count = Valuation.query.count()
    agent_count = Agent.query.count()
    
    # Get recent users
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    
    # Get recent valuations
    recent_valuations = Valuation.query.order_by(Valuation.valuation_date.desc()).limit(5).all()
    
    # Get ETL jobs
    etl_jobs = ETLJob.query.order_by(ETLJob.start_time.desc()).limit(5).all()
    
    return render_template('admin/dashboard.html', 
                           user_count=user_count,
                           property_count=property_count,
                           valuation_count=valuation_count,
                           agent_count=agent_count,
                           recent_users=recent_users,
                           recent_valuations=recent_valuations,
                           etl_jobs=etl_jobs)


@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """Admin user management route"""
    page = request.args.get('page', 1, type=int)
    users = User.query.paginate(page=page, per_page=20)
    return render_template('admin/users.html', users=users)


@admin_bp.route('/users/<int:user_id>')
@login_required
@admin_required
def user_detail(user_id):
    """Admin user detail route"""
    user = User.query.get_or_404(user_id)
    return render_template('admin/user_detail.html', user=user)


@admin_bp.route('/agents')
@login_required
@admin_required
def agents():
    """Admin agent management route"""
    agents = Agent.query.all()
    return render_template('admin/agents.html', agents=agents)


@admin_bp.route('/agents/<int:agent_id>')
@login_required
@admin_required
def agent_detail(agent_id):
    """Admin agent detail route"""
    agent = Agent.query.get_or_404(agent_id)
    logs = AgentLog.query.filter_by(agent_id=agent_id).order_by(AgentLog.timestamp.desc()).limit(100).all()
    return render_template('admin/agent_detail.html', agent=agent, logs=logs)


@admin_bp.route('/etl_config')
@login_required
@admin_required
def etl_config():
    """Admin ETL configuration route"""
    return render_template('admin/etl_config.html')


@admin_bp.route('/settings')
@login_required
@admin_required
def settings():
    """Admin system settings route"""
    return render_template('admin/settings.html')


# Error handlers
@error_bp.app_errorhandler(404)
def handle_404(e):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404


@error_bp.app_errorhandler(500)
def handle_500(e):
    """Handle 500 errors"""
    return render_template('errors/500.html'), 500