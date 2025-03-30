from flask import render_template, request, jsonify, redirect, url_for, flash, session
import requests
import os
import json
from app import app, db
from models import Property, Valuation, GISFeature, User
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# API base URL
API_BASE_URL = "http://localhost:5001"  # API server running on port 5001

# API Key for backend communication
API_KEY = os.environ.get("BCBS_VALUES_API_KEY", "")

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/valuation')
@login_required
def valuation():
    """Render the valuation tool page."""
    return render_template('valuation_form.html')

@app.route('/valuation/result')
@login_required
def valuation_result():
    """Display valuation results."""
    property_id = request.args.get('property_id')
    if not property_id:
        return redirect(url_for('valuation'))
    
    # Try to get valuation data from the database
    property_data = Property.query.filter_by(property_id=property_id).first()
    if property_data:
        # Get latest valuation
        valuation_data = Valuation.query.filter_by(property_id=property_data.id).order_by(Valuation.valuation_date.desc()).first()
        return render_template('valuation_result.html', property=property_data, valuation=valuation_data)
    
    # If not in database, try to fetch from API
    try:
        url = f"{API_BASE_URL}/api/properties/{property_id}/valuation"
        headers = {"X-API-KEY": API_KEY}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            valuation_data = response.json()
            return render_template('valuation_result.html', property_id=property_id, valuation=valuation_data)
    except requests.RequestException:
        pass
    
    # If we get here, we couldn't find the valuation
    return render_template('valuation_result.html', property_id=property_id, error="No valuation found for this property.")

@app.route('/properties')
@login_required
def properties_list():
    """Display a list of all properties."""
    # Get query parameters for filtering
    neighborhood = request.args.get('neighborhood')
    min_value = request.args.get('min_value')
    max_value = request.args.get('max_value')
    
    # Start with base query
    query = Property.query
    
    # Apply filters if provided
    if neighborhood:
        query = query.filter(Property.address.ilike(f"%{neighborhood}%"))
    
    # Get properties
    properties = query.all()
    
    # If we need valuation filtering, we need to fetch valuations too
    if min_value or max_value:
        filtered_properties = []
        for prop in properties:
            # Get latest valuation
            latest_valuation = Valuation.query.filter_by(property_id=prop.id).order_by(Valuation.valuation_date.desc()).first()
            
            if latest_valuation:
                if min_value and latest_valuation.estimated_value < float(min_value):
                    continue
                if max_value and latest_valuation.estimated_value > float(max_value):
                    continue
                # Add property with its valuation data
                prop.latest_valuation = latest_valuation
                filtered_properties.append(prop)
        
        properties = filtered_properties
    
    return render_template('properties.html', properties=properties)

@app.route('/properties/<property_id>')
@login_required
def property_detail(property_id):
    """Display detailed information about a specific property."""
    property_data = Property.query.filter_by(property_id=property_id).first()
    
    if not property_data:
        # Try to fetch from API
        try:
            url = f"{API_BASE_URL}/api/properties/{property_id}"
            headers = {"X-API-KEY": API_KEY}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                property_data = response.json()
                return render_template('property_detail.html', property=property_data)
        except requests.RequestException:
            pass
        
        # If we get here, we couldn't find the property
        return render_template('404.html'), 404
    
    # Get valuations for this property
    valuations = Valuation.query.filter_by(property_id=property_data.id).order_by(Valuation.valuation_date.desc()).all()
    
    # Get GIS features for this property
    gis_features = GISFeature.query.filter_by(property_id=property_data.id).all()
    
    return render_template('property_detail.html', property=property_data, valuations=valuations, gis_features=gis_features)

@app.route('/search')
@login_required
def search():
    """Search properties by address, ID, or other criteria."""
    return render_template('search.html')

@app.route('/search/results')
@login_required
def search_results():
    """Display search results."""
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('search'))
    
    # Search in database
    properties = Property.query.filter(
        (Property.property_id.ilike(f"%{query}%")) | 
        (Property.address.ilike(f"%{query}%"))
    ).all()
    
    return render_template('search_results.html', properties=properties, query=query)

@app.route('/what-if-analysis')
@login_required
def what_if_analysis():
    """Interactive what-if analysis tool for property valuations."""
    property_id = request.args.get('property_id')
    if not property_id:
        # Without a specific property, redirect to properties list
        return redirect(url_for('properties_list'))
    
    property_data = Property.query.filter_by(property_id=property_id).first()
    if not property_data:
        return render_template('404.html'), 404
    
    # Get latest valuation
    valuation = Valuation.query.filter_by(property_id=property_data.id).order_by(Valuation.valuation_date.desc()).first()
    
    return render_template('what_if_analysis.html', property=property_data, valuation=valuation)

# Proxy API endpoints to avoid CORS issues
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """
    API proxy to forward requests to the API server and return the response.
    This avoids CORS issues by having the frontend and API on the same domain.
    """
    # Build the target URL
    url = f"{API_BASE_URL}/api/{path}"
    
    # Forward the request headers
    headers = {
        key: value for key, value in request.headers if key != 'Host'
    }
    
    # Add API key for authentication
    headers["X-API-KEY"] = API_KEY
    
    try:
        # Forward the request with appropriate method
        if request.method == 'GET':
            resp = requests.get(url, headers=headers, params=request.args)
        elif request.method == 'POST':
            resp = requests.post(url, headers=headers, json=request.json)
        elif request.method == 'PUT':
            resp = requests.put(url, headers=headers, json=request.json)
        elif request.method == 'DELETE':
            resp = requests.delete(url, headers=headers)
        else:
            return jsonify({"error": "Method not supported"}), 405
        
        return jsonify(resp.json()), resp.status_code
    except requests.RequestException:
        return jsonify({"error": "API server not available"}), 503

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists', 'danger')
            return render_template('register.html')
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate form
        if not current_user.check_password(current_password):
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('profile'))
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return redirect(url_for('profile'))
        
        # Update password
        current_user.set_password(new_password)
        db.session.commit()
        
        flash('Password updated successfully', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("You don't have permission to access this page", "danger")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard page"""
    from datetime import datetime
    # Get counts for dashboard
    properties_count = Property.query.count()
    users_count = User.query.count()
    valuations_count = Valuation.query.count()
    gis_features_count = GISFeature.query.count()
    
    return render_template('admin/dashboard.html',
                          properties_count=properties_count,
                          users_count=users_count,
                          valuations_count=valuations_count,
                          gis_features_count=gis_features_count,
                          now=datetime.utcnow())

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500