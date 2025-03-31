from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user, login_user, logout_user
from models import User, Property, Valuation, Neighborhood

# Create a blueprint for the main routes
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Home page route."""
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard route."""
    return render_template('reactive_dashboard.html')

@main_bp.route('/properties')
def properties():
    """Properties listing route."""
    # This would typically fetch properties from the database
    # For now, we'll just render a template
    return render_template('properties.html')

@main_bp.route('/property/<int:property_id>')
def property_detail(property_id):
    """Property detail route."""
    # This would typically fetch a specific property from the database
    # For now, we'll just render a template with the property_id
    return render_template('property_detail.html', property_id=property_id)

@main_bp.route('/analysis')
def analysis():
    """Analysis route."""
    return render_template('analysis.html')

@main_bp.route('/about')
def about():
    """About page route."""
    return render_template('about.html')

# Authentication routes
@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login route."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=request.form.get('remember', False))
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@main_bp.route('/logout')
@login_required
def logout():
    """Logout route."""
    logout_user()
    return redirect(url_for('main.index'))

@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    # Registration form handling would go here
    return render_template('register.html')

@main_bp.route('/user/profile')
@login_required
def user_profile():
    """User profile route."""
    return render_template('user_profile.html')

# API Documentation route
@main_bp.route('/api-docs')
def api_docs():
    """API documentation route."""
    return render_template('api_docs.html')

# Error handlers
@main_bp.app_errorhandler(404)
def page_not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@main_bp.app_errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500