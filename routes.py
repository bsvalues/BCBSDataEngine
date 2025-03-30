import logging
from datetime import datetime

from flask import render_template, redirect, url_for, flash, request, abort
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from app import db
from models import User, Property, Valuation, GISFeature, PropertySearch, SavedProperty, ETLJob


# Set up logging
logger = logging.getLogger(__name__)


def register_routes(app):
    #----------------------------------------------------------
    # Authentication Routes
    #----------------------------------------------------------
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = 'remember' in request.form
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'danger')
        
        return render_template('login.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out', 'info')
        return redirect(url_for('index'))
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            organization = request.form.get('organization')
            
            # Check if passwords match
            if password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('register.html')
            
            # Check if username exists
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'danger')
                return render_template('register.html')
            
            # Check if email exists
            if User.query.filter_by(email=email).first():
                flash('Email already exists', 'danger')
                return render_template('register.html')
            
            # Create new user
            new_user = User(username=username, email=email, organization=organization)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html')
    
    @app.route('/forgot-password', methods=['GET', 'POST'])
    def forgot_password():
        # For now, this is a placeholder
        return render_template('login.html')
    
    
    #----------------------------------------------------------
    # User Profile Routes
    #----------------------------------------------------------
    @app.route('/profile', methods=['GET', 'POST'])
    @login_required
    def profile():
        if request.method == 'POST':
            # Handle profile update
            if 'update_profile' in request.form:
                email = request.form.get('email')
                
                # Check if email changed and is not already taken
                if email != current_user.email and User.query.filter_by(email=email).first():
                    flash('Email already exists', 'danger')
                else:
                    current_user.email = email
                    db.session.commit()
                    flash('Profile updated successfully', 'success')
            
            # Handle password change
            elif 'change_password' in request.form:
                current_password = request.form.get('current_password')
                new_password = request.form.get('new_password')
                confirm_new_password = request.form.get('confirm_new_password')
                
                if not current_user.check_password(current_password):
                    flash('Current password is incorrect', 'danger')
                elif new_password != confirm_new_password:
                    flash('New passwords do not match', 'danger')
                else:
                    current_user.set_password(new_password)
                    db.session.commit()
                    flash('Password changed successfully', 'success')
        
        return render_template('profile.html')
    
    
    #----------------------------------------------------------
    # Main Application Routes
    #----------------------------------------------------------
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        # Get user's recent valuations
        recent_valuations = Valuation.query.filter_by(user_id=current_user.id).order_by(Valuation.valuation_date.desc()).limit(5).all()
        
        # Get user's saved properties
        saved_properties = SavedProperty.query.filter_by(user_id=current_user.id).order_by(SavedProperty.saved_at.desc()).limit(5).all()
        
        return render_template('dashboard.html', 
                               recent_valuations=recent_valuations,
                               saved_properties=saved_properties)
    
    @app.route('/properties')
    @login_required
    def properties():
        # For now, this is a placeholder
        return render_template('dashboard.html')
    
    @app.route('/valuation', methods=['GET', 'POST'])
    @login_required
    def valuation():
        # For now, this is a placeholder
        return render_template('dashboard.html')
    
    @app.route('/search')
    def search():
        # For now, this is a placeholder
        return render_template('index.html')
    
    
    #----------------------------------------------------------
    # Admin Routes
    #----------------------------------------------------------
    @app.route('/admin/dashboard')
    @login_required
    def admin_dashboard():
        if not current_user.is_admin:
            abort(403)
        
        # Get all users
        users = User.query.all()
        
        # Get property count
        properties = Property.query.count()
        
        # Get valuation count
        valuations = Valuation.query.count()
        
        # Get recent ETL jobs
        recent_etl = ETLJob.query.order_by(ETLJob.start_time.desc()).limit(10).all()
        
        return render_template('admin/dashboard.html', 
                               users=users, 
                               properties=properties,
                               valuations=valuations,
                               recent_etl=recent_etl)
    
    @app.route('/admin/users')
    @login_required
    def admin_users():
        if not current_user.is_admin:
            abort(403)
        
        # For now, this is a placeholder
        return render_template('admin/dashboard.html')
    
    @app.route('/admin/properties')
    @login_required
    def admin_properties():
        if not current_user.is_admin:
            abort(403)
        
        # For now, this is a placeholder
        return render_template('admin/dashboard.html')
    
    @app.route('/admin/etl')
    @login_required
    def admin_etl():
        if not current_user.is_admin:
            abort(403)
        
        # For now, this is a placeholder
        return render_template('admin/dashboard.html')
    
    
    #----------------------------------------------------------
    # Static Pages
    #----------------------------------------------------------
    @app.route('/about')
    def about():
        # For now, this is a placeholder
        return render_template('index.html')
    
    @app.route('/services')
    def services():
        # For now, this is a placeholder
        return render_template('index.html')
    
    @app.route('/contact')
    def contact():
        # For now, this is a placeholder
        return render_template('index.html')
    
    
    #----------------------------------------------------------
    # Error Handlers
    #----------------------------------------------------------
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template('403.html'), 403
    
    
    logger.info("All routes registered successfully")