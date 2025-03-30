from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.urls import url_parse
from app import db
from models import User, Property, PropertyValuation, ETLJob
from forms import LoginForm, RegistrationForm, PropertyForm, ValuationRequestForm, ProfileForm


# Create blueprints
main_bp = Blueprint('main', __name__)
auth_bp = Blueprint('auth', __name__)
admin_bp = Blueprint('admin', __name__)


# Main routes
@main_bp.route('/')
def index():
    return render_template('index.html', title='Home')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Get user's properties
    user_properties = Property.query.filter_by(owner_id=current_user.id).all()
    
    # Get recent valuations
    recent_valuations = PropertyValuation.query.filter_by(
        requested_by_id=current_user.id
    ).order_by(PropertyValuation.valuation_date.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                           title='Dashboard',
                           properties=user_properties,
                           valuations=recent_valuations)


@main_bp.route('/properties')
@login_required
def properties():
    user_properties = Property.query.filter_by(owner_id=current_user.id).all()
    return render_template('properties.html', 
                           title='My Properties',
                           properties=user_properties)


@main_bp.route('/properties/add', methods=['GET', 'POST'])
@login_required
def add_property():
    form = PropertyForm()
    if form.validate_on_submit():
        property = Property(
            address=form.address.data,
            city=form.city.data,
            state=form.state.data,
            zip_code=form.zip_code.data,
            bedrooms=form.bedrooms.data,
            bathrooms=form.bathrooms.data,
            square_feet=form.square_feet.data,
            lot_size=form.lot_size.data,
            year_built=form.year_built.data,
            property_type=form.property_type.data,
            latitude=form.latitude.data,
            longitude=form.longitude.data,
            owner_id=current_user.id
        )
        db.session.add(property)
        db.session.commit()
        flash('Property added successfully!', 'success')
        return redirect(url_for('main.properties'))
    return render_template('property_form.html', 
                           title='Add Property',
                           form=form)


@main_bp.route('/properties/<int:property_id>')
@login_required
def property_detail(property_id):
    property = Property.query.get_or_404(property_id)
    if property.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this property.', 'danger')
        return redirect(url_for('main.properties'))
    
    valuations = PropertyValuation.query.filter_by(property_id=property.id).order_by(
        PropertyValuation.valuation_date.desc()).all()
    
    return render_template('property_detail.html',
                           title=f'Property Details: {property.address}',
                           property=property,
                           valuations=valuations)


@main_bp.route('/valuations/request/<int:property_id>', methods=['GET', 'POST'])
@login_required
def request_valuation(property_id):
    property = Property.query.get_or_404(property_id)
    if property.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to request a valuation for this property.', 'danger')
        return redirect(url_for('main.properties'))
    
    form = ValuationRequestForm()
    if form.validate_on_submit():
        # Here we would integrate with the valuation engine
        # For now, we'll just create a sample valuation
        valuation = PropertyValuation(
            property_id=property.id,
            requested_by_id=current_user.id,
            estimated_value=350000.00,  # This would come from the valuation engine
            confidence_score=0.85,
            valuation_method=form.valuation_method.data,
            valuation_notes=form.notes.data
        )
        db.session.add(valuation)
        db.session.commit()
        flash('Valuation request submitted successfully!', 'success')
        return redirect(url_for('main.property_detail', property_id=property.id))
    
    return render_template('valuation_request.html',
                           title=f'Request Valuation: {property.address}',
                           property=property,
                           form=form)


@main_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm()
    if form.validate_on_submit():
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('main.profile'))
    elif request.method == 'GET':
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
        form.username.data = current_user.username
    
    return render_template('profile.html', title='Profile', form=form)


# Authentication routes
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.dashboard')
        return redirect(next_page)
    
    return render_template('login.html', title='Log In', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html', title='Register', form=form)


# Admin routes
@admin_bp.route('/')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('You do not have permission to access the admin dashboard.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    users_count = User.query.count()
    properties_count = Property.query.count()
    valuations_count = PropertyValuation.query.count()
    etl_jobs = ETLJob.query.order_by(ETLJob.start_time.desc()).limit(5).all()
    
    return render_template('admin/dashboard.html',
                           title='Admin Dashboard',
                           users_count=users_count,
                           properties_count=properties_count,
                           valuations_count=valuations_count,
                           etl_jobs=etl_jobs)


@admin_bp.route('/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    users = User.query.all()
    return render_template('admin/users.html', title='User Management', users=users)


@admin_bp.route('/properties')
@login_required
def admin_properties():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    properties = Property.query.all()
    return render_template('admin/properties.html', title='Property Management', properties=properties)


@admin_bp.route('/etl-jobs')
@login_required
def admin_etl_jobs():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    etl_jobs = ETLJob.query.order_by(ETLJob.start_time.desc()).all()
    return render_template('admin/etl_jobs.html', title='ETL Job Management', etl_jobs=etl_jobs)