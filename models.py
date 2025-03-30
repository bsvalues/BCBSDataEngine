from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    properties = db.relationship('Property', back_populates='owner', lazy='dynamic')
    valuations = db.relationship('PropertyValuation', back_populates='requested_by', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Property(db.Model):
    __tablename__ = 'properties'
    
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(200), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(2), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Integer)
    lot_size = db.Column(db.Float)
    year_built = db.Column(db.Integer)
    property_type = db.Column(db.String(50))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    
    # Foreign keys
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    owner = db.relationship('User', back_populates='properties')
    valuations = db.relationship('PropertyValuation', back_populates='property', lazy='dynamic')
    features = db.relationship('PropertyFeature', back_populates='property', lazy='dynamic')
    
    def __repr__(self):
        return f'<Property {self.address}, {self.city}, {self.state}>'


class PropertyFeature(db.Model):
    __tablename__ = 'property_features'
    
    id = db.Column(db.Integer, primary_key=True)
    feature_name = db.Column(db.String(100), nullable=False)
    feature_value = db.Column(db.String(200))
    
    # Foreign keys
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    
    # Relationships
    property = db.relationship('Property', back_populates='features')
    
    def __repr__(self):
        return f'<PropertyFeature {self.feature_name}: {self.feature_value}>'


class PropertyValuation(db.Model):
    __tablename__ = 'property_valuations'
    
    id = db.Column(db.Integer, primary_key=True)
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    estimated_value = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float)
    valuation_method = db.Column(db.String(50), nullable=False)
    valuation_notes = db.Column(db.Text)
    
    # Foreign keys
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    requested_by_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    property = db.relationship('Property', back_populates='valuations')
    requested_by = db.relationship('User', back_populates='valuations')
    
    def __repr__(self):
        return f'<PropertyValuation ${self.estimated_value:,.2f} on {self.valuation_date}>'


class ETLJob(db.Model):
    __tablename__ = 'etl_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    job_type = db.Column(db.String(50), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='running')
    records_processed = db.Column(db.Integer, default=0)
    records_succeeded = db.Column(db.Integer, default=0)
    records_failed = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ETLJob {self.job_type} ({self.status})>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))