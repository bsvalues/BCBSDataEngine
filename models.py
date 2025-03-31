from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    api_key = db.Column(db.String(64), unique=True, index=True)
    
    # Relationships
    properties = db.relationship('Property', back_populates='owner', lazy='dynamic')
    valuations = db.relationship('Valuation', back_populates='user', lazy='dynamic')
    
    def set_password(self, password):
        """Set the user's password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches the user's password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Property(db.Model):
    """Property model representing real estate properties"""
    __tablename__ = 'properties'
    
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(256), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    zip_code = db.Column(db.String(20), nullable=False)
    property_type = db.Column(db.String(50), nullable=False)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Integer)
    lot_size = db.Column(db.Float)
    year_built = db.Column(db.Integer)
    last_sold_date = db.Column(db.Date)
    last_sold_price = db.Column(db.Float)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    neighborhood = db.Column(db.String(100))
    description = db.Column(db.Text)
    features = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    owner = db.relationship('User', back_populates='properties')
    valuations = db.relationship('Valuation', back_populates='property', lazy='dynamic')
    images = db.relationship('PropertyImage', back_populates='property', lazy='dynamic')
    
    def __repr__(self):
        return f'<Property {self.address}, {self.city}, {self.state}>'


class PropertyImage(db.Model):
    """Property images model for storing property photos"""
    __tablename__ = 'property_images'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    path = db.Column(db.String(512), nullable=False)
    description = db.Column(db.String(256))
    is_primary = db.Column(db.Boolean, default=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    
    # Relationships
    property = db.relationship('Property', back_populates='images')
    
    def __repr__(self):
        return f'<PropertyImage {self.filename} for Property {self.property_id}>'


class Valuation(db.Model):
    """Valuation model for storing property valuations"""
    __tablename__ = 'valuations'
    
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    agent_id = db.Column(db.Integer, db.ForeignKey('agents.id'))
    estimated_value = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    valuation_method = db.Column(db.String(100), nullable=False)
    model_version = db.Column(db.String(50))
    adjusted_value = db.Column(db.Float)
    adjustment_factors = db.Column(db.Text)
    comparable_properties = db.Column(db.Text)
    metrics = db.Column(db.Text)
    notes = db.Column(db.Text)
    
    # Relationships
    property = db.relationship('Property', back_populates='valuations')
    user = db.relationship('User', back_populates='valuations')
    agent = db.relationship('Agent', back_populates='valuations')
    
    def __repr__(self):
        return f'<Valuation ${self.estimated_value:.2f} for Property {self.property_id}>'


class Agent(db.Model):
    """Agent model for valuation agents/models"""
    __tablename__ = 'agents'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    agent_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='idle')
    model_version = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime)
    processing_count = db.Column(db.Integer, default=0)
    success_count = db.Column(db.Integer, default=0)
    error_count = db.Column(db.Integer, default=0)
    average_confidence = db.Column(db.Float, default=0.0)
    configuration = db.Column(db.Text)
    
    # Relationships
    valuations = db.relationship('Valuation', back_populates='agent', lazy='dynamic')
    logs = db.relationship('AgentLog', back_populates='agent', lazy='dynamic')
    
    def __repr__(self):
        return f'<Agent {self.name} ({self.agent_type})>'


class AgentLog(db.Model):
    """Agent log model for agent activity logging"""
    __tablename__ = 'agent_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agents.id'), nullable=False)
    log_level = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text)
    
    # Relationships
    agent = db.relationship('Agent', back_populates='logs')
    
    def __repr__(self):
        return f'<AgentLog {self.log_level} for Agent {self.agent_id}>'


class ETLJob(db.Model):
    """ETL job model for tracking data pipeline jobs"""
    __tablename__ = 'etl_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    job_type = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    records_processed = db.Column(db.Integer, default=0)
    total_records = db.Column(db.Integer)
    success_count = db.Column(db.Integer, default=0)
    error_count = db.Column(db.Integer, default=0)
    error_details = db.Column(db.Text)
    configuration = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ETLJob {self.job_type} from {self.source}>'


class ApiKey(db.Model):
    """API key model for API authentication"""
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    last_used = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    permissions = db.Column(db.String(256))
    
    # Relationships
    user = db.relationship('User')
    
    def __repr__(self):
        return f'<ApiKey {self.name} for User {self.user_id}>'


class MarketTrend(db.Model):
    """Market trends model for storing market analytics"""
    __tablename__ = 'market_trends'
    
    id = db.Column(db.Integer, primary_key=True)
    neighborhood = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    trend_date = db.Column(db.Date, nullable=False)
    median_price = db.Column(db.Float)
    average_price = db.Column(db.Float)
    price_per_sqft = db.Column(db.Float)
    inventory_count = db.Column(db.Integer)
    days_on_market = db.Column(db.Integer)
    month_over_month = db.Column(db.Float)
    year_over_year = db.Column(db.Float)
    property_type = db.Column(db.String(50))
    
    def __repr__(self):
        return f'<MarketTrend {self.neighborhood}, {self.city} on {self.trend_date}>'