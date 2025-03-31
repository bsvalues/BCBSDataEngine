from datetime import datetime
from app import db
from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import JSONB

class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    role = db.Column(db.String(20), default='user')  # 'user', 'admin', 'valuation_agent'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    properties = db.relationship('Property', backref='owner', lazy='dynamic')
    valuations = db.relationship('PropertyValuation', backref='created_by', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'


class Property(db.Model):
    """Model for property information."""
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(2), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)
    neighborhood = db.Column(db.String(100), index=True)
    property_type = db.Column(db.String(50), nullable=False, index=True)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Float)
    year_built = db.Column(db.Integer)
    lot_size = db.Column(db.Float)  # in acres
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    last_sale_price = db.Column(db.Float)
    last_sale_date = db.Column(db.Date)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = db.Column(db.Text)
    # GIS and other derived data
    gis_features = db.Column(JSONB)
    # Relationships
    valuations = db.relationship('PropertyValuation', backref='property', lazy='dynamic')
    features = db.relationship('PropertyFeature', backref='property', lazy='dynamic')
    
    def __repr__(self):
        return f'<Property {self.address}, {self.city}, {self.state}>'
    
    @property
    def full_address(self):
        return f"{self.address}, {self.city}, {self.state} {self.zip_code}"


class PropertyValuation(db.Model):
    """Model for property valuation records."""
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    estimated_value = db.Column(db.Float, nullable=False)
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    valuation_method = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float)  # 0.0 to 1.0
    model_features = db.Column(JSONB)  # Store the input features used
    comparable_properties = db.Column(JSONB)  # Store comparable properties used
    market_trends = db.Column(JSONB)  # Store market trend data
    gis_features = db.Column(JSONB)  # GIS feature impacts on valuation
    etl_job_id = db.Column(db.Integer, db.ForeignKey('etl_job.id'))
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'))
    
    def __repr__(self):
        return f'<PropertyValuation {self.property_id} - ${self.estimated_value:,.2f}>'


class PropertyFeature(db.Model):
    """Model for additional property features."""
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    feature_type = db.Column(db.String(50), nullable=False)  # e.g., 'pool', 'garage', 'fireplace'
    feature_value = db.Column(db.String(100))  # e.g., size, count, yes/no
    feature_description = db.Column(db.Text)
    
    def __repr__(self):
        return f'<PropertyFeature {self.property_id} - {self.feature_type}>'


class ETLJob(db.Model):
    """Model for ETL job tracking."""
    id = db.Column(db.Integer, primary_key=True)
    job_type = db.Column(db.String(50), nullable=False)  # e.g., 'property_import', 'market_data_update'
    status = db.Column(db.String(20), nullable=False, default='pending')  # 'pending', 'running', 'completed', 'failed'
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    progress = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    records_processed = db.Column(db.Integer, default=0)
    records_total = db.Column(db.Integer, default=0)
    source = db.Column(db.String(255))
    message = db.Column(db.Text)
    error = db.Column(db.Text)
    # Relationships
    valuations = db.relationship('PropertyValuation', backref='etl_job', lazy='dynamic')
    
    def __repr__(self):
        return f'<ETLJob {self.job_type} - {self.status}>'


class Agent(db.Model):
    """Model for valuation agents."""
    id = db.Column(db.Integer, primary_key=True)
    agent_type = db.Column(db.String(50), nullable=False)  # e.g., 'regression', 'ensemble', 'gis'
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='idle')  # 'idle', 'processing', 'error'
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    version = db.Column(db.String(20), default='1.0.0')
    success_rate = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    api_key = db.Column(db.String(64))
    configuration = db.Column(JSONB)
    # Relationships
    valuations = db.relationship('PropertyValuation', backref='agent', lazy='dynamic')
    logs = db.relationship('AgentLog', backref='agent', lazy='dynamic')
    
    def __repr__(self):
        return f'<Agent {self.name} ({self.agent_type})>'


class AgentLog(db.Model):
    """Model for agent logs."""
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    level = db.Column(db.String(10), default='info')  # 'debug', 'info', 'warning', 'error', 'critical'
    message = db.Column(db.Text, nullable=False)
    details = db.Column(JSONB)
    
    def __repr__(self):
        return f'<AgentLog {self.agent_id} - {self.level}>'