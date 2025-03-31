from app import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # ensure password hash field has length of at least 256
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    api_keys = db.relationship('ApiKey', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'


class ApiKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(64), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<ApiKey {self.name}>'


class Property(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.String(64), unique=True, nullable=False)
    address = db.Column(db.String(256), nullable=False)
    neighborhood = db.Column(db.String(128), nullable=True)
    property_type = db.Column(db.String(64), nullable=False)
    year_built = db.Column(db.Integer, nullable=True)
    bedrooms = db.Column(db.Integer, nullable=True)
    bathrooms = db.Column(db.Float, nullable=True)
    living_area = db.Column(db.Float, nullable=True)  # in square feet
    land_area = db.Column(db.Float, nullable=True)  # in square feet
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    valuations = db.relationship('PropertyValuation', backref='property', lazy='dynamic')
    
    def __repr__(self):
        return f'<Property {self.property_id} - {self.address}>'


class PropertyValuation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    estimated_value = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    valuation_method = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PropertyValuation ${self.estimated_value:.2f} for Property ID {self.property_id}>'


class EtlStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(64), nullable=False, default='idle')  # idle, processing, completed, error
    progress = db.Column(db.Float, default=0.0)  # 0 to 1.0
    last_update = db.Column(db.DateTime, default=datetime.utcnow)
    records_processed = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, default=0.0)
    average_processing_time = db.Column(db.Float, default=0.0)
    completeness = db.Column(db.Float, default=0.0)
    accuracy = db.Column(db.Float, default=0.0)
    timeliness = db.Column(db.Float, default=0.0)
    
    def __repr__(self):
        return f'<EtlStatus {self.status} - {self.progress:.1%}>'


class DataSource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(64), nullable=False, default='idle')  # idle, queued, processing, completed, error
    records = db.Column(db.Integer, default=0)
    etl_status_id = db.Column(db.Integer, db.ForeignKey('etl_status.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    etl_status = db.relationship('EtlStatus', backref='sources')
    
    def __repr__(self):
        return f'<DataSource {self.name} - {self.status}>'


class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(64), unique=True, nullable=False)
    agent_type = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(64), nullable=False, default='idle')  # active, idle, error
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    queue_size = db.Column(db.Integer, default=0)
    total_processed = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, default=0.0)
    average_processing_time = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    logs = db.relationship('AgentLog', backref='agent', lazy='dynamic')
    
    def __repr__(self):
        return f'<Agent {self.agent_id} - {self.status}>'


class AgentLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    level = db.Column(db.String(16), nullable=False, default='info')  # info, warning, error
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<AgentLog {self.level} - {self.agent.agent_id}>'