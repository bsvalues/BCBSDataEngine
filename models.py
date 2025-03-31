"""
Database models for the BCBS Values application.

This module defines the database schema for the application.
"""
from datetime import datetime
from app import db


class Property(db.Model):
    """
    Property model representing real estate properties.
    """
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.String(32), unique=True, nullable=False)
    address = db.Column(db.String(128), nullable=False)
    city = db.Column(db.String(64), nullable=False)
    state = db.Column(db.String(2), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)
    neighborhood = db.Column(db.String(64))
    property_type = db.Column(db.String(32))
    year_built = db.Column(db.Integer)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Integer)
    lot_size = db.Column(db.Integer)
    last_sale_date = db.Column(db.Date)
    last_sale_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    valuations = db.relationship('PropertyValuation', back_populates='property', order_by='desc(PropertyValuation.valuation_date)', cascade='all, delete-orphan')
    features = db.relationship('PropertyFeature', back_populates='property', cascade='all, delete-orphan')
    
    @property
    def estimated_value(self):
        """Return the latest estimated value for the property."""
        if self.valuations:
            return self.valuations[0].estimated_value
        return None
    
    @property
    def valuation_date(self):
        """Return the date of the latest valuation."""
        if self.valuations:
            return self.valuations[0].valuation_date
        return None
    
    @property
    def latest_valuation(self):
        """Return the latest valuation object."""
        if self.valuations:
            return self.valuations[0]
        return None
    
    def __repr__(self):
        return f"<Property {self.property_id} - {self.address}>"


class PropertyValuation(db.Model):
    """
    Property valuation model representing the estimated value of a property.
    """
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    estimated_value = db.Column(db.Float, nullable=False)
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    valuation_method = db.Column(db.String(32))
    confidence_score = db.Column(db.Float)
    inputs = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    property = db.relationship('Property', back_populates='valuations')
    
    def __repr__(self):
        return f"<PropertyValuation {self.id} - {self.estimated_value}>"


class PropertyFeature(db.Model):
    """
    Property feature model representing additional features of a property.
    """
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    feature_name = db.Column(db.String(64), nullable=False)
    feature_value = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    property = db.relationship('Property', back_populates='features')
    
    def __repr__(self):
        return f"<PropertyFeature {self.id} - {self.feature_name}: {self.feature_value}>"


class ETLJob(db.Model):
    """
    ETL job model representing a data extraction, transformation, and loading job.
    """
    id = db.Column(db.Integer, primary_key=True)
    job_name = db.Column(db.String(64), nullable=False)
    job_type = db.Column(db.String(32), nullable=False)
    source = db.Column(db.String(64))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(16), default='running')
    records_processed = db.Column(db.Integer, default=0)
    records_failed = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f"<ETLJob {self.id} - {self.job_name}>"


class Agent(db.Model):
    """
    Agent model representing an autonomous agent in the system.
    """
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(36), unique=True, nullable=False)
    agent_name = db.Column(db.String(64), nullable=False)
    agent_type = db.Column(db.String(32), nullable=False)
    status = db.Column(db.String(16), default='idle')
    last_heartbeat = db.Column(db.DateTime, default=datetime.utcnow)
    current_task = db.Column(db.String(128))
    queue_size = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, default=100.0)
    error_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    logs = db.relationship('AgentLog', back_populates='agent', order_by='desc(AgentLog.timestamp)', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Agent {self.agent_id} - {self.agent_name}>"


class AgentLog(db.Model):
    """
    Agent log model representing a log entry from an agent.
    """
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    level = db.Column(db.String(16), default='info')
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    agent = db.relationship('Agent', back_populates='logs')
    
    def __repr__(self):
        return f"<AgentLog {self.id} - {self.level}: {self.message[:30]}>"


# User model if using traditional authentication
# For now, we're using Replit's built-in authentication