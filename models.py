"""
Database models for the BCBS Values application.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Union

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app import db


class Property(db.Model):
    """Model representing a real estate property."""
    __tablename__ = 'properties'
    
    id = Column(Integer, primary_key=True)
    property_id = Column(String(20), unique=True, nullable=False, index=True)
    address = Column(String(255), nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(50), nullable=False)
    zip_code = Column(String(20), nullable=False)
    neighborhood = Column(String(100), index=True)
    property_type = Column(String(50), index=True)
    bedrooms = Column(Integer)
    bathrooms = Column(Float)
    square_feet = Column(Float)
    year_built = Column(Integer)
    lot_size = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    last_sale_date = Column(DateTime)
    last_sale_price = Column(Float)
    estimated_value = Column(Float)
    valuation_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    valuations = relationship("PropertyValuation", back_populates="property", cascade="all, delete-orphan")
    features = relationship("PropertyFeature", back_populates="property", cascade="all, delete-orphan")
    
    @property
    def latest_valuation(self):
        """Get the most recent valuation for this property."""
        if self.valuations:
            return sorted(self.valuations, key=lambda v: v.valuation_date, reverse=True)[0]
        return None


class PropertyValuation(db.Model):
    """Model representing a property valuation."""
    __tablename__ = 'property_valuations'
    
    id = Column(Integer, primary_key=True)
    property_id = Column(Integer, ForeignKey('properties.id'), nullable=False, index=True)
    estimated_value = Column(Float, nullable=False)
    valuation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    valuation_method = Column(String(50), nullable=False)
    confidence_score = Column(Float)
    adj_r2_score = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    inputs = Column(JSON)
    gis_adjustments = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="valuations")


class PropertyFeature(db.Model):
    """Model representing additional property features."""
    __tablename__ = 'property_features'
    
    id = Column(Integer, primary_key=True)
    property_id = Column(Integer, ForeignKey('properties.id'), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float)
    feature_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="features")


class User(db.Model):
    """Model representing a user of the application."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class ETLJob(db.Model):
    """Model representing an ETL job for data ingestion."""
    __tablename__ = 'etl_jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(36), unique=True, nullable=False)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default='pending')  # pending, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    source_info = Column(JSON)


class Agent(db.Model):
    """Model representing a valuation agent (service)."""
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(36), unique=True, nullable=False)
    agent_name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)  # valuation, gis, etc.
    status = Column(String(20), nullable=False, default='idle')  # idle, busy, error, offline
    last_heartbeat = Column(DateTime)
    current_task = Column(String(255))
    queue_size = Column(Integer, default=0)
    success_rate = Column(Float, default=1.0)
    error_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    logs = relationship("AgentLog", back_populates="agent", cascade="all, delete-orphan",
                        order_by="desc(AgentLog.timestamp)")


class AgentLog(db.Model):
    """Model representing logs from a valuation agent."""
    __tablename__ = 'agent_logs'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False, index=True)
    level = Column(String(10), nullable=False, default='info')  # info, warning, error, debug
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    details = Column(JSON)
    
    # Relationships
    agent = relationship("Agent", back_populates="logs")