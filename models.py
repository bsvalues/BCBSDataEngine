"""
Database models for the property valuation system.

This module uses SQLAlchemy declarative base from db package to ensure
consistency between the Flask app and FastAPI.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

# Import Base from db package
from db import Base

class Property(Base):
    """
    Property model representing real estate properties.
    """
    __tablename__ = 'properties'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Relationships
    valuations = relationship("PropertyValuation", backref="property", cascade="all, delete-orphan")
    
    # Identifiers
    mls_id = Column(String(50))
    listing_id = Column(String(50))
    property_id = Column(String(50))
    parcel_id = Column(String(50))
    apn = Column(String(50))
    
    # Location information
    address = Column(String(255))
    city = Column(String(100))
    county = Column(String(100))
    state = Column(String(50))
    zip_code = Column(String(20))
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Property characteristics
    property_type = Column(String(50))
    bedrooms = Column(Float)
    bathrooms = Column(Float)
    total_rooms = Column(Integer)
    square_feet = Column(Float)
    lot_size = Column(Float)
    year_built = Column(Integer)
    stories = Column(Float)
    basement = Column(String(50))
    garage = Column(String(50))
    garage_spaces = Column(Integer)
    pool = Column(String(10))
    view = Column(String(50))
    construction_type = Column(String(50))
    roof_type = Column(String(50))
    foundation_type = Column(String(50))
    
    # Valuation information
    list_price = Column(Float)
    estimated_value = Column(Float)
    last_sale_price = Column(Float)
    last_sale_date = Column(DateTime)
    land_value = Column(Float)
    improvement_value = Column(Float)
    total_value = Column(Float)
    assessment_year = Column(Integer)
    
    # Listing information
    listing_date = Column(DateTime)
    status = Column(String(50))
    days_on_market = Column(Integer)
    listing_agent = Column(String(100))
    listing_office = Column(String(100))
    
    # Data source and import information
    data_source = Column(String(50))
    import_date = Column(DateTime)
    
    def __repr__(self):
        """String representation of a Property object."""
        return f"<Property(id={self.id}, address='{self.address}', city='{self.city}', state='{self.state}')>"


class ValidationResult(Base):
    """
    Validation result model for storing data validation history.
    """
    __tablename__ = 'validation_results'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Timestamps
    timestamp = Column(DateTime)
    
    # Validation status and results
    status = Column(String(20))
    results = Column(Text)
    
    def __repr__(self):
        """String representation of a ValidationResult object."""
        return f"<ValidationResult(id={self.id}, timestamp='{self.timestamp}', status='{self.status}')>"


class PropertyValuation(Base):
    """
    Property valuation model for storing valuation results from the various models.
    """
    __tablename__ = 'property_valuations'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Foreign key to Property
    property_id = Column(Integer, ForeignKey('properties.id'))
    
    # Valuation timestamp
    valuation_date = Column(DateTime)
    
    # Core valuation data
    estimated_value = Column(Float)
    confidence_score = Column(Float)
    prediction_interval_low = Column(Float)
    prediction_interval_high = Column(Float)
    
    # Model metadata
    model_name = Column(String(100))
    model_version = Column(String(50))
    model_r2_score = Column(Float)
    
    # Feature importance and contribution data
    feature_importance = Column(JSON)  # Stores feature importance as JSON
    top_features = Column(Text)        # Comma-separated list of top features
    
    # Comparables used
    comparable_properties = Column(JSON)  # IDs of comparable properties used in valuation
    
    # Valuation factors
    location_factor = Column(Float)    # How much location contributed to value
    size_factor = Column(Float)        # How much size contributed to value
    condition_factor = Column(Float)   # How much condition contributed to value
    market_factor = Column(Float)      # How much market trends contributed to value
    
    # Raw model outputs
    raw_model_outputs = Column(JSON)   # Additional model-specific outputs
    
    def __repr__(self):
        """String representation of a PropertyValuation object."""
        return f"<PropertyValuation(id={self.id}, property_id={self.property_id}, value=${self.estimated_value:,.2f}, date='{self.valuation_date}')>"

# Function to initialize models with a Flask-SQLAlchemy instance if needed
def init_models(database):
    """
    Initialize models for use with Flask-SQLAlchemy.
    This is only used for Flask app integration.
    
    Returns:
        tuple: Property, ValidationResult, PropertyValuation classes
    """
    return Property, ValidationResult, PropertyValuation