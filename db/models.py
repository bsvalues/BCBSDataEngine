"""
Database models for the property valuation system.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Property(Base):
    """
    Property model representing real estate properties.
    """
    __tablename__ = 'properties'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
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
