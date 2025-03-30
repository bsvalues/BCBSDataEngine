from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_admin = db.Column(db.Boolean, default=False)
    organization = db.Column(db.String(128))
    
    # Relationships
    property_searches = db.relationship('PropertySearch', back_populates='user', lazy='dynamic')
    valuations = db.relationship('Valuation', back_populates='user', lazy='dynamic')
    saved_properties = db.relationship('SavedProperty', back_populates='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Property(db.Model):
    __tablename__ = 'properties'
    
    id = db.Column(db.Integer, primary_key=True)
    parcel_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    address = db.Column(db.String(256), nullable=False, index=True)
    city = db.Column(db.String(64), nullable=False)
    state = db.Column(db.String(32), nullable=False)
    zip_code = db.Column(db.String(16), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Integer)
    year_built = db.Column(db.Integer)
    lot_size = db.Column(db.Float)  # In acres
    property_type = db.Column(db.String(32))
    neighborhood = db.Column(db.String(64), index=True)
    school_district = db.Column(db.String(64))
    last_sale_date = db.Column(db.Date)
    last_sale_price = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Data source tracking
    source = db.Column(db.String(16))  # 'PACS', 'MLS', 'NARRPR', etc.
    source_id = db.Column(db.String(64))
    
    # Relationships
    valuations = db.relationship('Valuation', back_populates='property', lazy='dynamic')
    gis_features = db.relationship('GISFeature', back_populates='property', uselist=False)
    saved_by = db.relationship('SavedProperty', back_populates='property', lazy='dynamic')
    
    def __repr__(self):
        return f'<Property {self.address}>'


class GISFeature(db.Model):
    __tablename__ = 'gis_features'
    
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False, unique=True)
    
    # Proximity scores (0-100)
    proximity_schools = db.Column(db.Integer)
    proximity_parks = db.Column(db.Integer)
    proximity_shopping = db.Column(db.Integer)
    proximity_healthcare = db.Column(db.Integer)
    proximity_transit = db.Column(db.Integer)
    
    # Neighborhood metrics
    neighborhood_quality = db.Column(db.Float)  # Composite score (0-10)
    walkability_score = db.Column(db.Integer)  # (0-100)
    flood_risk = db.Column(db.Float)  # (0-10, 10 being highest risk)
    
    # Environmental factors
    elevation = db.Column(db.Float)  # In feet
    slope = db.Column(db.Float)  # In degrees
    
    # GIS metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_source = db.Column(db.String(64))
    
    # Relationship
    property = db.relationship('Property', back_populates='gis_features')
    
    def __repr__(self):
        return f'<GISFeature for Property #{self.property_id}>'


class Valuation(db.Model):
    __tablename__ = 'valuations'
    
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Valuation results
    estimated_value = db.Column(db.Integer, nullable=False)  # In dollars
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    confidence_score = db.Column(db.Float)  # (0-1)
    
    # Model details
    model_type = db.Column(db.String(32), nullable=False)  # 'basic', 'advanced', 'gis_enhanced'
    model_version = db.Column(db.String(32))
    
    # Advanced metrics
    comp_low = db.Column(db.Integer)  # Low end of comparable range
    comp_high = db.Column(db.Integer)  # High end of comparable range
    price_per_sqft = db.Column(db.Float)
    
    # Adjustments applied
    adjustment_location = db.Column(db.Float, default=0.0)
    adjustment_condition = db.Column(db.Float, default=0.0)
    adjustment_size = db.Column(db.Float, default=0.0)
    adjustment_features = db.Column(db.Float, default=0.0)
    adjustment_market = db.Column(db.Float, default=0.0)
    
    # Relationships
    property = db.relationship('Property', back_populates='valuations')
    user = db.relationship('User', back_populates='valuations')
    
    def __repr__(self):
        return f'<Valuation ${self.estimated_value} for {self.property.address}>'


class PropertySearch(db.Model):
    __tablename__ = 'property_searches'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    search_query = db.Column(db.String(256), nullable=False)
    search_date = db.Column(db.DateTime, default=datetime.utcnow)
    search_params = db.Column(db.JSON)  # Store filter parameters
    
    # Relationship
    user = db.relationship('User', back_populates='property_searches')
    
    def __repr__(self):
        return f'<PropertySearch {self.search_query}>'


class SavedProperty(db.Model):
    __tablename__ = 'saved_properties'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    saved_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    
    # Define a unique constraint to prevent duplicate saves
    __table_args__ = (db.UniqueConstraint('user_id', 'property_id', name='unique_user_property'),)
    
    # Relationships
    user = db.relationship('User', back_populates='saved_properties')
    property = db.relationship('Property', back_populates='saved_by')
    
    def __repr__(self):
        return f'<SavedProperty {self.property.address} by {self.user.username}>'


class ETLJob(db.Model):
    __tablename__ = 'etl_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(32), nullable=False)  # 'PACS', 'MLS', 'NARRPR', etc.
    status = db.Column(db.String(32), nullable=False)  # 'pending', 'running', 'success', 'error'
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    records_processed = db.Column(db.Integer, default=0)
    records_imported = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ETLJob {self.source} {self.status}>'