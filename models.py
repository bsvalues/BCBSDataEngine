from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    """User model for authentication and user management."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256))
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), default='user')  # 'user', 'admin', 'analyst'

    def set_password(self, password):
        """Set the password hash from a plain text password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Property(db.Model):
    """Property model for real estate properties."""
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(256), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(2), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)
    neighborhood = db.Column(db.String(100), index=True)
    property_type = db.Column(db.String(50), index=True)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Integer)
    lot_size = db.Column(db.Float)  # in acres
    year_built = db.Column(db.Integer)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    valuations = db.relationship('Valuation', back_populates='property', lazy='dynamic')
    features = db.relationship('PropertyFeature', back_populates='property', lazy='dynamic')
    
    def __repr__(self):
        return f'<Property {self.address}, {self.city}, {self.state}>'

class Valuation(db.Model):
    """Valuation model for property valuations."""
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    estimated_value = db.Column(db.Integer, nullable=False)  # in dollars
    confidence_score = db.Column(db.Float)  # 0-100 scale
    valuation_method = db.Column(db.String(50))  # 'basic', 'enhanced', 'advanced_gis'
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Advanced metrics
    price_per_sqft = db.Column(db.Float)
    comp_price_range_low = db.Column(db.Integer)
    comp_price_range_high = db.Column(db.Integer)
    market_trend_adjustment = db.Column(db.Float)  # percentage adjustment based on market trends
    location_score = db.Column(db.Float)  # 0-100 scale
    
    # Model metrics 
    regression_r2 = db.Column(db.Float)  # R-squared value for regression models
    mse = db.Column(db.Float)  # Mean squared error
    mae = db.Column(db.Float)  # Mean absolute error
    
    # Relationships
    property = db.relationship('Property', back_populates='valuations')
    factors = db.relationship('ValuationFactor', back_populates='valuation', lazy='dynamic')
    
    def __repr__(self):
        return f'<Valuation ${self.estimated_value} for Property ID: {self.property_id}>'

class ValuationFactor(db.Model):
    """Factors that contribute to a property valuation."""
    id = db.Column(db.Integer, primary_key=True)
    valuation_id = db.Column(db.Integer, db.ForeignKey('valuation.id'), nullable=False)
    factor_name = db.Column(db.String(100), nullable=False)
    factor_value = db.Column(db.Float, nullable=False)  # The factor's numeric value
    factor_weight = db.Column(db.Float, nullable=False)  # The factor's weight in the valuation
    factor_contribution = db.Column(db.Float, nullable=False)  # Dollar amount contribution to valuation
    
    # Relationships
    valuation = db.relationship('Valuation', back_populates='factors')
    
    def __repr__(self):
        return f'<ValuationFactor {self.factor_name}: ${self.factor_contribution}>'

class PropertyFeature(db.Model):
    """Features of a property beyond basic attributes."""
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('property.id'), nullable=False)
    feature_name = db.Column(db.String(100), nullable=False)
    feature_value = db.Column(db.String(100))
    feature_type = db.Column(db.String(50))  # 'categorical', 'numerical', 'boolean'
    
    # Relationships
    property = db.relationship('Property', back_populates='features')
    
    def __repr__(self):
        return f'<PropertyFeature {self.feature_name}: {self.feature_value}>'

class Neighborhood(db.Model):
    """Neighborhood information including GIS-derived metrics."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(2), nullable=False)
    median_price = db.Column(db.Integer)
    price_trend = db.Column(db.Float)  # percentage change year-over-year
    school_rating = db.Column(db.Float)  # 0-10 scale
    crime_rate = db.Column(db.Float)
    walk_score = db.Column(db.Float)  # 0-100 scale
    transit_score = db.Column(db.Float)  # 0-100 scale
    bike_score = db.Column(db.Float)  # 0-100 scale
    boundary_geojson = db.Column(db.Text)  # GeoJSON of neighborhood boundary
    
    def __repr__(self):
        return f'<Neighborhood {self.name}, {self.city}, {self.state}>'

class ETLStatus(db.Model):
    """Status tracking for ETL pipeline processes."""
    id = db.Column(db.Integer, primary_key=True)
    process_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'waiting', 'in_progress', 'completed', 'failed'
    records_processed = db.Column(db.Integer, default=0)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ETLStatus {self.process_name}: {self.status}>'

class AgentStatus(db.Model):
    """Status tracking for system agents."""
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(100), nullable=False, unique=True)
    agent_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'active', 'idle', 'error'
    tasks_completed = db.Column(db.Integer, default=0)
    tasks_failed = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float)
    last_active = db.Column(db.DateTime)
    version = db.Column(db.String(20))
    
    def __repr__(self):
        return f'<AgentStatus {self.agent_name}: {self.status}>'