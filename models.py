from app import db
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f"<User {self.username}>"


class Property(db.Model):
    __tablename__ = "properties"
    
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    address = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(2), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)
    bedrooms = db.Column(db.Float, nullable=True)
    bathrooms = db.Column(db.Float, nullable=True)
    square_feet = db.Column(db.Float, nullable=True)
    lot_size = db.Column(db.Float, nullable=True)
    year_built = db.Column(db.Integer, nullable=True)
    property_type = db.Column(db.String(50), nullable=True)
    last_sold_price = db.Column(db.Float, nullable=True)
    last_sold_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Geographic data
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    
    # Relationships
    valuations = db.relationship("Valuation", back_populates="property", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Property {self.property_id}: {self.address}>"


class Valuation(db.Model):
    __tablename__ = "valuations"
    
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey("properties.id"), nullable=False)
    estimated_value = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    model_used = db.Column(db.String(100), nullable=False)
    valuation_date = db.Column(db.DateTime, default=datetime.utcnow)
    features_used = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship back to property
    property = db.relationship("Property", back_populates="valuations")
    
    def __repr__(self):
        return f"<Valuation for {self.property_id}: ${self.estimated_value:,.2f}>"


class GISFeature(db.Model):
    __tablename__ = "gis_features"
    
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey("properties.id"), nullable=False)
    feature_type = db.Column(db.String(100), nullable=False)
    feature_value = db.Column(db.Float, nullable=True)
    feature_name = db.Column(db.String(255), nullable=False)
    feature_description = db.Column(db.Text, nullable=True)
    feature_data = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define relationships if needed here
    
    def __repr__(self):
        return f"<GISFeature {self.feature_type} for property {self.property_id}>"