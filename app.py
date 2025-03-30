"""
Main Flask application for the BCBS_Values real estate valuation system.
This provides a web interface for users to interact with the valuation system.
"""
import os
import json
import datetime
from pathlib import Path
import logging
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs_values_dev_key")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Initialize models using our db instance
from models import init_models
Property, ValidationResult, PropertyValuation = init_models(db)

# Create all database tables if they don't exist yet
with app.app_context():
    db.create_all()

# ETL status data
ETL_STATUS = {
    "status": "completed",
    "last_run": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(),
    "sources_processed": [
        {"name": "MLS", "status": "success", "records": 1250},
        {"name": "NARRPR", "status": "success", "records": 875},
        {"name": "PACS", "status": "warning", "records": 432}
    ],
    "records_processed": 2557,
    "validation_status": "passed_with_warnings",
    "validation_details": {
        "completeness": {"status": "passed", "score": 98.2},
        "data_types": {"status": "passed", "score": 100.0},
        "numeric_ranges": {"status": "warning", "issues": 17},
        "dates": {"status": "passed", "score": 99.5},
        "duplicates": {"status": "warning", "issues": 5},
        "cross_source": {"status": "passed", "score": 97.8}
    },
    "errors": [
        {
            "source": "PACS",
            "error_type": "validation_warning",
            "message": "15 properties have lot_size outside expected range",
            "severity": "warning"
        },
        {
            "source": "MLS",
            "error_type": "validation_warning",
            "message": "5 properties have duplicate parcel IDs",
            "severity": "warning"
        }
    ]
}

# Helper function to convert property DB models to dictionary
def property_to_dict(property_obj):
    """Convert a Property object to a dictionary for API responses"""
    return {
        "id": property_obj.id,
        "property_id": property_obj.property_id,
        "address": property_obj.address,
        "city": property_obj.city,
        "county": property_obj.county,
        "state": property_obj.state,
        "zip_code": property_obj.zip_code,
        "bedrooms": property_obj.bedrooms,
        "bathrooms": property_obj.bathrooms,
        "square_feet": property_obj.square_feet,
        "lot_size": property_obj.lot_size,
        "year_built": property_obj.year_built,
        "estimated_value": property_obj.estimated_value,
        "last_sale_price": property_obj.last_sale_price,
        "last_sale_date": property_obj.last_sale_date.isoformat() if property_obj.last_sale_date else None,
        "property_type": property_obj.property_type,
        "data_source": property_obj.data_source
    }

# Register routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/properties')
def properties():
    """Render the properties page with data from the database."""
    # Get properties from the database
    props = Property.query.filter_by(county='Benton', state='WA').limit(20).all()
    
    # Convert Property objects to dictionaries for the template
    property_list = [property_to_dict(p) for p in props]
    
    return render_template('properties.html', properties=property_list)

@app.route('/property/<int:id>')
def property_detail(id):
    """Render property detail page with valuation."""
    # Get property from the database by ID
    property_obj = Property.query.get(id)
    
    if property_obj:
        # Convert Property object to dictionary
        property_data = property_to_dict(property_obj)
        
        # Get any valuations for this property
        valuations = PropertyValuation.query.filter_by(property_id=id).order_by(PropertyValuation.valuation_date.desc()).first()
        
        # Add valuation data if available
        if valuations:
            property_data["confidence_score"] = valuations.confidence_score
            property_data["valuation_date"] = valuations.valuation_date.isoformat() if valuations.valuation_date else None
            property_data["model_used"] = valuations.model_name
            property_data["prediction_interval"] = {
                "low": valuations.prediction_interval_low,
                "high": valuations.prediction_interval_high
            }
            property_data["features_used"] = {}
            if valuations.feature_importance:
                property_data["features_used"] = valuations.feature_importance
            
            # Add comparable properties if available
            if valuations.comparable_properties:
                property_data["comparable_properties"] = valuations.comparable_properties
        
        return render_template('property_detail.html', property=property_data)
    else:
        return render_template('404.html'), 404

@app.route('/property/by-property-id/<property_id>')
def property_by_property_id(property_id):
    """Find property by its property_id field and redirect to the detail page."""
    property_obj = Property.query.filter_by(property_id=property_id).first()
    
    if property_obj:
        return property_detail(property_obj.id)
    else:
        return render_template('404.html'), 404

@app.route('/validation')
def validation():
    """Render the validation results page."""
    # Get the most recent validation result from the database
    validation_result = ValidationResult.query.order_by(ValidationResult.timestamp.desc()).first()
    
    if validation_result and validation_result.results:
        try:
            validation_data = json.loads(validation_result.results)
            return render_template('validation.html', validation_results=validation_data)
        except json.JSONDecodeError:
            logger.error("Error decoding validation results JSON")
    
    # Fall back to sample data if no validation results in DB
    return render_template('validation.html', validation_results=ETL_STATUS)

@app.route('/what-if-analysis')
def what_if_analysis():
    """Render the What-If Analysis page."""
    # Get a random property to use for the analysis
    property_obj = Property.query.filter(Property.estimated_value != None).first()
    
    if not property_obj:
        return render_template('404.html', message="No properties with valuation data found. Please add property data first."), 404
    
    # Pass the property and its valuation to the template
    return render_template('what_if_analysis.html', 
                          property=property_obj, 
                          original_valuation=property_obj.estimated_value or 0)

@app.route('/search')
def search():
    """Handle property search."""
    query = request.args.get('query', '')
    results = []
    
    if query:
        # Search the database for properties matching the query
        search_term = f"%{query}%"
        results_query = Property.query.filter(
            (Property.address.ilike(search_term)) |
            (Property.city.ilike(search_term)) |
            (Property.property_id.ilike(search_term))
        ).limit(50)
        
        # Convert results to dictionaries
        results = [property_to_dict(p) for p in results_query]
    
    return render_template('search_results.html', 
                          query=query, 
                          results=results)

# API Routes
@app.route('/api/properties')
def api_properties():
    """API endpoint to get properties from the database."""
    # Get query parameters for filtering
    limit = request.args.get('limit', 20, type=int)
    min_value = request.args.get('min_value', None, type=float)
    max_value = request.args.get('max_value', None, type=float)
    property_type = request.args.get('property_type', None)
    
    # Start with a base query
    query = Property.query.filter_by(county='Benton', state='WA')
    
    # Apply filters
    if min_value is not None:
        query = query.filter(Property.estimated_value >= min_value)
    if max_value is not None:
        query = query.filter(Property.estimated_value <= max_value)
    if property_type is not None:
        query = query.filter(Property.property_type == property_type)
    
    # Get results and convert to list of dictionaries
    properties = [property_to_dict(p) for p in query.limit(limit).all()]
    
    return jsonify(properties)

@app.route('/api/property/<int:id>')
def api_property(id):
    """API endpoint to get property details by database ID."""
    # Get property from the database
    property_obj = Property.query.get(id)
    
    if property_obj:
        # Convert to dictionary
        property_data = property_to_dict(property_obj)
        
        # Get valuation data if available
        valuation = PropertyValuation.query.filter_by(property_id=id).order_by(PropertyValuation.valuation_date.desc()).first()
        if valuation:
            property_data["valuation"] = {
                "estimated_value": valuation.estimated_value,
                "confidence_score": valuation.confidence_score,
                "valuation_date": valuation.valuation_date.isoformat() if valuation.valuation_date else None,
                "model_name": valuation.model_name,
                "model_version": valuation.model_version,
                "prediction_interval": {
                    "low": valuation.prediction_interval_low,
                    "high": valuation.prediction_interval_high
                }
            }
        
        return jsonify(property_data)
    else:
        return jsonify({"error": "Property not found"}), 404

@app.route('/api/property/by-property-id/<property_id>')
def api_property_by_property_id(property_id):
    """API endpoint to get property details by property_id field."""
    # Find property by property_id
    property_obj = Property.query.filter_by(property_id=property_id).first()
    
    if property_obj:
        return api_property(property_obj.id)
    else:
        return jsonify({"error": "Property not found"}), 404

@app.route('/api/validation')
def api_validation():
    """API endpoint for validation results."""
    # Get the most recent validation result
    validation_result = ValidationResult.query.order_by(ValidationResult.timestamp.desc()).first()
    
    if validation_result and validation_result.results:
        try:
            validation_data = json.loads(validation_result.results)
            return jsonify(validation_data)
        except json.JSONDecodeError:
            logger.error("Error decoding validation results JSON")
    
    # Fall back to sample data if no validation results in DB
    return jsonify(ETL_STATUS)

@app.route('/api/what-if-analysis', methods=['POST'])
def api_what_if_analysis():
    """API endpoint for what-if analysis calculations."""
    # Get JSON data from request
    data = request.get_json()
    
    if not data or 'property_id' not in data or 'parameters' not in data:
        return jsonify({"error": "Missing required data"}), 400
    
    # Get property from database
    property_obj = Property.query.get(data['property_id'])
    if not property_obj:
        return jsonify({"error": "Property not found"}), 404
    
    # Get parameters
    params = data['parameters']
    
    # Get base valuation from the property
    base_value = property_obj.estimated_value or 0
    
    # Calculate new valuation based on parameters (simplified model)
    cap_rate = params.get('capRate', 0.05)
    market_adjustment = params.get('marketTrendAdjustment', 0)
    renovation_impact = params.get('renovationImpact', 0)
    
    # Apply cap rate adjustment (lower cap rate = higher value)
    cap_rate_adjustment = (0.05 / cap_rate) - 1
    
    # Apply other adjustments
    market_adjustment_value = market_adjustment * base_value
    renovation_adjustment_value = renovation_impact * base_value
    
    # Calculate new valuation with all adjustments
    new_valuation = base_value * (1 + cap_rate_adjustment) + market_adjustment_value + renovation_adjustment_value
    
    # Ensure the value doesn't go below zero
    new_valuation = max(new_valuation, 0)
    
    # Calculate factor contributions
    location_weight = params.get('locationWeight', 0.4)
    square_footage_weight = params.get('squareFootageWeight', 0.3)
    amenities_weight = params.get('amenitiesWeight', 0.2)
    market_weight = abs(market_adjustment)
    renovation_weight = renovation_impact
    
    # Normalize weights to ensure they sum to 1
    total_weight = location_weight + square_footage_weight + amenities_weight + market_weight + renovation_weight
    if total_weight > 0:
        location_factor = (location_weight / total_weight) * new_valuation
        size_factor = (square_footage_weight / total_weight) * new_valuation
        amenities_factor = (amenities_weight / total_weight) * new_valuation
        market_factor = (market_weight / total_weight) * new_valuation
        renovation_factor = (renovation_weight / total_weight) * new_valuation
    else:
        # Default equal distribution if all weights are zero
        location_factor = size_factor = amenities_factor = market_factor = renovation_factor = new_valuation / 5
    
    # Return results
    return jsonify({
        "original_valuation": base_value,
        "adjusted_valuation": new_valuation,
        "percent_change": ((new_valuation - base_value) / base_value * 100) if base_value > 0 else 0,
        "factors": {
            "location": location_factor,
            "size": size_factor,
            "amenities": amenities_factor,
            "market": market_factor,
            "renovation": renovation_factor
        }
    })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)