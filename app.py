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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs_values_dev_key")

# Sample data for property valuations
SAMPLE_VALUATIONS = [
    {
        "property_id": "PROP-1001",
        "address": "123 Cherry Lane, Richland, WA 99352",
        "estimated_value": 425000.00,
        "confidence_score": 0.92,
        "model_used": "advanced_regression",
        "valuation_date": datetime.datetime.now().isoformat(),
        "features_used": {
            "square_feet": 2450,
            "bedrooms": 4,
            "bathrooms": 2.5,
            "year_built": 1998,
            "lot_size": 12000
        },
        "comparable_properties": [
            {"id": "COMP-101", "address": "125 Cherry Lane", "sale_price": 415000},
            {"id": "COMP-102", "address": "130 Cherry Lane", "sale_price": 432000}
        ]
    },
    {
        "property_id": "PROP-1002",
        "address": "456 Oak Street, Kennewick, WA 99336",
        "estimated_value": 375000.00,
        "confidence_score": 0.88,
        "model_used": "hedonic_price_model",
        "valuation_date": datetime.datetime.now().isoformat(),
        "features_used": {
            "square_feet": 2100,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "year_built": 2005,
            "lot_size": 9500
        },
        "comparable_properties": [
            {"id": "COMP-201", "address": "460 Oak Street", "sale_price": 368000},
            {"id": "COMP-202", "address": "470 Oak Street", "sale_price": 382500}
        ]
    }
]

# Sample ETL status data
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

# Register routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/properties')
def properties():
    """Render the properties page with data."""
    return render_template('properties.html', properties=SAMPLE_VALUATIONS)

@app.route('/property/<property_id>')
def property_detail(property_id):
    """Render property detail page with valuation."""
    property_data = next(
        (p for p in SAMPLE_VALUATIONS if p["property_id"] == property_id), 
        None
    )
    
    if property_data:
        return render_template('property_detail.html', property=property_data)
    else:
        return render_template('404.html'), 404

@app.route('/validation')
def validation():
    """Render the validation results page."""
    return render_template('validation.html', validation_results=ETL_STATUS)

@app.route('/search')
def search():
    """Handle property search."""
    query = request.args.get('query', '')
    results = []
    
    if query:
        results = [p for p in SAMPLE_VALUATIONS 
                  if query.lower() in p["address"].lower()]
    
    return render_template('search_results.html', 
                          query=query, 
                          results=results)

# API Routes
@app.route('/api/properties')
def api_properties():
    """API endpoint to get properties."""
    return jsonify(SAMPLE_VALUATIONS)

@app.route('/api/property/<property_id>')
def api_property(property_id):
    """API endpoint to get property details."""
    property_data = next(
        (p for p in SAMPLE_VALUATIONS if p["property_id"] == property_id), 
        None
    )
    
    if property_data:
        return jsonify(property_data)
    else:
        return jsonify({"error": "Property not found"}), 404

@app.route('/api/validation')
def api_validation():
    """API endpoint for validation results."""
    return jsonify(ETL_STATUS)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)