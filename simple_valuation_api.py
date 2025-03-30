"""
Simple Valuation API for BCBS_Values

This module provides a Flask-based REST API for the BCBS_Values property valuation system.
It exposes endpoints for property valuation with and without GIS data integration,
as well as a simple web interface for users to interact with the valuation system.
"""

import os
import json
import logging
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the valuation functions
try:
    from src.valuation import estimate_property_value, train_basic_valuation_model, train_multiple_regression_model
    logger.info("Successfully imported valuation functions from src package")
except ImportError:
    # Fallback to direct import if src package is not available
    try:
        from valuation import estimate_property_value, train_basic_valuation_model, train_multiple_regression_model
        logger.info("Successfully imported valuation functions directly")
    except ImportError:
        logger.error("Failed to import valuation functions. Make sure they are accessible in your PYTHONPATH.")
        raise

# Create Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Define reference points and neighborhood ratings
REF_POINTS = {
    'downtown_richland': {
        'lat': 46.2804, 
        'lon': -119.2752, 
        'weight': 1.0  # Downtown Richland
    },
    'downtown_kennewick': {
        'lat': 46.2112, 
        'lon': -119.1367, 
        'weight': 0.9  # Downtown Kennewick
    },
    'downtown_pasco': {
        'lat': 46.2395, 
        'lon': -119.1005, 
        'weight': 0.8  # Downtown Pasco
    },
    'columbia_center_mall': {
        'lat': 46.2185, 
        'lon': -119.2232, 
        'weight': 0.7  # Columbia Center Mall
    },
    'hanford_site': {
        'lat': 46.5506, 
        'lon': -119.4913, 
        'weight': 0.3  # Hanford Site
    }
}

NEIGHBORHOOD_RATINGS = {
    'Richland': 1.15,       # Premium location
    'West Richland': 1.05,  # Above average
    'Kennewick': 1.0,       # Average
    'Pasco': 0.95,          # Slightly below average
    'Benton City': 0.9,     # Below average
    'Prosser': 0.85,        # Further below average
    
    # Add common neighborhoods
    'Meadow Springs': 1.2,  # Premium Richland neighborhood
    'Horn Rapids': 1.1,     # Above average Richland neighborhood
    'Queensgate': 1.15,     # Premium West Richland neighborhood
    'Southridge': 1.05,     # Above average Kennewick neighborhood
    'Road 68': 1.0,         # Average Pasco neighborhood
    
    # Default for unknown locations
    'Unknown': 1.0
}

# Global variable to store training data
TRAINING_DATA = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({
        "status": "healthy",
        "message": "BCBS_Values Valuation API is operational."
    })

@app.route('/api/valuation', methods=['POST'])
def valuation():
    """
    Estimate property value based on features.
    
    Expects JSON input with property features:
    {
        "square_feet": 2000,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "year_built": 2005,
        "latitude": 46.2804,
        "longitude": -119.2752,
        "city": "Richland",
        "neighborhood": "Meadow Springs",
        "use_gis": true
    }
    
    Returns valuation result with predicted value and model metrics.
    """
    global TRAINING_DATA
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Check for required fields
        required_fields = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Load training data if not already loaded
        if TRAINING_DATA is None:
            try:
                # Try to load from test data path first, then fallback to sample paths
                data_paths = [
                    'data/test_properties.csv',  # Our new test data with price information
                    'data/benton_county_properties.csv',
                    'data/property_data.csv',
                    'data/pacs_sample.csv'
                ]
                
                for path in data_paths:
                    if os.path.exists(path):
                        logger.info(f"Loading training data from {path}")
                        TRAINING_DATA = pd.read_csv(path)
                        break
                
                if TRAINING_DATA is None:
                    # If no files found, generate fresh test data
                    logger.info("No data files found, generating fresh test data")
                    from data.generate_test_data import generate_test_data
                    test_data_path = generate_test_data(num_properties=50)
                    TRAINING_DATA = pd.read_csv(test_data_path)
                    
                logger.info(f"Training data loaded with {len(TRAINING_DATA)} properties")
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                return jsonify({"error": f"Error loading training data: {str(e)}"}), 500
        
        # Prepare target property
        target_property = pd.DataFrame({
            'property_id': ['TARGET'],
            'square_feet': [data.get('square_feet')],
            'bedrooms': [data.get('bedrooms')],
            'bathrooms': [data.get('bathrooms')],
            'year_built': [data.get('year_built')]
        })
        
        # Add GIS data if available
        if 'latitude' in data and 'longitude' in data:
            target_property['latitude'] = [data.get('latitude')]
            target_property['longitude'] = [data.get('longitude')]
        
        if 'city' in data:
            target_property['city'] = [data.get('city')]
        
        if 'neighborhood' in data:
            target_property['neighborhood'] = [data.get('neighborhood')]
        
        # Choose valuation method
        use_gis = data.get('use_gis', True)
        
        # Run valuation
        result = estimate_property_value(
            TRAINING_DATA,
            target_property,
            ref_points=REF_POINTS,
            neighborhood_ratings=NEIGHBORHOOD_RATINGS,
            use_gis_features=use_gis
        )
        
        # Handle errors
        if 'error' in result:
            return jsonify({"error": result['error']}), 500
        
        # Format response
        response = {
            "property": {
                "square_feet": data.get('square_feet'),
                "bedrooms": data.get('bedrooms'),
                "bathrooms": data.get('bathrooms'),
                "year_built": data.get('year_built'),
                "city": data.get('city', 'Unknown'),
                "neighborhood": data.get('neighborhood', 'Unknown')
            },
            "valuation": {
                "predicted_value": round(result['predicted_value'], 2),
                "confidence_interval": result.get('confidence_interval', [0, 0]),
                "r2_score": result.get('r2_score', 0),
                "model_type": "GIS-enhanced" if use_gis else "Standard"
            }
        }
        
        # Add GIS metrics if available
        if use_gis and 'gis_metrics' in result:
            response['valuation']['gis_metrics'] = {
                "features_used": result['gis_metrics'].get('features_used', []),
                "adjustment_factor": result['gis_metrics'].get('adjustment_factor', 1.0)
            }
        
        # Add feature importance if available
        if 'feature_importance' in result:
            response['valuation']['feature_importance'] = result['feature_importance']
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in valuation endpoint: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/neighborhoods', methods=['GET'])
def get_neighborhoods():
    """Return the list of neighborhoods and their quality ratings."""
    return jsonify({
        "neighborhoods": {k: v for k, v in NEIGHBORHOOD_RATINGS.items() if k != 'Unknown'}
    })

@app.route('/api/reference-points', methods=['GET'])
def get_reference_points():
    """Return the list of reference points used for GIS calculations."""
    return jsonify({
        "reference_points": REF_POINTS
    })

# Web interface routes
@app.route('/', methods=['GET'])
def index():
    """Render the main web interface."""
    return render_template('valuation_form.html')

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template('valuation_form.html')

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)