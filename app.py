"""
BCBS Property Valuation Dashboard Application

This application serves as the web interface for the BCBS property valuation system,
providing a dashboard with model performance metrics and a simple property valuation tool.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing the property valuation functions
try:
    from src.valuation import estimate_property_value
    logger.info("Successfully imported valuation functions")
except ImportError as e:
    logger.warning(f"Failed to import valuation functions: {str(e)}")
    logger.info("Using fallback valuation function")
    
    # Define a simple fallback valuation function
    def estimate_property_value(property_data, target_property=None, **kwargs):
        """
        Simple fallback property valuation function when the main module is not available.
        """
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        logger.info("Using fallback valuation function")
        
        # Find price column
        price_cols = ['list_price', 'sale_price', 'price', 'value']
        price_col = None
        for col in price_cols:
            if col in property_data.columns:
                price_col = col
                break
        
        if not price_col:
            return {'error': 'No price column found in data'}
        
        # Basic features
        features = ['square_feet', 'bedrooms', 'bathrooms']
        
        # Train a simple model
        X = property_data[features].values
        y = property_data[price_col].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate R²
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        # Calculate feature importance
        coef = model.coef_
        abs_coef = np.abs(coef)
        norm_coef = abs_coef / np.sum(abs_coef)
        
        feature_importance = []
        for i, feature in enumerate(features):
            feature_importance.append({
                'feature': feature,
                'importance': float(norm_coef[i]),
                'coefficient': float(coef[i])
            })
        
        # Predict target property value if provided
        predicted_value = None
        if target_property is not None:
            X_target = target_property[features].values
            predicted_value = float(model.predict(X_target)[0])
        
        return {
            'predicted_value': predicted_value,
            'r2_score': float(test_r2),
            'feature_importance': feature_importance,
            'model_type': 'basic_linear_regression (fallback)',
            'model': model
        }

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs-valuation-dashboard-secret")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize SQLAlchemy with Flask app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Import and initialize models
with app.app_context():
    from models import Property, ValidationResult, PropertyValuation
    db.create_all()

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/valuation', methods=['GET', 'POST'])
def valuation_tool():
    """
    Property valuation tool page.
    
    GET: Render the valuation form
    POST: Process form data and return valuation result
    """
    if request.method == 'POST':
        # Extract form data
        try:
            square_feet = float(request.form.get('square_feet', 0))
            bedrooms = float(request.form.get('bedrooms', 0))
            bathrooms = float(request.form.get('bathrooms', 0))
            year_built = int(request.form.get('year_built', 0))
            
            # Create target property dataframe
            target_property = pd.DataFrame({
                'square_feet': [square_feet],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'year_built': [year_built]
            })
            
            # Sample training data (in production, this would come from a database)
            properties = create_sample_data(15)
            
            # Perform valuation
            if estimate_property_value:
                result = estimate_property_value(
                    properties, 
                    target_property=target_property,
                    use_gis_features=False,
                    use_multiple_regression=True,
                    include_advanced_metrics=True
                )
                
                return render_template(
                    'valuation_result.html',
                    result=result,
                    property_details={
                        'square_feet': square_feet,
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'year_built': year_built
                    }
                )
            else:
                return render_template(
                    'valuation_form.html', 
                    error="Valuation engine not available"
                )
                
        except ValueError as e:
            return render_template(
                'valuation_form.html', 
                error=f"Invalid input: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in valuation: {str(e)}", exc_info=True)
            return render_template(
                'valuation_form.html', 
                error=f"Valuation error: {str(e)}"
            )
    
    # GET request - show the form
    return render_template('valuation_form.html')

@app.route('/api/valuation', methods=['POST'])
def api_valuation():
    """
    Simple API endpoint for property valuation.
    
    Expected JSON payload:
    {
        "square_feet": 1750,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2000
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create target property dataframe
        target_property = pd.DataFrame({
            'square_feet': [float(data['square_feet'])],
            'bedrooms': [float(data['bedrooms'])],
            'bathrooms': [float(data['bathrooms'])],
            'year_built': [int(data['year_built'])]
        })
        
        # Sample training data (in production, this would come from a database)
        properties = create_sample_data(15)
        
        # Perform valuation
        if estimate_property_value:
            result = estimate_property_value(
                properties, 
                target_property=target_property,
                use_gis_features=False,
                use_multiple_regression=True,
                include_advanced_metrics=True
            )
            
            # Remove any non-serializable objects
            if 'model' in result:
                del result['model']
            if 'statsmodel' in result:
                del result['statsmodel']
                
            return jsonify(result)
        else:
            return jsonify({
                'error': 'Valuation engine not available'
            }), 500
                
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

def create_sample_data(n_samples=15):
    """Create sample property data for testing."""
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate property IDs
    property_ids = [f'P{i:03d}' for i in range(n_samples)]
    
    # Generate property features
    square_feet = np.random.normal(2000, 500, n_samples).astype(int).clip(min=800)
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.5, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                p=[0.05, 0.1, 0.4, 0.2, 0.15, 0.05, 0.05])
    year_built = np.random.randint(1970, 2020, n_samples)
    
    # Calculate property values based on a simplified formula
    list_prices = (
        square_feet * 120 +                # $120 per sq ft base
        bedrooms * 12000 +                 # $12k per bedroom
        bathrooms * 18000 +                # $18k per bathroom
        (2023 - year_built) * -800 +       # Depreciation by age
        np.random.normal(0, 15000, n_samples)  # Random variation
    ).astype(int).clip(min=150000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'property_id': property_ids,
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'list_price': list_prices
    })
    
    return df

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)