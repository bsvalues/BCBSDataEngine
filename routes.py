from flask import render_template, request, jsonify, flash, redirect, url_for
import logging
import pandas as pd
from db.database import Database
from etl.data_validation import DataValidator
from src.valuation import estimate_property_value

# Configure logging
logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all routes with the Flask app."""
    
    @app.route('/')
    def index():
        """Render the home page."""
        return render_template('index.html')
    
    @app.route('/properties')
    def properties():
        """Render the properties page with data from the database."""
        try:
            db = Database()
            # Get only Benton County, WA properties by default
            property_data = db.get_all_properties(benton_county_only=True)
            db.close()
            
            # Convert to list of dictionaries for template
            properties_list = property_data.to_dict('records') if not property_data.empty else []
            
            return render_template('properties.html', 
                                  properties=properties_list, 
                                  count=len(properties_list),
                                  location_focus="Benton County, Washington")
        except Exception as e:
            logger.error(f"Error loading properties: {str(e)}", exc_info=True)
            flash(f"Error loading properties: {str(e)}", "danger")
            return render_template('properties.html', properties=[], count=0, location_focus="Benton County, Washington")
    
    @app.route('/validation')
    def validation():
        """Render the validation results page."""
        try:
            db = Database()
            validation_results = db.get_validation_results(limit=10)
            db.close()
            
            return render_template('validation.html', 
                                  validation_results=validation_results, 
                                  count=len(validation_results))
        except Exception as e:
            logger.error(f"Error loading validation results: {str(e)}", exc_info=True)
            flash(f"Error loading validation results: {str(e)}", "danger")
            return render_template('validation.html', validation_results=[], count=0)
    
    @app.route('/search', methods=['GET', 'POST'])
    def search():
        """Handle property search."""
        if request.method == 'POST':
            # Get search criteria from form
            criteria = {}
            
            if request.form.get('city'):
                criteria['city'] = request.form.get('city')
            
            if request.form.get('state'):
                criteria['state'] = request.form.get('state')
            
            if request.form.get('zip_code'):
                criteria['zip_code'] = request.form.get('zip_code')
            
            if request.form.get('min_price'):
                criteria['min_price'] = float(request.form.get('min_price'))
            
            if request.form.get('max_price'):
                criteria['max_price'] = float(request.form.get('max_price'))
            
            if request.form.get('property_type'):
                criteria['property_type'] = request.form.get('property_type')
            
            # Query database with criteria
            try:
                db = Database()
                property_data = db.get_properties_by_criteria(criteria, benton_county_only=True)
                db.close()
                
                # Convert to list of dictionaries for template
                properties_list = property_data.to_dict('records') if not property_data.empty else []
                
                return render_template('search_results.html', 
                                      properties=properties_list, 
                                      count=len(properties_list),
                                      criteria=criteria,
                                      location_focus="Benton County, Washington")
            except Exception as e:
                logger.error(f"Error searching properties: {str(e)}", exc_info=True)
                flash(f"Error searching properties: {str(e)}", "danger")
                return render_template('search_results.html', properties=[], count=0, criteria=criteria, 
                                       location_focus="Benton County, Washington")
        else:
            # GET request - show search form
            return render_template('search.html')
    
    @app.route('/api/properties')
    def api_properties():
        """API endpoint to get properties."""
        try:
            db = Database()
            # Get only Benton County, WA properties by default
            property_data = db.get_all_properties(benton_county_only=True)
            db.close()
            
            # Convert to list of dictionaries for API
            properties_list = property_data.to_dict('records') if not property_data.empty else []
            
            return jsonify({
                'status': 'success',
                'count': len(properties_list),
                'location_focus': 'Benton County, Washington',
                'properties': properties_list
            })
        except Exception as e:
            logger.error(f"API error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/validation')
    def api_validation():
        """API endpoint to run validation."""
        try:
            db = Database()
            validator = DataValidator(db)
            results = validator.validate_all()
            db.close()
            
            return jsonify({
                'status': 'success',
                'validation_results': results
            })
        except Exception as e:
            logger.error(f"API error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    @app.route('/api/valuation/<int:property_id>')
    def api_valuation(property_id):
        """API endpoint to get property valuation."""
        try:
            db = Database()
            
            # Get all properties for training data
            all_properties = db.get_all_properties(benton_county_only=True)
            
            if all_properties.empty:
                return jsonify({
                    'status': 'error',
                    'message': 'No properties available for valuation model training'
                }), 400
                
            # Get the target property
            target_property = all_properties[all_properties['id'] == property_id]
            
            if target_property.empty:
                return jsonify({
                    'status': 'error',
                    'message': f'Property with ID {property_id} not found'
                }), 404
                
            # Remove target property from training data
            training_data = all_properties[all_properties['id'] != property_id]
            
            # Make a copy of target property to remove price info for prediction
            prediction_property = target_property.copy()
            for price_col in ['list_price', 'estimated_value', 'last_sale_price']:
                if price_col in prediction_property.columns:
                    prediction_property[price_col] = None
            
            # Run valuation model
            result = estimate_property_value(training_data, prediction_property)
            
            # Add original property data for comparison
            property_data = target_property.iloc[0].to_dict()
            
            # Return results
            return jsonify({
                'status': 'success',
                'property': property_data,
                'valuation_results': result
            })
            
        except Exception as e:
            logger.error(f"API valuation error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    @app.route('/property/<int:property_id>')
    def property_detail(property_id):
        """Render property detail page with valuation."""
        try:
            db = Database()
            
            # Get all properties
            all_properties = db.get_all_properties(benton_county_only=True)
            
            if all_properties.empty:
                flash("No properties available for valuation model training", "warning")
                return redirect(url_for('properties'))
                
            # Get the target property
            target_property = all_properties[all_properties['id'] == property_id]
            
            if target_property.empty:
                flash(f"Property with ID {property_id} not found", "danger")
                return redirect(url_for('properties'))
                
            # Get property data for the template
            property_data = target_property.iloc[0].to_dict()
            
            # Remove target property from training data
            training_data = all_properties[all_properties['id'] != property_id]
            
            # Make a copy of target property to remove price info for prediction
            prediction_property = target_property.copy()
            for price_col in ['list_price', 'estimated_value', 'last_sale_price']:
                if price_col in prediction_property.columns:
                    prediction_property[price_col] = None
            
            # Run valuation model
            valuation_results = estimate_property_value(training_data, prediction_property)
            
            # Get original price for comparison
            original_price = None
            price_source = None
            for price_col in ['list_price', 'estimated_value', 'last_sale_price']:
                if price_col in property_data and property_data[price_col] is not None:
                    original_price = property_data[price_col]
                    price_source = price_col
                    break
            
            # Calculate price difference if both prices exist
            price_difference = None
            price_difference_percent = None
            if original_price and 'predicted_value' in valuation_results and valuation_results['predicted_value']:
                price_difference = valuation_results['predicted_value'] - original_price
                price_difference_percent = (price_difference / original_price) * 100
            
            db.close()
            
            # Render template with all data
            return render_template('property_detail.html', 
                                  property=property_data,
                                  valuation_results=valuation_results,
                                  original_price=original_price,
                                  price_source=price_source,
                                  price_difference=price_difference,
                                  price_difference_percent=price_difference_percent,
                                  location_focus="Benton County, Washington")
            
        except Exception as e:
            logger.error(f"Error loading property detail: {str(e)}", exc_info=True)
            flash(f"Error loading property detail: {str(e)}", "danger")
            return redirect(url_for('properties'))
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return render_template('500.html'), 500