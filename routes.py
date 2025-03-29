from flask import render_template, request, jsonify, flash, redirect, url_for
import logging
from db.database import Database
from etl.data_validation import DataValidator

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
            property_data = db.get_all_properties()
            db.close()
            
            # Convert to list of dictionaries for template
            properties_list = property_data.to_dict('records') if not property_data.empty else []
            
            return render_template('properties.html', 
                                  properties=properties_list, 
                                  count=len(properties_list))
        except Exception as e:
            logger.error(f"Error loading properties: {str(e)}", exc_info=True)
            flash(f"Error loading properties: {str(e)}", "danger")
            return render_template('properties.html', properties=[], count=0)
    
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
                property_data = db.get_properties_by_criteria(criteria)
                db.close()
                
                # Convert to list of dictionaries for template
                properties_list = property_data.to_dict('records') if not property_data.empty else []
                
                return render_template('search_results.html', 
                                      properties=properties_list, 
                                      count=len(properties_list),
                                      criteria=criteria)
            except Exception as e:
                logger.error(f"Error searching properties: {str(e)}", exc_info=True)
                flash(f"Error searching properties: {str(e)}", "danger")
                return render_template('search_results.html', properties=[], count=0, criteria=criteria)
        else:
            # GET request - show search form
            return render_template('search.html')
    
    @app.route('/api/properties')
    def api_properties():
        """API endpoint to get properties."""
        try:
            db = Database()
            property_data = db.get_all_properties()
            db.close()
            
            # Convert to list of dictionaries for API
            properties_list = property_data.to_dict('records') if not property_data.empty else []
            
            return jsonify({
                'status': 'success',
                'count': len(properties_list),
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
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return render_template('500.html'), 500