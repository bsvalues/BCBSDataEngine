"""
Routes for the BCBS Values application.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from flask import Blueprint, Flask, jsonify, request, render_template, redirect, url_for, flash, session
import pandas as pd
import numpy as np

from app import app, db
from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
from forms import PropertyValuationForm, PropertySearchForm
from src.valuation import EnhancedGISValuationEngine
from src.enhanced_gis_features import calculate_gis_adjustments, load_gis_features

# Configure logging
logger = logging.getLogger(__name__)

# GIS features
try:
    gis_features = load_gis_features('gis_features_metadata.json')
    if gis_features:
        logger.info(f"Loaded GIS features with {len(gis_features)} categories")
    else:
        logger.warning("No GIS features loaded, will create default")
        # Create a simple default GIS feature set for demo purposes
        gis_features = {
            "neighborhoods": {
                "Downtown": {"quality": 0.8, "crime_rate": 0.3, "school_rating": 0.7, "walkability": 0.9},
                "Suburbs": {"quality": 0.7, "crime_rate": 0.2, "school_rating": 0.8, "walkability": 0.5},
                "Rural": {"quality": 0.6, "crime_rate": 0.1, "school_rating": 0.6, "walkability": 0.3}
            },
            "points_of_interest": {
                "schools": [
                    {"name": "Central High School", "latitude": 44.5646, "longitude": -123.2620, "quality": 8.5},
                    {"name": "West Elementary", "latitude": 44.5539, "longitude": -123.2820, "quality": 9.0}
                ],
                "parks": [
                    {"name": "City Park", "latitude": 44.5700, "longitude": -123.2750},
                    {"name": "Riverfront Park", "latitude": 44.5623, "longitude": -123.2550}
                ],
                "water": [
                    {"name": "Willamette River", "latitude": 44.5550, "longitude": -123.2500}
                ]
            }
        }
        
        # Save default GIS features for future use
        with open('gis_features_metadata.json', 'w') as f:
            json.dump(gis_features, f, indent=2)
            
except Exception as e:
    logger.error(f"Error loading GIS features: {str(e)}")
    gis_features = {}


# Main routes
@app.route('/')
def index():
    """Render the home page."""
    # Get summary counts
    property_count = Property.query.count()
    valuation_count = PropertyValuation.query.count()
    
    # Get latest valuations for chart data
    latest_valuations = db.session.query(
        Property.neighborhood,
        db.func.avg(PropertyValuation.estimated_value).label('avg_value')
    )\
    .join(PropertyValuation, Property.id == PropertyValuation.property_id)\
    .group_by(Property.neighborhood)\
    .order_by(db.func.avg(PropertyValuation.estimated_value).desc())\
    .limit(10)\
    .all()
    
    # Format chart data
    neighborhood_labels = [v[0] if v[0] else 'Unknown' for v in latest_valuations]
    neighborhood_values = [float(v[1]) for v in latest_valuations]
    
    return render_template(
        'index.html',
        property_count=property_count,
        valuation_count=valuation_count,
        neighborhood_labels=json.dumps(neighborhood_labels),
        neighborhood_values=json.dumps(neighborhood_values)
    )


@app.route('/valuation', methods=['GET', 'POST'])
def valuation():
    """Render the property valuation form."""
    form = PropertyValuationForm()
    
    if form.validate_on_submit():
        # Prepare property data
        property_data = {
            'address': form.address.data,
            'city': form.city.data,
            'state': form.state.data,
            'zip_code': form.zip_code.data,
            'neighborhood': form.neighborhood.data,
            'property_type': form.property_type.data,
            'bedrooms': form.bedrooms.data,
            'bathrooms': form.bathrooms.data,
            'square_feet': form.square_feet.data,
            'year_built': form.year_built.data,
            'lot_size': form.lot_size.data,
            'latitude': form.latitude.data,
            'longitude': form.longitude.data,
            'valuation_method': form.valuation_method.data,
        }
        
        # For demo purposes, calculate a simple valuation
        try:
            # Use square footage as base if available
            if property_data['square_feet']:
                base_value = float(property_data['square_feet']) * 150  # $150 per sq ft
            else:
                base_value = 250000  # Default base value
            
            # Adjust for bedrooms and bathrooms
            if property_data['bedrooms']:
                base_value += int(property_data['bedrooms']) * 25000  # $25k per bedroom
            if property_data['bathrooms']:
                base_value += float(property_data['bathrooms']) * 15000  # $15k per bathroom
            
            # Adjust for age
            if property_data['year_built']:
                age_factor = (2025 - int(property_data['year_built'])) / 100
                base_value *= (1 - age_factor)  # Older homes have lower values
            
            # Apply GIS adjustments if coordinates provided
            gis_adjustments = None
            if property_data['latitude'] and property_data['longitude'] and gis_features:
                adjusted_value, adjustments = calculate_gis_adjustments(
                    property_data['latitude'], property_data['longitude'], base_value, gis_features
                )
                gis_adjustments = adjustments
                base_value = adjusted_value
            
            # Apply variation based on valuation method
            method_variations = {
                'linear_regression': 1.0,
                'ridge_regression': 0.98,
                'lasso_regression': 0.97,
                'elastic_net': 0.99,
                'lightgbm': 1.05,
                'xgboost': 1.03,
                'enhanced_regression': 1.02
            }
            estimated_value = base_value * method_variations.get(property_data['valuation_method'], 1.0)
            
            # Create property record
            property_id = f"PROP-{os.urandom(4).hex().upper()}"
            property = Property(
                property_id=property_id,
                address=property_data['address'],
                city=property_data['city'],
                state=property_data['state'],
                zip_code=property_data['zip_code'],
                neighborhood=property_data['neighborhood'],
                property_type=property_data['property_type'],
                bedrooms=property_data['bedrooms'],
                bathrooms=property_data['bathrooms'],
                square_feet=property_data['square_feet'],
                year_built=property_data['year_built'],
                lot_size=property_data['lot_size'],
                latitude=property_data['latitude'],
                longitude=property_data['longitude'],
            )
            db.session.add(property)
            db.session.flush()  # Get ID without committing
            
            # Create valuation record
            valuation = PropertyValuation(
                property_id=property.id,
                estimated_value=estimated_value,
                valuation_date=datetime.utcnow(),
                valuation_method=property_data['valuation_method'],
                confidence_score=0.85,  # Simulated confidence score
                adj_r2_score=0.82,  # Simulated metrics
                rmse=12500,
                mae=9500,
                inputs=property_data,
                gis_adjustments=gis_adjustments
            )
            db.session.add(valuation)
            db.session.commit()
            
            # Add model metrics based on method
            if property_data['valuation_method'] == 'enhanced_regression':
                model_info = "Enhanced Regression combines multiple model types for better accuracy."
                top_factors = ["Square Footage", "Location Quality", "Bedrooms", "Bathrooms", "Year Built"]
            elif property_data['valuation_method'] == 'lightgbm':
                model_info = "LightGBM is a gradient boosting framework that uses tree-based learning algorithms."
                top_factors = ["Square Footage", "Year Built", "Bedrooms", "Neighborhood", "Lot Size"]
            elif property_data['valuation_method'] == 'xgboost':
                model_info = "XGBoost is an optimized distributed gradient boosting library."
                top_factors = ["Square Footage", "Bedrooms", "Bathrooms", "Year Built", "Property Type"]
            else:
                model_info = "Linear regression models establish relationships between property features and values."
                top_factors = ["Square Footage", "Bedrooms", "Bathrooms", "Year Built", "Neighborhood"]
            
            # Redirect to results page
            return render_template(
                'valuation_result.html',
                property=property,
                valuation=valuation,
                estimated_value=estimated_value,
                model_info=model_info,
                top_factors=top_factors,
                factor_weights=[0.35, 0.25, 0.15, 0.15, 0.10],  # Simulated weights
                gis_adjustments=gis_adjustments
            )
        
        except Exception as e:
            logger.error(f"Error calculating valuation: {str(e)}")
            flash(f"Error calculating valuation: {str(e)}", 'danger')
    
    return render_template('valuation_form.html', form=form)


@app.route('/properties', methods=['GET'])
def properties():
    """Render the property search page."""
    # Get unique neighborhoods and property types for filter dropdowns
    neighborhoods = db.session.query(Property.neighborhood)\
        .filter(Property.neighborhood.isnot(None))\
        .distinct()\
        .order_by(Property.neighborhood)\
        .all()
    neighborhoods = [n[0] for n in neighborhoods]
    
    property_types = db.session.query(Property.property_type)\
        .filter(Property.property_type.isnot(None))\
        .distinct()\
        .order_by(Property.property_type)\
        .all()
    property_types = [pt[0] for pt in property_types]
    
    form = PropertySearchForm(neighborhoods=neighborhoods, property_types=property_types)
    
    # Get search parameters
    neighborhood = request.args.get('neighborhood')
    property_type = request.args.get('property_type')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    min_bedrooms = request.args.get('min_bedrooms')
    search_query = request.args.get('search_query')
    
    # Build query
    query = Property.query
    
    if neighborhood:
        query = query.filter(Property.neighborhood == neighborhood)
        form.neighborhood.data = neighborhood
    
    if property_type:
        query = query.filter(Property.property_type == property_type)
        form.property_type.data = property_type
    
    if min_bedrooms:
        query = query.filter(Property.bedrooms >= int(min_bedrooms))
        form.min_bedrooms.data = min_bedrooms
    
    if min_price:
        # We need to join with PropertyValuation to filter by price
        query = query.join(PropertyValuation)\
            .filter(PropertyValuation.estimated_value >= float(min_price))
        form.min_price.data = min_price
    
    if max_price:
        if 'property_valuation' not in str(query):
            query = query.join(PropertyValuation)
        query = query.filter(PropertyValuation.estimated_value <= float(max_price))
        form.max_price.data = max_price
    
    if search_query:
        query = query.filter(
            (Property.address.ilike(f'%{search_query}%')) |
            (Property.neighborhood.ilike(f'%{search_query}%')) |
            (Property.city.ilike(f'%{search_query}%'))
        )
        form.search_query.data = search_query
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    properties = pagination.items
    
    return render_template(
        'properties.html',
        form=form,
        properties=properties,
        pagination=pagination
    )


@app.route('/property/<property_id>')
def property_detail(property_id):
    """Render the property detail page."""
    property = Property.query.filter_by(property_id=property_id).first_or_404()
    
    # Get valuations
    valuations = PropertyValuation.query.filter_by(property_id=property.id)\
        .order_by(PropertyValuation.valuation_date.desc())\
        .all()
    
    # Get property features
    features = PropertyFeature.query.filter_by(property_id=property.id).all()
    
    # Prepare valuation history chart data
    dates = [v.valuation_date.strftime('%Y-%m-%d') for v in valuations]
    values = [float(v.estimated_value) for v in valuations]
    
    return render_template(
        'property_detail.html',
        property=property,
        valuations=valuations,
        features=features,
        dates=json.dumps(dates),
        values=json.dumps(values)
    )


@app.route('/dashboard')
def dashboard():
    """Render the admin dashboard."""
    # Get summary counts and data
    property_count = Property.query.count()
    valuation_count = PropertyValuation.query.count()
    
    # Property type distribution
    property_types = db.session.query(
        Property.property_type,
        db.func.count(Property.id).label('count')
    )\
    .group_by(Property.property_type)\
    .order_by(db.func.count(Property.id).desc())\
    .all()
    
    # Recent valuations
    recent_valuations = PropertyValuation.query\
        .join(Property, PropertyValuation.property_id == Property.id)\
        .order_by(PropertyValuation.valuation_date.desc())\
        .limit(10)\
        .all()
    
    # Valuation method distribution
    valuation_methods = db.session.query(
        PropertyValuation.valuation_method,
        db.func.count(PropertyValuation.id).label('count')
    )\
    .group_by(PropertyValuation.valuation_method)\
    .order_by(db.func.count(PropertyValuation.id).desc())\
    .all()
    
    # Average values by neighborhood
    neighborhood_values = db.session.query(
        Property.neighborhood,
        db.func.avg(PropertyValuation.estimated_value).label('avg_value')
    )\
    .join(PropertyValuation, Property.id == PropertyValuation.property_id)\
    .group_by(Property.neighborhood)\
    .order_by(db.func.avg(PropertyValuation.estimated_value).desc())\
    .all()
    
    # Format chart data
    property_type_labels = [pt[0] if pt[0] else 'Unknown' for pt in property_types]
    property_type_counts = [pt[1] for pt in property_types]
    
    method_labels = [m[0] for m in valuation_methods]
    method_counts = [m[1] for m in valuation_methods]
    
    neighborhood_labels = [n[0] if n[0] else 'Unknown' for n in neighborhood_values]
    neighborhood_values = [float(n[1]) for n in neighborhood_values]
    
    return render_template(
        'dashboard.html',
        property_count=property_count,
        valuation_count=valuation_count,
        recent_valuations=recent_valuations,
        property_type_labels=json.dumps(property_type_labels),
        property_type_counts=json.dumps(property_type_counts),
        method_labels=json.dumps(method_labels),
        method_counts=json.dumps(method_counts),
        neighborhood_labels=json.dumps(neighborhood_labels),
        neighborhood_values=json.dumps(neighborhood_values)
    )


@app.route('/what-if-analysis')
def what_if_analysis():
    """Render the what-if analysis page."""
    return render_template('what_if_analysis.html')


@app.route('/api/what-if', methods=['POST'])
def what_if_api():
    """API endpoint for what-if analysis."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Get base property data
        property_id = data.get('property_id')
        base_value = data.get('base_value', 300000)
        
        # Get adjustment parameters
        cap_rate = data.get('cap_rate', 0.05)
        location_weight = data.get('location_weight', 0.3)
        size_weight = data.get('size_weight', 0.4)
        condition_weight = data.get('condition_weight', 0.3)
        
        # Apply adjustments
        adjusted_value = base_value * (1 + (cap_rate - 0.05) * 10)  # 10% change per 0.01 cap rate change
        
        # Calculate factor contributions
        location_contribution = base_value * location_weight
        size_contribution = base_value * size_weight
        condition_contribution = base_value * condition_weight
        
        # Return results
        return jsonify({
            'original_value': base_value,
            'adjusted_value': adjusted_value,
            'difference': adjusted_value - base_value,
            'difference_percent': (adjusted_value - base_value) / base_value * 100,
            'factors': {
                'location': {
                    'weight': location_weight,
                    'contribution': location_contribution
                },
                'size': {
                    'weight': size_weight,
                    'contribution': size_contribution
                },
                'condition': {
                    'weight': condition_weight,
                    'contribution': condition_contribution
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in what-if analysis: {str(e)}")
        return jsonify({'error': f'Error in what-if analysis: {str(e)}'}), 500


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500