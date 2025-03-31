from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, SelectField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length, Optional, NumberRange

class PropertyValuationForm(FlaskForm):
    """Form for property valuation input."""
    
    # Location Information
    address = StringField('Address', validators=[DataRequired(), Length(min=5, max=255)])
    city = StringField('City', validators=[DataRequired(), Length(min=2, max=100)])
    state = StringField('State', validators=[DataRequired(), Length(min=2, max=2)])
    zip_code = StringField('Zip Code', validators=[DataRequired(), Length(min=5, max=10)])
    neighborhood = StringField('Neighborhood', validators=[Optional(), Length(max=100)])
    
    # Property Characteristics
    property_type = SelectField(
        'Property Type', 
        choices=[
            ('single_family', 'Single Family Home'),
            ('condo', 'Condominium'),
            ('townhouse', 'Townhouse'),
            ('multi_family', 'Multi-Family Home'),
            ('land', 'Vacant Land')
        ],
        validators=[DataRequired()]
    )
    bedrooms = IntegerField('Bedrooms', validators=[Optional(), NumberRange(min=0, max=20)])
    bathrooms = FloatField('Bathrooms', validators=[Optional(), NumberRange(min=0, max=15)])
    square_feet = FloatField('Square Feet', validators=[Optional(), NumberRange(min=0, max=100000)])
    year_built = IntegerField('Year Built', validators=[Optional(), NumberRange(min=1800, max=2025)])
    lot_size = FloatField('Lot Size (acres)', validators=[Optional(), NumberRange(min=0, max=1000)])
    
    # Geolocation (optional but enhances valuation with GIS data)
    latitude = FloatField('Latitude', validators=[Optional(), NumberRange(min=-90, max=90)])
    longitude = FloatField('Longitude', validators=[Optional(), NumberRange(min=-180, max=180)])
    
    # Valuation Method
    valuation_method = SelectField(
        'Valuation Method',
        choices=[
            ('enhanced_regression', 'Enhanced Regression (Recommended)'),
            ('lightgbm', 'LightGBM Gradient Boosting'),
            ('xgboost', 'XGBoost Gradient Boosting'),
            ('linear_regression', 'Linear Regression'),
            ('ridge_regression', 'Ridge Regression'),
            ('lasso_regression', 'Lasso Regression'),
            ('elastic_net', 'Elastic Net Regression')
        ],
        default='enhanced_regression',
        validators=[DataRequired()]
    )
    
    submit = SubmitField('Calculate Valuation')


class PropertySearchForm(FlaskForm):
    """Form for property search functionality."""
    
    search_query = StringField('Search', validators=[Optional()])
    
    neighborhood = SelectField('Neighborhood', validators=[Optional()], choices=[])
    
    property_type = SelectField(
        'Property Type',
        choices=[
            ('', 'All'),
            ('single_family', 'Single Family Home'),
            ('condo', 'Condominium'),
            ('townhouse', 'Townhouse'),
            ('multi_family', 'Multi-Family Home'),
            ('land', 'Vacant Land')
        ],
        validators=[Optional()]
    )
    
    min_bedrooms = IntegerField('Min Bedrooms', validators=[Optional(), NumberRange(min=0, max=20)])
    
    min_price = FloatField('Min Price', validators=[Optional(), NumberRange(min=0)])
    max_price = FloatField('Max Price', validators=[Optional(), NumberRange(min=0)])
    
    submit = SubmitField('Search Properties')