"""
Forms for the BCBS Values application.
"""
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, SelectField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Optional, NumberRange


class PropertyValuationForm(FlaskForm):
    """Form for property valuation."""
    address = StringField('Address', validators=[DataRequired()])
    city = StringField('City', validators=[DataRequired()])
    state = StringField('State', validators=[DataRequired()])
    zip_code = StringField('Zip Code', validators=[DataRequired()])
    neighborhood = StringField('Neighborhood', validators=[Optional()])
    property_type = SelectField('Property Type', choices=[
        ('single_family', 'Single Family Home'),
        ('condo', 'Condominium'),
        ('townhouse', 'Townhouse'),
        ('multi_family', 'Multi-Family Home'),
        ('land', 'Vacant Land'),
    ], validators=[DataRequired()])
    bedrooms = IntegerField('Bedrooms', validators=[NumberRange(min=0), Optional()])
    bathrooms = FloatField('Bathrooms', validators=[NumberRange(min=0), Optional()])
    square_feet = FloatField('Square Feet', validators=[NumberRange(min=0), Optional()])
    year_built = IntegerField('Year Built', validators=[NumberRange(min=1800), Optional()])
    lot_size = FloatField('Lot Size (acres)', validators=[NumberRange(min=0), Optional()])
    latitude = FloatField('Latitude', validators=[Optional()])
    longitude = FloatField('Longitude', validators=[Optional()])
    valuation_method = SelectField('Valuation Method', choices=[
        ('enhanced_regression', 'Enhanced Regression (Recommended)'),
        ('lightgbm', 'LightGBM Gradient Boosting'),
        ('xgboost', 'XGBoost'),
        ('linear_regression', 'Linear Regression'),
        ('ridge_regression', 'Ridge Regression'),
        ('lasso_regression', 'Lasso Regression'),
        ('elastic_net', 'Elastic Net'),
    ], validators=[DataRequired()])
    submit = SubmitField('Calculate Valuation')


class PropertySearchForm(FlaskForm):
    """Form for property search."""
    search_query = StringField('Search', validators=[Optional()])
    neighborhood = SelectField('Neighborhood', validators=[Optional()])
    property_type = SelectField('Property Type', validators=[Optional()])
    min_price = FloatField('Min Price', validators=[NumberRange(min=0), Optional()])
    max_price = FloatField('Max Price', validators=[NumberRange(min=0), Optional()])
    min_bedrooms = IntegerField('Min Bedrooms', validators=[NumberRange(min=0), Optional()])
    submit = SubmitField('Search')
    
    def __init__(self, *args, **kwargs):
        neighborhoods = kwargs.pop('neighborhoods', [])
        property_types = kwargs.pop('property_types', [])
        super(PropertySearchForm, self).__init__(*args, **kwargs)
        
        # Add empty option at the beginning
        neighborhood_choices = [('', 'All Neighborhoods')] + [(n, n) for n in neighborhoods]
        property_type_choices = [('', 'All Property Types')] + [(pt, pt) for pt in property_types]
        
        self.neighborhood.choices = neighborhood_choices
        self.property_type.choices = property_type_choices