"""
Forms for BCBS Property Valuation application.
Includes user authentication, property, and valuation forms.
"""
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms import FloatField, IntegerField, TextAreaField, HiddenField, DateField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, NumberRange
from wtforms.validators import ValidationError, Regexp
from datetime import datetime

from models import User

class LoginForm(FlaskForm):
    """User login form."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    """User registration form."""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=64, message="Username must be between 3 and 64 characters.")
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message="Please enter a valid email address."),
        Length(max=120)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message="Password must be at least 8 characters long."),
        Regexp(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]+$',
               message="Password must include at least one letter, one number, and one special character.")
    ])
    password2 = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message="Passwords must match.")
    ])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        """Check if username already exists."""
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Username already in use. Please choose a different one.')
    
    def validate_email(self, email):
        """Check if email already exists."""
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Email address already registered. Please use a different one or login.')


class PasswordResetRequestForm(FlaskForm):
    """Form for requesting password reset."""
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')


class PasswordResetForm(FlaskForm):
    """Form for resetting password."""
    password = PasswordField('New Password', validators=[
        DataRequired(),
        Length(min=8, message="Password must be at least 8 characters long."),
        Regexp(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]+$',
               message="Password must include at least one letter, one number, and one special character.")
    ])
    password2 = PasswordField('Confirm New Password', validators=[
        DataRequired(),
        EqualTo('password', message="Passwords must match.")
    ])
    submit = SubmitField('Reset Password')


class ApiKeyForm(FlaskForm):
    """Form for creating a new API key."""
    name = StringField('Key Name', validators=[
        DataRequired(),
        Length(min=3, max=64, message="Key name must be between 3 and 64 characters.")
    ])
    submit = SubmitField('Generate API Key')


class PropertySearchForm(FlaskForm):
    """Form for searching properties."""
    query = StringField('Search', validators=[Optional()])
    property_type = SelectField('Property Type', choices=[
        ('', 'All Types'),
        ('single_family', 'Single Family'),
        ('condo', 'Condominium'),
        ('townhouse', 'Townhouse'),
        ('multi_family', 'Multi-Family'),
        ('land', 'Land')
    ], validators=[Optional()])
    neighborhood = StringField('Neighborhood', validators=[Optional()])
    min_price = FloatField('Min Price', validators=[Optional(), NumberRange(min=0)])
    max_price = FloatField('Max Price', validators=[Optional(), NumberRange(min=0)])
    min_bedrooms = IntegerField('Min Bedrooms', validators=[Optional(), NumberRange(min=0)])
    min_bathrooms = FloatField('Min Bathrooms', validators=[Optional(), NumberRange(min=0)])
    min_area = FloatField('Min Area (sq ft)', validators=[Optional(), NumberRange(min=0)])
    submit = SubmitField('Search')
    
    def validate_max_price(self, max_price):
        """Ensure max price is greater than min price if both provided."""
        if self.min_price.data and max_price.data and self.min_price.data > max_price.data:
            raise ValidationError('Maximum price must be greater than minimum price.')


class PropertyForm(FlaskForm):
    """Form for adding or editing a property."""
    property_id = StringField('Property ID', validators=[
        DataRequired(),
        Length(min=5, max=64)
    ])
    address = StringField('Address', validators=[
        DataRequired(),
        Length(min=5, max=256)
    ])
    city = StringField('City', validators=[
        DataRequired(),
        Length(min=2, max=100)
    ])
    state = StringField('State', validators=[
        DataRequired(),
        Length(min=2, max=2)
    ])
    zip_code = StringField('ZIP Code', validators=[
        DataRequired(),
        Length(min=5, max=10)
    ])
    neighborhood = StringField('Neighborhood', validators=[Optional(), Length(max=128)])
    property_type = SelectField('Property Type', choices=[
        ('single_family', 'Single Family'),
        ('condo', 'Condominium'),
        ('townhouse', 'Townhouse'),
        ('multi_family', 'Multi-Family'),
        ('land', 'Land')
    ], validators=[DataRequired()])
    year_built = IntegerField('Year Built', validators=[
        Optional(),
        NumberRange(min=1800, max=datetime.now().year)
    ])
    bedrooms = IntegerField('Bedrooms', validators=[Optional(), NumberRange(min=0)])
    bathrooms = FloatField('Bathrooms', validators=[Optional(), NumberRange(min=0)])
    living_area = FloatField('Living Area (sq ft)', validators=[Optional(), NumberRange(min=0)])
    land_area = FloatField('Land Area (acres)', validators=[Optional(), NumberRange(min=0)])
    latitude = FloatField('Latitude', validators=[Optional()])
    longitude = FloatField('Longitude', validators=[Optional()])
    submit = SubmitField('Save Property')
    
    def validate_property_id(self, property_id):
        """Custom validation for property_id field."""
        from models import Property
        property_obj = Property.query.filter_by(property_id=property_id.data).first()
        
        # If this is a new property or the property_id hasn't changed
        if hasattr(self, 'original_property_id') and property_obj and property_obj.property_id != self.original_property_id:
            raise ValidationError('This Property ID is already in use. Please choose a different one.')


class ValuationForm(FlaskForm):
    """Form for creating a new property valuation."""
    property_id = HiddenField('Property ID', validators=[DataRequired()])
    valuation_method = SelectField('Valuation Method', choices=[
        ('enhanced_regression', 'Enhanced Regression'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('linear_regression', 'Linear Regression'),
        ('ridge_regression', 'Ridge Regression'),
        ('lasso_regression', 'Lasso Regression'),
        ('elastic_net', 'Elastic Net')
    ], validators=[DataRequired()])
    valuation_date = DateField('Valuation Date', validators=[DataRequired()], default=datetime.now)
    notes = TextAreaField('Notes', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Calculate Valuation')


class BatchValuationForm(FlaskForm):
    """Form for batch property valuation processing."""
    property_type = SelectField('Property Type', choices=[
        ('all', 'All Properties'),
        ('single_family', 'Single Family'),
        ('condo', 'Condominium'),
        ('townhouse', 'Townhouse'),
        ('multi_family', 'Multi-Family'),
        ('land', 'Land')
    ], validators=[DataRequired()])
    neighborhood = StringField('Neighborhood (optional)', validators=[Optional()])
    valuation_method = SelectField('Valuation Method', choices=[
        ('enhanced_regression', 'Enhanced Regression'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('linear_regression', 'Linear Regression'),
        ('ridge_regression', 'Ridge Regression'),
        ('lasso_regression', 'Lasso Regression'),
        ('elastic_net', 'Elastic Net')
    ], validators=[DataRequired()])
    submit = SubmitField('Start Batch Valuation')