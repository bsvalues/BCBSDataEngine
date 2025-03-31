from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, BooleanField, SubmitField
from wtforms import TextAreaField, SelectField, FloatField, IntegerField, DateField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, ValidationError
from models import User


class LoginForm(FlaskForm):
    """Form for user login"""
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class RegistrationForm(FlaskForm):
    """Form for user registration"""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=64, message="Username must be between 3 and 64 characters")
    ])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message="Password must be at least 8 characters long")
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message="Passwords must match")
    ])
    first_name = StringField('First Name', validators=[Optional(), Length(max=64)])
    last_name = StringField('Last Name', validators=[Optional(), Length(max=64)])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        """Validate that the username is unique"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username is already taken. Please choose a different one.')
    
    def validate_email(self, email):
        """Validate that the email is unique"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email is already registered. Please use a different one.')


class PropertyForm(FlaskForm):
    """Form for adding or editing properties"""
    address = StringField('Address', validators=[DataRequired(), Length(max=256)])
    city = StringField('City', validators=[DataRequired(), Length(max=100)])
    state = StringField('State', validators=[DataRequired(), Length(max=50)])
    zip_code = StringField('ZIP Code', validators=[DataRequired(), Length(max=20)])
    property_type = SelectField('Property Type', choices=[
        ('residential', 'Residential'),
        ('commercial', 'Commercial'),
        ('industrial', 'Industrial'),
        ('land', 'Land'),
        ('multi_family', 'Multi-Family')
    ], validators=[DataRequired()])
    bedrooms = IntegerField('Bedrooms', validators=[Optional()])
    bathrooms = FloatField('Bathrooms', validators=[Optional()])
    square_feet = IntegerField('Square Feet', validators=[Optional()])
    lot_size = FloatField('Lot Size (acres)', validators=[Optional()])
    year_built = IntegerField('Year Built', validators=[Optional()])
    last_sold_date = DateField('Last Sold Date', format='%Y-%m-%d', validators=[Optional()])
    last_sold_price = FloatField('Last Sold Price ($)', validators=[Optional()])
    latitude = FloatField('Latitude', validators=[Optional()])
    longitude = FloatField('Longitude', validators=[Optional()])
    neighborhood = StringField('Neighborhood', validators=[Optional(), Length(max=100)])
    description = TextAreaField('Description', validators=[Optional()])
    features = TextAreaField('Features', validators=[Optional()])
    submit = SubmitField('Save Property')


class ValuationForm(FlaskForm):
    """Form for requesting a property valuation"""
    property_id = SelectField('Property', coerce=int, validators=[DataRequired()])
    valuation_method = SelectField('Valuation Method', choices=[
        ('enhanced_regression', 'Enhanced Regression'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('ensemble', 'Ensemble'),
        ('gis_enhanced', 'GIS Enhanced')
    ], validators=[DataRequired()])
    notes = TextAreaField('Notes', validators=[Optional()])
    submit = SubmitField('Request Valuation')


class AgentForm(FlaskForm):
    """Form for adding or editing valuation agents"""
    name = StringField('Agent Name', validators=[DataRequired(), Length(max=100)])
    agent_type = SelectField('Agent Type', choices=[
        ('regression', 'Regression'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('ensemble', 'Ensemble'),
        ('gis', 'GIS')
    ], validators=[DataRequired()])
    description = TextAreaField('Description', validators=[Optional()])
    model_version = StringField('Model Version', validators=[Optional(), Length(max=50)])
    configuration = TextAreaField('Configuration (JSON)', validators=[Optional()])
    submit = SubmitField('Save Agent')


class ApiKeyForm(FlaskForm):
    """Form for creating or editing API keys"""
    name = StringField('Key Name', validators=[DataRequired(), Length(max=100)])
    permissions = SelectField('Permissions', choices=[
        ('read', 'Read Only'),
        ('read_write', 'Read & Write'),
        ('full', 'Full Access')
    ], validators=[DataRequired()])
    submit = SubmitField('Generate API Key')


class SearchForm(FlaskForm):
    """Form for searching properties and valuations"""
    query = StringField('Search', validators=[Optional()])
    property_type = SelectField('Property Type', choices=[
        ('', 'All Types'),
        ('residential', 'Residential'),
        ('commercial', 'Commercial'),
        ('industrial', 'Industrial'),
        ('land', 'Land'),
        ('multi_family', 'Multi-Family')
    ], validators=[Optional()])
    city = StringField('City', validators=[Optional(), Length(max=100)])
    min_price = FloatField('Min Price', validators=[Optional()])
    max_price = FloatField('Max Price', validators=[Optional()])
    min_beds = IntegerField('Min Beds', validators=[Optional()])
    min_baths = FloatField('Min Baths', validators=[Optional()])
    min_sqft = IntegerField('Min Sq. Ft.', validators=[Optional()])
    submit = SubmitField('Search')


class ProfileForm(FlaskForm):
    """Form for updating user profile"""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=64, message="Username must be between 3 and 64 characters")
    ])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    first_name = StringField('First Name', validators=[Optional(), Length(max=64)])
    last_name = StringField('Last Name', validators=[Optional(), Length(max=64)])
    submit = SubmitField('Update Profile')
    
    def __init__(self, original_username, original_email, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email
    
    def validate_username(self, username):
        """Validate that the username is unique, except for the current user"""
        if username.data != self.original_username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('Username is already taken. Please choose a different one.')
    
    def validate_email(self, email):
        """Validate that the email is unique, except for the current user"""
        if email.data != self.original_email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('Email is already registered. Please use a different one.')


class ChangePasswordForm(FlaskForm):
    """Form for changing password"""
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[
        DataRequired(),
        Length(min=8, message="Password must be at least 8 characters long")
    ])
    confirm_new_password = PasswordField('Confirm New Password', validators=[
        DataRequired(),
        EqualTo('new_password', message="Passwords must match")
    ])
    submit = SubmitField('Change Password')


class ContactForm(FlaskForm):
    """Form for contact page"""
    name = StringField('Name', validators=[DataRequired(), Length(max=100)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    subject = StringField('Subject', validators=[DataRequired(), Length(max=200)])
    message = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField('Send Message')