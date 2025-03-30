from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, IntegerField, FloatField, SelectField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, ValidationError
from models import User


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Log In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')


class ProfileForm(FlaskForm):
    username = StringField('Username', render_kw={'readonly': True})
    first_name = StringField('First Name', validators=[Length(max=50)])
    last_name = StringField('Last Name', validators=[Length(max=50)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    submit = SubmitField('Update Profile')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None and user.username != self.username.data:
            raise ValidationError('Please use a different email address.')


class PropertyForm(FlaskForm):
    address = StringField('Street Address', validators=[DataRequired(), Length(max=200)])
    city = StringField('City', validators=[DataRequired(), Length(max=100)])
    state = StringField('State', validators=[DataRequired(), Length(min=2, max=2)])
    zip_code = StringField('ZIP Code', validators=[DataRequired(), Length(min=5, max=10)])
    bedrooms = IntegerField('Bedrooms', validators=[Optional()])
    bathrooms = FloatField('Bathrooms', validators=[Optional()])
    square_feet = IntegerField('Square Feet', validators=[Optional()])
    lot_size = FloatField('Lot Size (acres)', validators=[Optional()])
    year_built = IntegerField('Year Built', validators=[Optional()])
    property_type = SelectField('Property Type', choices=[
        ('single_family', 'Single Family'),
        ('condo', 'Condominium'),
        ('townhouse', 'Townhouse'),
        ('multi_family', 'Multi-Family'),
        ('land', 'Land/Lot'),
        ('commercial', 'Commercial')
    ])
    latitude = FloatField('Latitude', validators=[Optional()])
    longitude = FloatField('Longitude', validators=[Optional()])
    submit = SubmitField('Add Property')


class ValuationRequestForm(FlaskForm):
    valuation_method = SelectField('Valuation Method', choices=[
        ('basic', 'Basic Valuation'),
        ('enhanced', 'Enhanced ML Valuation'),
        ('advanced_gis', 'Advanced GIS Valuation')
    ])
    notes = TextAreaField('Notes', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Request Valuation')