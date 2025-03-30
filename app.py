import os
import logging
from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy.orm import DeclarativeBase


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Define base model class for SQLAlchemy
class Base(DeclarativeBase):
    pass


# Initialize extensions
db = SQLAlchemy(model_class=Base)
login_manager = LoginManager()


def create_app():
    # Create and configure the app
    app = Flask(__name__)
    
    # Configure the secret key
    app.secret_key = os.environ.get("SESSION_SECRET")
    if not app.secret_key:
        logger.warning("No SESSION_SECRET set. Using default (insecure) secret key.")
        app.secret_key = "bcbs_values_default_secret_key"  # Default for development
    
    # Configure the database
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    if not app.config["SQLALCHEMY_DATABASE_URI"]:
        logger.warning("No DATABASE_URL set. Using default SQLite database.")
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///bcbs_values.db"
    
    # Configure SQLAlchemy
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize extensions with the app
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message_category = 'warning'
    
    # Import models and register the user loader
    with app.app_context():
        from models import User, Property, Valuation, GISFeature
        
        @login_manager.user_loader
        def load_user(user_id):
            return User.query.get(int(user_id))
        
        # Create database tables
        db.create_all()
        
        # Register blueprints and routes
        from routes import register_routes
        register_routes(app)
        
        # Register error handlers
        @app.errorhandler(404)
        def not_found_error(error):
            return render_template('404.html'), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            db.session.rollback()
            return render_template('500.html'), 500
    
    return app


# Create the Flask app instance
app = create_app()

# Import this at the end to avoid circular imports
from flask import render_template