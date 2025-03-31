import os
import logging
from datetime import datetime

from flask import Flask, request, g
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy.orm import DeclarativeBase
from flask_cors import CORS


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a base class for declarative models
class Base(DeclarativeBase):
    pass

# Initialize extensions
db = SQLAlchemy(model_class=Base)
login_manager = LoginManager()

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs-property-valuation-dev-key")

# Enable CORS for API routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

# Request handlers
@app.before_request
def before_request():
    g.request_start_time = datetime.utcnow()
    
@app.after_request
def after_request(response):
    if hasattr(g, 'request_start_time'):
        request_duration = datetime.utcnow() - g.request_start_time
        logger.debug(f"Request to {request.path} completed in {request_duration.total_seconds():.3f}s")
    return response

@app.context_processor
def inject_current_year():
    return {'now': datetime.utcnow()}

# Initialize database
with app.app_context():
    # Import models here to avoid circular imports
    import models  # noqa: F401
    
    # Create all tables
    db.create_all()
    
    logger.info("Database tables created")