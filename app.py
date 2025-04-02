import os
import logging

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy with a base class
class Base(DeclarativeBase):
    pass

# Initialize database
db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)

# Configure Flask app
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs_values_session_secret_key_2025")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize extensions
db.init_app(app)

# Import routes 
with app.app_context():
    # Import models first
    from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog
    
    # Create tables if they don't exist
    logger.info("Creating database tables...")
    db.create_all()
    
    # Import routes after models are loaded
    import routes

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)