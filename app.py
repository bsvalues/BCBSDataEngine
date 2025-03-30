import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
migrate = Migrate()
login_manager = LoginManager()


def create_app():
    # create the app
    app = Flask(__name__)
    
    # setup a secret key, required by sessions
    app.secret_key = os.environ.get("SESSION_SECRET", "bcbs_values_session_secret_key_2025")
    
    # configure the database, relative to the app instance folder
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Set up login manager
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    # initialize the app with extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    with app.app_context():
        # register blueprints
        from routes import main_bp, auth_bp, admin_bp
        app.register_blueprint(main_bp)
        app.register_blueprint(auth_bp, url_prefix='/auth')
        app.register_blueprint(admin_bp, url_prefix='/admin')
        
        # Import and register models to ensure they're known to SQLAlchemy
        import models
        
        # Create database tables
        db.create_all()
        
        return app