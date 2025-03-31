import os
import logging
from datetime import datetime

from flask import Flask, request, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_login import LoginManager, current_user
from werkzeug.middleware.proxy_fix import ProxyFix


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""
    pass


# Initialize extensions
db = SQLAlchemy(model_class=Base)
login_manager = LoginManager()

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development-secret-key")

# Apply middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions with app
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"

# Import models to ensure they are registered with SQLAlchemy
with app.app_context():
    from models import User, Property, Valuation, Agent, AgentLog, ETLJob, ApiKey, MarketTrend, PropertyImage
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    """Load a user from the database"""
    return User.query.get(int(user_id))


@app.before_request
def before_request():
    """Execute before each request"""
    g.request_start_time = datetime.utcnow()
    if current_user.is_authenticated:
        current_user.last_login = datetime.utcnow()
        db.session.commit()


@app.after_request
def after_request(response):
    """Execute after each request"""
    if hasattr(g, 'request_start_time'):
        elapsed = datetime.utcnow() - g.request_start_time
        app.logger.debug(f"Request processed in {elapsed.total_seconds():.4f} seconds")
    return response


@app.context_processor
def inject_current_year():
    """Inject the current year into all templates"""
    return {'current_year': datetime.utcnow().year}


# Register blueprints
from routes import main_bp, auth_bp, api_bp, admin_bp, error_bp

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(error_bp)


# Register error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return error_bp.handle_404(e)


@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return error_bp.handle_500(e)