import os
import sys
import logging
from flask import Flask, send_from_directory, redirect, url_for, render_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='.', template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs-dashboard-secret-key")

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return send_from_directory('.', 'dashboard_static.html')

@app.route('/dashboard')
def dashboard():
    """Serve the static dashboard."""
    return send_from_directory('.', 'dashboard_static.html')

@app.route('/bcbs-dashboard')
def bcbs_dashboard():
    """Serve the BCBS dashboard."""
    return send_from_directory('.', 'bcbs_dashboard.html')

@app.route('/interactive-dashboard')
def interactive_dashboard():
    """Serve the interactive dashboard from templates."""
    try:
        return send_from_directory('templates', 'reactive_dashboard.html')
    except:
        logger.error("Interactive dashboard template not found")
        return send_from_directory('.', 'dashboard_static.html')

@app.route('/demo')
def demo():
    """Serve the demo dashboard."""
    try:
        return send_from_directory('.', 'dashboard_demo.html')
    except:
        logger.error("Demo dashboard not found")
        return send_from_directory('.', 'dashboard_static.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    try:
        return send_from_directory('.', path)
    except:
        logger.error(f"File not found: {path}")
        return send_from_directory('.', '404.html'), 404

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return send_from_directory('.', '404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return "Internal Server Error", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Flask server on {host}:{port}")
    logger.info("Available routes:")
    logger.info("  / - Main dashboard")
    logger.info("  /dashboard - Static dashboard")
    logger.info("  /bcbs-dashboard - BCBS dashboard")
    logger.info("  /interactive-dashboard - Interactive dashboard")
    logger.info("  /demo - Demo dashboard")
    
    app.run(host=host, port=port, debug=True)