#!/usr/bin/env python
"""
Flask Server for the BCBS Values Dashboard
This server uses Flask to serve the dashboard and handle API requests.
"""

import os
import sys
import logging
from flask import Flask, send_file, jsonify, request

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)

# Set the secret key
app.secret_key = os.environ.get("SESSION_SECRET", "development_key")

# Constants
DEFAULT_PAGE = 'dashboard_static.html'
PORT = 5002

@app.route('/')
def index():
    """Serve the default dashboard page."""
    logger.info(f"Serving default dashboard page: {DEFAULT_PAGE}")
    return send_file(DEFAULT_PAGE)

@app.route('/<path:path>')
def serve_file(path):
    """Serve any file in the current directory."""
    logger.info(f"Serving file: {path}")
    try:
        return send_file(path)
    except Exception as e:
        logger.error(f"Error serving file {path}: {str(e)}")
        return jsonify({"error": f"File not found or could not be served: {str(e)}"}), 404

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.path}")
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"500 error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info(f"Starting Flask server on http://0.0.0.0:{PORT}/")
    logger.info(f"Default page: {DEFAULT_PAGE}")
    
    # Verify that the default page exists
    if not os.path.exists(DEFAULT_PAGE):
        logger.warning(f"WARNING: Default page '{DEFAULT_PAGE}' not found in current directory.")
        logger.info(f"Current directory contains: {', '.join(os.listdir('.'))}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)