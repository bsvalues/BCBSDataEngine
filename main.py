"""
Main entry point for the BCBS Values application.

This module imports the necessary components and starts the web server.
"""
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from app import app
from replit_auth import make_replit_blueprint

# Import routes
import routes

# Set up Replit authentication
app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")

# Use ProxyFix to make sure redirects work with https
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)