"""
Main script for the BCBS_Values web application.
"""
from app import app

# Export the app for Gunicorn to use
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)