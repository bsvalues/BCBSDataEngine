"""
Main entry point for the BCBS Values application.
"""
from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)