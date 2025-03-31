"""
Startup script for the BCBS Values web application.
This script seeds the database with sample data and then runs the web application.
"""
import os
import sys
import logging
from datetime import datetime

from app import app, db
import seed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize the database with seed data if needed"""
    logger.info("Checking database...")
    try:
        with app.app_context():
            # Create tables if they don't exist
            db.create_all()
            
            # Seed the database with sample data
            seed_data.seed_database()
            
            logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def main():
    """Main entry point"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting BCBS Values web application at {timestamp}")
    
    # Initialize the database
    init_database()
    
    # Start the web application
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()