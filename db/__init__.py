"""
DB Package initialization
This module provides a common database connection for the entire application.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create the declarative base that will be used for all models
Base = declarative_base()

# Set up database connection
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create engine and session
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    # Handle case where DATABASE_URL is not available
    print("WARNING: DATABASE_URL not set. Database functionality will be limited.")
    engine = None
    SessionLocal = None

def get_db():
    """Get a database session"""
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()