#!/usr/bin/env python3
"""
Run the diagnostics script with proper environment set up from the database.
This is a simple wrapper to ensure all environment variables are correctly set.
"""

import os
import sys
import subprocess
import logging
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("diagnostics_runner")

def setup_environment():
    """Set up environment variables from DATABASE_URL if present"""
    # Set default session secret if not present
    if not os.environ.get("SESSION_SECRET"):
        os.environ["SESSION_SECRET"] = "bcbs_values_session_secret_key_2025"
        logger.info("Setting default SESSION_SECRET")
    
    # Parse DATABASE_URL if it exists
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        logger.info(f"Found DATABASE_URL: {db_url.split('@')[0].split(':')[0]}:*****@{db_url.split('@')[1]}")
        
        try:
            # Parse connection details
            result = urlparse(db_url)
            
            # Set individual environment variables if they don't exist
            if not os.environ.get("PGHOST"):
                os.environ["PGHOST"] = result.hostname or "localhost"
                logger.info(f"Setting PGHOST from DATABASE_URL: {os.environ['PGHOST']}")
            
            if not os.environ.get("PGPORT"):
                os.environ["PGPORT"] = str(result.port or "5432")
                logger.info(f"Setting PGPORT from DATABASE_URL: {os.environ['PGPORT']}")
            
            if not os.environ.get("PGDATABASE"):
                os.environ["PGDATABASE"] = result.path[1:] if result.path else "postgres"
                logger.info(f"Setting PGDATABASE from DATABASE_URL: {os.environ['PGDATABASE']}")
            
            if not os.environ.get("PGUSER"):
                os.environ["PGUSER"] = result.username or "postgres"
                logger.info(f"Setting PGUSER from DATABASE_URL: {os.environ['PGUSER']}")
            
            if not os.environ.get("PGPASSWORD"):
                os.environ["PGPASSWORD"] = result.password or ""
                logger.info("Setting PGPASSWORD from DATABASE_URL: ********")
        except Exception as e:
            logger.error(f"Error parsing DATABASE_URL: {str(e)}")
    else:
        logger.warning("DATABASE_URL not found. Using default database settings.")
        # Set default values if DATABASE_URL is not available
        os.environ.setdefault("PGHOST", "localhost")
        os.environ.setdefault("PGPORT", "5432")
        os.environ.setdefault("PGDATABASE", "postgres")
        os.environ.setdefault("PGUSER", "postgres")
        os.environ.setdefault("PGPASSWORD", "postgres")
        
        # Construct DATABASE_URL from individual components
        os.environ["DATABASE_URL"] = f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
        logger.info(f"Constructed DATABASE_URL from environment variables")
    
    # Set up API keys with default values if not present
    os.environ.setdefault("API_KEY", "bcbs_values_api_key_2025")
    os.environ.setdefault("BCBS_VALUES_API_KEY", "bcbs_values_api_key_2025")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("ENABLE_CACHING", "true")

def run_diagnostics():
    """Run the diagnose_env.py script"""
    logger.info("Starting environment diagnostics...")
    
    # Set up the environment
    setup_environment()
    
    # Check if diagnose_env.py exists
    if not os.path.exists("diagnose_env.py"):
        logger.error("diagnose_env.py not found. Cannot run diagnostics.")
        return 1
    
    # Make diagnose_env.py executable if it's not already
    if not os.access("diagnose_env.py", os.X_OK):
        try:
            os.chmod("diagnose_env.py", 0o755)
            logger.info("Made diagnose_env.py executable")
        except Exception as e:
            logger.warning(f"Could not make diagnose_env.py executable: {str(e)}")
    
    # Run the diagnostics script
    logger.info("Running diagnose_env.py...")
    try:
        # Try running with Python first
        process = subprocess.run(
            [sys.executable, "diagnose_env.py"],
            check=False,
            env=os.environ
        )
        return process.returncode
    except Exception as e:
        logger.error(f"Error running diagnose_env.py: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run_diagnostics())