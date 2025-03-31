"""
Main entry point for BCBS Values application.

This module serves as the entry point for the BCBS Values application, handling:
1. Initialization of all components
2. ETL pipeline execution
3. Database loading
4. Valuation engine initialization
5. API startup

All major operations are logged with timestamps and detailed diagnostics.
Errors during startup are captured and logged to help diagnose issues.

Author: BCBS Engineering Team
Last Updated: 2025-03-31
"""
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
import json
import logging
from logging.handlers import RotatingFileHandler
import argparse

# Import app and components
from app import app, db
import models  # Import models to ensure they're registered with SQLAlchemy
from src import valuation

# Import ETL utilities
try:
    from etl.pipeline import ETLPipeline
except ImportError:
    ETLPipeline = None

# Import logging utilities
from utils.logging_config import setup_logging, get_etl_logger, get_api_logger, get_valuation_logger

# Configure main application logger
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a timestamped log filename
log_filename = os.path.join(LOGS_DIR, f"bcbs_startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up the main logger
logger = setup_logging(
    log_level=logging.DEBUG,
    log_file=log_filename,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    module_name="main"
)

# Get component-specific loggers
etl_logger = get_etl_logger()
api_logger = get_api_logger()
valuation_logger = get_valuation_logger()

def initialize_database():
    """
    Initialize the database connection and create tables if they don't exist.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    logger.info("Initializing database connection")
    try:
        with app.app_context():
            # Check if database tables exist
            table_names = db.engine.table_names()
            logger.debug(f"Existing tables: {', '.join(table_names) if table_names else 'None'}")
            
            # Create tables if they don't exist
            if not table_names:
                logger.info("No tables found. Creating database schema.")
                db.create_all()
                logger.info("Database schema created successfully")
            else:
                logger.info(f"Database contains {len(table_names)} tables. Schema exists.")
            
            # Log database connection details (excluding sensitive info)
            db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
            sanitized_uri = db_uri.split('@')[-1] if '@' in db_uri else db_uri
            logger.info(f"Connected to database: {sanitized_uri}")
            
            return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_valuation_engine():
    """
    Initialize the property valuation engine.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    logger.info("Initializing valuation engine")
    try:
        # Call initialization function from valuation module
        valuation_logger.info("Loading valuation models and parameters")
        
        # Check if valuation module has required components
        if hasattr(valuation, 'initialize'):
            valuation.initialize()
            logger.info("Valuation engine initialized successfully")
        else:
            logger.warning("Valuation engine initialize() method not found. Using default configuration.")
        
        # Log valuation model configuration
        if hasattr(valuation, 'get_model_info'):
            model_info = valuation.get_model_info()
            logger.info(f"Valuation model information: {model_info}")
        
        return True
    except Exception as e:
        logger.error(f"Valuation engine initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_etl_pipeline(sources=None, validate_only=False):
    """
    Run the ETL pipeline to extract, transform, and load property data.
    
    Args:
        sources (list): List of data sources to process (None for all)
        validate_only (bool): If True, only validate data without loading
    
    Returns:
        dict: Results of the ETL pipeline run
    """
    logger.info(f"Starting ETL pipeline with sources={sources}, validate_only={validate_only}")
    etl_start_time = time.time()
    
    try:
        if ETLPipeline is None:
            logger.warning("ETL Pipeline module not found. Skipping ETL process.")
            return {
                'status': 'skipped',
                'error': 'ETL Pipeline module not available',
                'timestamp': datetime.now().isoformat()
            }
        
        # Initialize and run ETL pipeline
        etl_logger.info("Initializing ETL pipeline")
        pipeline = ETLPipeline()
        
        # Log ETL configuration
        etl_logger.info(f"ETL Configuration: sources={sources}, validate_only={validate_only}")
        
        # Run the pipeline
        etl_logger.info("Executing ETL pipeline")
        results = pipeline.run_pipeline(sources=sources, validate_only=validate_only)
        
        # Calculate execution time
        execution_time = time.time() - etl_start_time
        
        # Log results summary
        etl_logger.info(f"ETL pipeline completed in {execution_time:.2f} seconds")
        etl_logger.info(f"Status: {results.get('status', 'unknown')}")
        etl_logger.info(f"Records processed: {results.get('records_processed', 0)}")
        etl_logger.info(f"Valid records: {results.get('valid_records', 0)}")
        etl_logger.info(f"Invalid records: {results.get('invalid_records', 0)}")
        
        # Include execution time in results
        results['execution_time_seconds'] = execution_time
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

def initialize_api():
    """
    Initialize the API components.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    logger.info("Initializing API components")
    try:
        # Log API configuration
        api_logger.info("Loading API configuration")
        
        # Check for API key
        api_key = os.environ.get('API_KEY')
        if api_key:
            api_logger.info("API key found in environment")
        else:
            api_logger.warning("No API key found in environment. Using default key for development.")
        
        # Check for additional API-related configuration
        jwt_secret = os.environ.get('JWT_SECRET')
        if jwt_secret:
            api_logger.info("JWT secret found in environment")
        else:
            api_logger.warning("No JWT secret found in environment. Using default for development.")
        
        # Log available endpoints (this would ideally come from a route registry)
        api_logger.info("API endpoints available: /api/valuations, /api/etl-status, /api/agent-status")
        
        # Log API initialization success
        logger.info("API components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"API initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='BCBS Values Application')
    parser.add_argument('--run-etl', action='store_true', help='Run ETL pipeline during startup')
    parser.add_argument('--etl-sources', nargs='+', help='ETL sources to process (space-separated)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate ETL data without loading')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Logging level')
    
    return parser.parse_args()

def startup_diagnostics():
    """
    Gather and log diagnostic information about the system.
    """
    logger.info("=============== STARTUP DIAGNOSTICS ===============")
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    
    # Environment
    logger.info(f"Environment: {'Production' if os.environ.get('PRODUCTION') else 'Development'}")
    
    # Database configuration
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    logger.info(f"Database URI: {'PostgreSQL' if 'postgresql' in db_uri else 'SQLite' if 'sqlite' in db_uri else 'Other'}")
    
    # API key status
    logger.info(f"API key configured: {'Yes' if os.environ.get('API_KEY') else 'No'}")
    
    # Check for ETL configuration
    etl_config_file = os.path.join(os.path.dirname(__file__), 'configs', 'etl_config.json')
    logger.info(f"ETL configuration exists: {'Yes' if os.path.exists(etl_config_file) else 'No'}")
    
    # Log directory
    logger.info(f"Log directory: {LOGS_DIR}")
    
    # Available disk space for logs
    if sys.platform != 'win32':
        try:
            import shutil
            stats = shutil.disk_usage(LOGS_DIR)
            logger.info(f"Available disk space: {stats.free / (1024**3):.2f} GB")
        except Exception as e:
            logger.warning(f"Could not check disk space: {str(e)}")
    
    logger.info("====================================================")

def main():
    """
    Main entry point for the application.
    """
    # Record start time for performance tracking
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure log level based on arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Log level set to {args.log_level}")
    
    # Log startup banner
    logger.info("="*50)
    logger.info("BCBS VALUES APPLICATION STARTING")
    logger.info(f"Startup time: {datetime.now().isoformat()}")
    logger.info("="*50)
    
    # Run startup diagnostics
    startup_diagnostics()
    
    # Initialize components
    success = True
    
    # Initialize database
    db_success = initialize_database()
    if not db_success:
        logger.warning("Database initialization had errors. Continuing with limited functionality.")
        success = False
    
    # Initialize valuation engine
    valuation_success = initialize_valuation_engine()
    if not valuation_success:
        logger.warning("Valuation engine initialization had errors. Continuing with limited functionality.")
        success = False
    
    # Run ETL pipeline if requested
    if args.run_etl:
        etl_results = run_etl_pipeline(sources=args.etl_sources, validate_only=args.validate_only)
        if etl_results.get('status') == 'failed':
            logger.warning("ETL pipeline had errors. Continuing with existing data.")
            success = False
    
    # Initialize API components
    api_success = initialize_api()
    if not api_success:
        logger.warning("API initialization had errors. API endpoints may not function correctly.")
        success = False
    
    # Calculate and log startup time
    startup_time = time.time() - start_time
    logger.info(f"Startup completed in {startup_time:.2f} seconds")
    
    # Log startup status
    if success:
        logger.info("All components initialized successfully. Application ready.")
    else:
        logger.warning("Some components had initialization errors. Application running with limitations.")
    
    # Provide instructions on log access
    print(f"Application startup logs are available at: {log_filename}")
    print(f"For continuous API logs, check: {os.path.join(LOGS_DIR, 'api_YYYYMMDD_HHMMSS.log')}")
    print(f"For ETL pipeline logs, check: {os.path.join(LOGS_DIR, 'etl_pipeline_YYYYMMDD_HHMMSS.log')}")
    print(f"For valuation engine logs, check: {os.path.join(LOGS_DIR, 'valuation.log')}")
    
    # Return the Flask app for WSGI servers
    return app

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run initialization
    app = main()
    
    # Start the Flask development server
    logger.info(f"Starting Flask server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=True)