"""
Run script to start the BCBS Values application with the Flask and FastAPI servers.
This script will start both servers in separate processes, enabling the full
system functionality.
"""
import os
import subprocess
import sys
import time
import threading
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the Flask port (main app) and FastAPI port
FLASK_PORT = 5000
API_PORT = 8000

def run_flask_app():
    """Run the Flask application using gunicorn."""
    logger.info(f"Starting Flask application on port {FLASK_PORT}")
    try:
        # Use gunicorn for production-ready server
        subprocess.run([
            "gunicorn",
            "--bind", f"0.0.0.0:{FLASK_PORT}",
            "--reuse-port",
            "--reload",
            "main:app"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Flask application failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Flask application stopped by user")

def run_fastapi_app():
    """Run the FastAPI application using uvicorn."""
    logger.info(f"Starting FastAPI application on port {API_PORT}")
    try:
        # Set the API port environment variable
        os.environ["API_PORT"] = str(API_PORT)
        
        # Run using uvicorn
        import uvicorn
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=API_PORT,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"FastAPI application failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("FastAPI application stopped by user")

def main():
    """Main function to start both applications."""
    logger.info("Starting BCBS Values integrated application")
    
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start FastAPI directly (in main thread)
    run_fastapi_app()
    
if __name__ == "__main__":
    main()