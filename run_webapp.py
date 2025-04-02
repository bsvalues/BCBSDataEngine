#!/usr/bin/env python3
"""
BCBS Values Platform WebApp Runner

This script starts the Multi-Implementation Server and
serves the BCBS Values Platform web application.
"""

import os
import sys
import logging
import json
import argparse
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('webapp.log')
    ]
)
logger = logging.getLogger(__name__)

def load_script_path():
    """
    Get the absolute path to this script.
    
    Returns:
        str: Absolute path to this script directory
    """
    return os.path.dirname(os.path.abspath(__file__))

def main():
    """
    Main entry point for the WebApp runner.
    """
    parser = argparse.ArgumentParser(description="BCBS Values Platform WebApp Runner")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5002, 
        help="Port to run the server on (default: 5002)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--content-dir", 
        type=str, 
        default=".", 
        help="Directory containing content to serve (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Log startup information
    logger.info("Starting BCBS Values Platform WebApp")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Script directory: {load_script_path()}")
    
    # Import the multi-implementation server
    try:
        script_dir = load_script_path()
        sys.path.append(script_dir)
        
        # First try direct import
        try:
            from multi_server import MultiImplementationServer
            logger.info("Imported MultiImplementationServer directly")
        except ImportError:
            # Try with full path
            sys.path.append(os.path.join(script_dir, "multi_server.py"))
            from multi_server import MultiImplementationServer
            logger.info("Imported MultiImplementationServer with path")
        
        # Start the server
        server = MultiImplementationServer(
            host=args.host,
            port=args.port,
            content_dir=args.content_dir
        )
        
        # Log the environment
        environment = server.environment
        logger.info(f"Environment: {json.dumps(environment, indent=2)}")
        
        # Start the server
        logger.info("Starting server...")
        result = server.start()
        
        if result["success"]:
            logger.info(f"Server started successfully with {result['implementation']} implementation")
            logger.info(f"Listening on http://{args.host}:{args.port}/")
            
            # Handle exit gracefully
            try:
                # Keep the server running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping server...")
                server.stop()
                logger.info("Server stopped")
        else:
            logger.error("Failed to start server")
            return 1
        
    except Exception as e:
        logger.error(f"Error starting WebApp: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())