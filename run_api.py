"""
Script to run the BCBS_Values FastAPI application.

This script launches the FastAPI application for the BCBS_Values
real estate valuation API on port 5001 with automatic reloading
enabled for development convenience.
"""
import uvicorn
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Run the enhanced FastAPI application
    logging.info("Starting BCBS_Values API server on port 5001...")
    uvicorn.run("api:app", host="0.0.0.0", port=5001, reload=True)