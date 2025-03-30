"""
A basic FastAPI server with a health check endpoint.

This module provides a simple FastAPI application with a health check 
endpoint that returns a status OK response.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Basic API Server",
    description="A simple API server with health check endpoint",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you'd restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is operational.
    
    Returns:
        dict: Status message indicating API is operational
    """
    logger.info("Health check endpoint accessed")
    return {"status": "OK"}

# Root endpoint that redirects to the health check
@app.get("/")
async def root():
    """
    Root endpoint that provides basic instruction.
    
    Returns:
        dict: A message directing users to the health check endpoint
    """
    return {"message": "API server is running. Visit /health for health check."}

# This conditional is for running the app directly with Python
# Without this, the app would start when imported
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the API server")
    uvicorn.run(
        "basic_api:app", 
        host="0.0.0.0", 
        port=5000, 
        reload=True
    )