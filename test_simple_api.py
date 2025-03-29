"""
A simple FastAPI application for testing.
"""
import os
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI(
    title="Test API",
    description="Simple test API",
    version="1.0.0"
)

@app.get("/")
async def root():
    """API root endpoint, provides basic API information."""
    return {
        "name": "Test API",
        "status": "running",
        "message": "This is a simple test API"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("test_simple_api:app", host="0.0.0.0", port=port, reload=True)