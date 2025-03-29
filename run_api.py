"""
Simple script to run the FastAPI application.
"""
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)