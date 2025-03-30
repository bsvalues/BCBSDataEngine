#!/usr/bin/env python3
"""
Integrated server script that starts both Flask and FastAPI servers in separate processes.
This ensures that both web interface and API endpoints are accessible at the same time.
"""
import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
FLASK_PORT = 5001
API_PORT = 8000

def start_flask_app():
    """Start the Flask web application"""
    logger.info("Starting Flask application on port %s", FLASK_PORT)
    flask_process = subprocess.Popen(
        [sys.executable, "main.py", "--web"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return flask_process

def start_api_server():
    """Start the FastAPI server"""
    logger.info("Starting FastAPI server on port %s", API_PORT)
    api_process = subprocess.Popen(
        [sys.executable, "run_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return api_process

def main():
    """Main function to run both servers"""
    logger.info("Starting BCBS_Values integrated server")
    
    # Start Flask application
    flask_process = start_flask_app()
    
    # Start API server
    api_process = start_api_server()
    
    # Monitor outputs
    logger.info("Monitoring server outputs (press Ctrl+C to stop)")
    try:
        while True:
            # Check Flask process output
            if flask_process.poll() is not None:
                logger.error("Flask process terminated with code %d", flask_process.returncode)
                break
                
            # Check API process output
            if api_process.poll() is not None:
                logger.error("API process terminated with code %d", api_process.returncode)
                break
                
            # Read and display process outputs
            while True:
                flask_output = flask_process.stdout.readline()
                if not flask_output:
                    break
                print(f"[Flask] {flask_output.strip()}")
                
            while True:
                api_output = api_process.stdout.readline()
                if not api_output:
                    break
                print(f"[API] {api_output.strip()}")
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
    finally:
        # Terminate processes
        for proc in [flask_process, api_process]:
            if proc.poll() is None:  # Process is still running
                logger.info("Terminating process %d", proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process %d did not terminate, killing it", proc.pid)
                    proc.kill()
    
    logger.info("All servers stopped")
    
if __name__ == "__main__":
    main()