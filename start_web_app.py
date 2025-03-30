"""
BCBS Values Web Application Server
This script starts the Flask-based web application for BCBS Values.
"""
import os
import sys
import logging
from datetime import datetime

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = f"logs/webapp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Generate a secure secret key if not provided
if not os.environ.get("SESSION_SECRET"):
    import secrets
    os.environ["SESSION_SECRET"] = secrets.token_hex(16)
    logging.warning("Generated temporary SESSION_SECRET. Please set this in environment for production.")

try:
    from app import app
    logging.info("Starting BCBS Values Web Application...")
    app.run(host="0.0.0.0", port=5001, debug=True)
except ImportError as e:
    logging.error(f"Failed to import app: {e}")
    logging.error("Make sure app.py exists and all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error starting web application: {e}")
    sys.exit(1)