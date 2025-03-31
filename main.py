import os
import logging
from app import app
import routes  # Import routes to register them with the app

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Get port from environment variable or use default 5000
    port = int(os.environ.get("PORT", 5000))
    
    logger.info(f"Starting BCBS Property Valuation app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)