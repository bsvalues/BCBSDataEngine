
from flask import Flask, jsonify
import logging
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/health')
def health_check():
    logger.info("Health check endpoint called")
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/')
def index():
    logger.info("Index endpoint called")
    return "System Operational"

@app.errorhandler(404)
def not_found(error):
    logger.error(f"404 error: {error}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting application server")
    app.run(host='0.0.0.0', port=5000)
