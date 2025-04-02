from flask import Flask, jsonify
import logging
import os # Added back since os is not used in edited code, but may be used in original code
from datetime import datetime

# Configure logging (from original code)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/')
def index():
    return "System Operational"

@app.errorhandler(404) # Added back from original code
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500) # Added back from original code
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)