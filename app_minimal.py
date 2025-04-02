from flask import Flask
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Index endpoint called")
    return "System Operational"

if __name__ == '__main__':
    logger.info("Starting application server")
    app.run(host='0.0.0.0', port=5000)