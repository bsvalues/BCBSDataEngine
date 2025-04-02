
from flask import Flask, jsonify
import sys
import platform

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'python_version': platform.python_version(),
        'platform': platform.platform()
    })

@app.route('/')
def index():
    return "BCBS Values Platform - Operational"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
