from flask import Flask, render_template, send_from_directory
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return send_from_directory('.', '404.html'), 404

if __name__ == '__main__':
    print("Starting BCBS Values Platform Flask Server...")
    print("============================================")
    print(f"Current directory: {os.getcwd()}")
    
    # List HTML files
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    print("HTML files found:")
    for html_file in html_files:
        print(f"  - {html_file}")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5002, debug=True)