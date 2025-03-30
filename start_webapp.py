from flask import Flask, render_template, send_from_directory, redirect
import os
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs_values_session_secret_key_2025")

@app.route('/')
def index():
    """Redirect to dashboard page"""
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard_static.html file"""
    # Read the HTML file
    with open('dashboard_static.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/demo')
def demo():
    """Serve the dashboard_demo.html file"""
    # Read the HTML file
    with open('dashboard_demo.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/<path:path>')
def serve_file(path):
    """Serve static files from the root directory"""
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)