from flask import Flask, render_template, send_from_directory, redirect, url_for
import os
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bcbs_values_session_secret_key_2025")

@app.route('/')
def index():
    """Redirect to interactive dashboard page"""
    return redirect('/interactive-dashboard')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard_static.html file"""
    # Read the HTML file
    with open('dashboard_static.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/interactive-dashboard')
def interactive_dashboard():
    """Render the new React-based interactive dashboard"""
    return render_template('reactive_dashboard.html', title='Benton County Property Valuation Dashboard')

@app.route('/demo')
def demo():
    """Serve the dashboard_demo.html file"""
    # Read the HTML file
    with open('dashboard_demo.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files from the static directory"""
    return send_from_directory('static', path)

@app.route('/<path:path>')
def serve_file(path):
    """Serve static files from the root directory"""
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)