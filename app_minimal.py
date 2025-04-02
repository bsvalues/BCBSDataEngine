
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/')
def index():
    return """
    <html>
        <head><title>BCBS Values Platform</title></head>
        <body>
            <h1>BCBS Values Platform</h1>
            <p>System is operational.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
