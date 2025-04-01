#!/usr/bin/env python3
"""
Enhanced HTTP Server for BCBS Values Platform

Features:
- Proper MIME type detection for different file types
- Detailed error handling and logging
- Port availability checking before startup
- Improved caching headers
- Graceful shutdown handling
"""
import http.server
import socketserver
import socket
import os
import sys
import time
import signal
import logging
import mimetypes
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger('bcbs_server')

# Configuration
PORT = 5002
HOST = "0.0.0.0"
PID_FILE = "server.pid"
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Initialize mime types
mimetypes.init()
# Add any missing mime types
if not mimetypes.types_map.get('.js'):
    mimetypes.add_type('application/javascript', '.js')
if not mimetypes.types_map.get('.css'):
    mimetypes.add_type('text/css', '.css')

class BCBSHandler(http.server.SimpleHTTPRequestHandler):
    """Enhanced HTTP request handler for BCBS Values Platform."""
    
    # Override the default log format for cleaner logs
    def log_message(self, format, *args):
        """Log message with client address."""
        logger.info(f"{self.client_address[0]} - {format % args}")
    
    def log_error(self, format, *args):
        """Log error with client address."""
        logger.error(f"{self.client_address[0]} - {format % args}")
    
    def end_headers(self):
        """Add custom headers to all responses."""
        # Add caching headers
        self.send_header('Cache-Control', 'max-age=86400')  # 24 hours
        self.send_header('X-Server', 'BCBS Values Platform')
        super().end_headers()
    
    def guess_type(self, path):
        """Determine Content-Type header for the given path."""
        content_type = mimetypes.guess_type(path)[0]
        if content_type is None:
            # Default to binary if we can't determine type
            return 'application/octet-stream'
        return content_type
    
    def send_error_page(self, code, message=None):
        """Send a custom error page."""
        try:
            # Try to serve our custom 404.html for 404 errors
            if code == 404 and os.path.exists('404.html'):
                self.error_message_format = open('404.html', 'r').read()
            else:
                self.error_message_format = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error {0}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 30px; text-align: center; }}
                        h1 {{ color: #d9534f; }}
                        .error-container {{ max-width: 800px; margin: 0 auto; }}
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <h1>Error {0}</h1>
                        <p>{1}</p>
                        <p>BCBS Values Platform Server</p>
                        <a href="/">Return to Home</a>
                    </div>
                </body>
                </html>
                """
            self.send_error(code, message)
        except Exception as e:
            logger.error(f"Error sending error page: {e}")
            # Fall back to default error handling
            super().send_error(code, message)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            # Special case for root path
            if self.path == '/':
                self.path = '/index.html'
            
            # Handle API requests
            if self.path.startswith('/api/'):
                return self.handle_api_request()
            
            # Try to serve the requested file
            file_path = self.translate_path(self.path)
            
            if os.path.isfile(file_path):
                # File exists, serve it
                return super().do_GET()
            else:
                # File doesn't exist, send 404
                self.send_error_page(404, f"File {self.path} not found")
                
        except Exception as e:
            logger.error(f"Error handling GET request: {e}")
            self.send_error_page(500, f"Internal server error: {str(e)}")
    
    def handle_api_request(self):
        """Handle API endpoint requests returning JSON data."""
        # Parse the API endpoint from the path
        endpoint = self.path[5:]  # Remove '/api/' prefix
        
        if endpoint == 'status':
            # Server status endpoint
            status_data = {
                'status': 'running',
                'uptime': self.server.get_uptime(),
                'server_time': datetime.now().isoformat(),
                'endpoints': ['/api/status', '/api/data']
            }
            self.send_json_response(200, status_data)
            
        elif endpoint == 'data':
            # Sample data for dashboard
            data = {
                'property_count': 12542,
                'average_value': 425000,
                'recent_valuations': [
                    {'id': 'PROP001', 'address': '123 Main St', 'value': 350000, 'date': '2025-03-28'},
                    {'id': 'PROP002', 'address': '456 Oak Ave', 'value': 550000, 'date': '2025-03-29'},
                    {'id': 'PROP003', 'address': '789 Pine Blvd', 'value': 425000, 'date': '2025-03-30'}
                ],
                'neighborhoods': ['Downtown', 'Westside', 'Harbor View', 'Northgate'],
                'data_updated': datetime.now().isoformat()
            }
            self.send_json_response(200, data)
            
        else:
            # Unknown API endpoint
            error = {'error': 'Unknown API endpoint', 'path': self.path}
            self.send_json_response(404, error)
    
    def send_json_response(self, code, data):
        """Send a JSON response with the given status code and data."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

class BCBSHTTPServer(socketserver.TCPServer):
    """Enhanced HTTP server with additional functionality."""
    
    allow_reuse_address = True
    
    def __init__(self, server_address, RequestHandlerClass):
        """Initialize server with start time for uptime tracking."""
        self.start_time = datetime.now()
        super().__init__(server_address, RequestHandlerClass)
    
    def get_uptime(self):
        """Return server uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

def is_port_available(port, host='localhost'):
    """Check if a port is available on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def write_pid_file():
    """Write the current process ID to the PID file."""
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"PID {os.getpid()} written to {PID_FILE}")
    except Exception as e:
        logger.error(f"Failed to write PID file: {e}")

def remove_pid_file():
    """Remove the PID file."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            logger.info(f"Removed PID file {PID_FILE}")
    except Exception as e:
        logger.error(f"Failed to remove PID file: {e}")

def setup_signal_handlers(httpd):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        httpd.server_close()
        remove_pid_file()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # CTRL+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def check_required_files():
    """Check if required files exist."""
    required_files = ['index.html', 'dashboard.html']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        logger.info("Creating minimal index.html as fallback")
        
        # Create a minimal index.html if it doesn't exist
        if 'index.html' in missing_files:
            with open('index.html', 'w') as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 30px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #0275d8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BCBS Values Platform</h1>
        <p>Welcome to the BCBS Values Platform server.</p>
        <p><a href="/dashboard.html">View Dashboard</a></p>
    </div>
</body>
</html>""")
            logger.info("Created fallback index.html")

def main():
    """Start the HTTP server with fault tolerance."""
    logger.info("BCBS Values Platform Enhanced Server")
    logger.info("====================================")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Current Working Directory: {os.getcwd()}")
    
    # Check for required files
    check_required_files()
    
    # List HTML files
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    logger.info("HTML files found:")
    for html_file in html_files:
        logger.info(f"  - {html_file}")
    
    # Check if port is available
    retries = 0
    while retries < MAX_RETRIES:
        if is_port_available(PORT, HOST):
            break
        
        logger.warning(f"Port {PORT} is not available. Retrying in {RETRY_DELAY} seconds...")
        retries += 1
        time.sleep(RETRY_DELAY)
    
    if retries >= MAX_RETRIES:
        logger.error(f"Port {PORT} is not available after {MAX_RETRIES} retries. Exiting.")
        sys.exit(1)
    
    # Start the server
    try:
        # Write PID file for process management
        write_pid_file()
        
        # Create and start server
        httpd = BCBSHTTPServer((HOST, PORT), BCBSHandler)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(httpd)
        
        logger.info(f"Starting server at http://{HOST}:{PORT}")
        logger.info("Press Ctrl+C to stop")
        
        # Start the server
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)
    finally:
        # Clean up
        remove_pid_file()

if __name__ == "__main__":
    main()