import os
import sys
import http.server
import socketserver
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use the same port that's defined in .replit file
PORT = 5002

class BCBSHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for BCBS Values Platform"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info("%s - %s", self.address_string(), format % args)
    
    def end_headers(self):
        """Add CORS headers to allow cross-origin requests"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start the HTTP server"""
    try:
        # Print server information
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        
        # List all HTML files in the current directory
        html_files = [f for f in os.listdir('.') if f.endswith('.html')]
        print("HTML files found:")
        for html_file in html_files:
            print(f"  - {html_file}")
        
        # Start the server
        server_address = ('0.0.0.0', PORT)
        httpd = socketserver.TCPServer(server_address, BCBSHandler)
        
        print(f"Serving BCBS Values Platform Dashboard at http://0.0.0.0:{PORT}/")
        print(f"Access the dashboard at http://0.0.0.0:{PORT}/dashboard.html")
        print("Press Ctrl+C to stop the server")
        
        # Start the server
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()