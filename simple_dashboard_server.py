#!/usr/bin/env python
"""
Simple HTTP Server for BCBS Dashboard using Python's built-in http.server module
This server doesn't require any external dependencies and should work with any Python installation.
"""

import http.server
import socketserver
import os
import sys
import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DashboardServer")

# Constants
PORT = 8080
DEFAULT_PAGE = 'dashboard_static.html'

class BCBSDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for BCBS Dashboard"""
    
    def do_GET(self):
        """Handle GET requests"""
        logger.info(f"GET request for: {self.path}")
        
        # Handle root path requests to serve the default dashboard
        if self.path == '/' or self.path == '':
            self.path = f"/{DEFAULT_PAGE}"
            logger.info(f"Serving default page: {self.path}")
        
        try:
            # Try to serve the requested file
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        except Exception as e:
            logger.error(f"Error serving {self.path}: {str(e)}")
            self.send_error(404, f"File not found: {self.path}")
    
    def log_message(self, format, *args):
        """Override to log to our configured logger instead of stderr"""
        logger.info("%s - - [%s] %s" % 
            (self.address_string(),
             self.log_date_time_string(),
             format % args))

def main():
    """Main function to start the server"""
    # Check if default page exists
    if not os.path.exists(DEFAULT_PAGE):
        logger.warning(f"WARNING: Default page '{DEFAULT_PAGE}' not found!")
        logger.info(f"Current directory contains: {', '.join(os.listdir('.'))}")
    
    logger.info(f"Starting HTTP server on http://0.0.0.0:{PORT}/")
    logger.info(f"Default page: {DEFAULT_PAGE}")
    logger.info(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Server will remain running until manually stopped")
    
    # Create and start the HTTP server
    handler = BCBSDashboardHandler
    httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        httpd.server_close()
        logger.info("Server stopped")

if __name__ == "__main__":
    main()