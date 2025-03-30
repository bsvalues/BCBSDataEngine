"""
Simple HTTP server for the BCBS_Values real estate valuation system.
"""
import http.server
import socketserver
import json
import time
from datetime import datetime
import os
import sys

# Add the current directory to the Python path if needed
if "." not in sys.path:
    sys.path.append(".")

# Set the port
PORT = 5000

class BCBSValuesHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for BCBS Values requests"""
    
    def _set_headers(self, status_code=200, content_type="application/json"):
        """Set the response headers"""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def _send_json_response(self, data, status_code=200):
        """Send a JSON response"""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data).encode('utf-8'))
        
    def _send_html_response(self, html_content, status_code=200):
        """Send an HTML response"""
        self._set_headers(status_code, "text/html")
        self.wfile.write(html_content.encode('utf-8'))
        
    def _get_sample_properties(self):
        """Return sample property data"""
        return [
            {
                "property_id": "BENT-12345",
                "address": "123 Main St, Richland, WA 99352",
                "estimated_value": 345000,
                "confidence_score": 0.85,
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "bedrooms": 3,
                "bathrooms": 2,
                "square_feet": 1850,
                "year_built": 2005,
                "property_type": "Single Family"
            },
            {
                "property_id": "BENT-67890",
                "address": "456 Oak Ave, Kennewick, WA 99336",
                "estimated_value": 289000,
                "confidence_score": 0.78,
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "bedrooms": 2,
                "bathrooms": 1.5,
                "square_feet": 1550,
                "year_built": 1995,
                "property_type": "Single Family"
            },
            {
                "property_id": "BENT-23456",
                "address": "789 Pine Ln, Pasco, WA 99301",
                "estimated_value": 425000,
                "confidence_score": 0.92,
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "bedrooms": 4,
                "bathrooms": 3,
                "square_feet": 2200,
                "year_built": 2018,
                "property_type": "Single Family"
            }
        ]
        
    def _get_property_by_id(self, property_id):
        """Get property by ID"""
        properties = self._get_sample_properties()
        for prop in properties:
            if prop["property_id"] == property_id:
                return prop
        return None
        
    def _get_validation_results(self):
        """Return sample validation results"""
        return {
            "validation_passed": True,
            "total_records": 150,
            "valid_records": 148,
            "invalid_records": 2,
            "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "validation_results": {
                "missing_values": {
                    "count": 1,
                    "details": ["Property BENT-45678 missing square_feet value"]
                },
                "invalid_values": {
                    "count": 1,
                    "details": ["Property BENT-98765 has invalid year_built (value: 3005)"]
                }
            }
        }
    
    def do_GET(self):
        """Handle GET requests"""
        # Home page
        if self.path == "/" or self.path == "/index.html":
            with open("templates/index.html", "r") as f:
                content = f.read()
            self._send_html_response(content)
            return
            
        # Properties page
        elif self.path == "/properties" or self.path == "/properties.html":
            try:
                with open("templates/properties.html", "r") as f:
                    content = f.read()
                self._send_html_response(content)
            except FileNotFoundError:
                # Fallback to basic HTML if template doesn't exist
                properties = self._get_sample_properties()
                content = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>BCBS Values - Properties</title>
                    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
                </head>
                <body>
                    <div class="container mt-4">
                        <h1>Benton County Properties</h1>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Property ID</th>
                                        <th>Address</th>
                                        <th>Estimated Value</th>
                                        <th>Confidence</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                """
                
                for prop in properties:
                    content += f"""
                    <tr>
                        <td>{prop['property_id']}</td>
                        <td>{prop['address']}</td>
                        <td>${prop['estimated_value']:,.2f}</td>
                        <td>{prop['confidence_score'] * 100:.1f}%</td>
                        <td><a href="/property/{prop['property_id']}" class="btn btn-sm btn-primary">View</a></td>
                    </tr>
                    """
                
                content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                self._send_html_response(content)
            return
            
        # API Endpoints
        elif self.path == "/api/properties" or self.path == "/api/valuations":
            properties = self._get_sample_properties()
            self._send_json_response(properties)
            return
            
        elif self.path.startswith("/api/property/") or self.path.startswith("/api/valuation/"):
            # Extract property ID from path
            parts = self.path.split("/")
            if len(parts) != 4:
                self._send_json_response({"error": "Invalid property ID"}, 400)
                return
                
            property_id = parts[3]
            property_data = self._get_property_by_id(property_id)
            
            if property_data:
                self._send_json_response(property_data)
            else:
                self._send_json_response({"error": "Property not found"}, 404)
            return
            
        elif self.path == "/api/validation" or self.path == "/api/etl-status":
            validation_results = self._get_validation_results()
            self._send_json_response(validation_results)
            return
            
        # Serve static files or 404
        else:
            try:
                super().do_GET()
            except:
                self._send_json_response({"error": "Not found"}, 404)
                
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self._set_headers()
        
    def log_message(self, format, *args):
        """Override to provide custom logging"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {self.address_string()} - {format % args}")

def run_server():
    """Run the server"""
    handler = BCBSValuesHandler
    
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Serving BCBS Values at http://0.0.0.0:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user")
        finally:
            httpd.server_close()
            print("Server closed")

if __name__ == "__main__":
    print("Starting BCBS Values simple server...")
    run_server()