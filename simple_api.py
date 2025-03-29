"""
Simple API implementation for the BCBS Values system.
This module provides a simplified HTTP API without external dependencies.
"""
import json
import datetime
import http.server
import socketserver
import urllib.parse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the port to listen on
PORT = 8000

# Sample data for property valuations
SAMPLE_VALUATIONS = [
    {
        "property_id": "PROP-1001",
        "address": "123 Cherry Lane, Richland, WA 99352",
        "estimated_value": 425000.00,
        "confidence_score": 0.92,
        "model_used": "advanced_regression",
        "valuation_date": datetime.datetime.now().isoformat(),
        "features_used": {
            "square_feet": 2450,
            "bedrooms": 4,
            "bathrooms": 2.5,
            "year_built": 1998,
            "lot_size": 12000
        },
        "comparable_properties": [
            {"id": "COMP-101", "address": "125 Cherry Lane", "sale_price": 415000},
            {"id": "COMP-102", "address": "130 Cherry Lane", "sale_price": 432000}
        ]
    },
    {
        "property_id": "PROP-1002",
        "address": "456 Oak Street, Kennewick, WA 99336",
        "estimated_value": 375000.00,
        "confidence_score": 0.88,
        "model_used": "hedonic_price_model",
        "valuation_date": datetime.datetime.now().isoformat(),
        "features_used": {
            "square_feet": 2100,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "year_built": 2005,
            "lot_size": 9500
        },
        "comparable_properties": [
            {"id": "COMP-201", "address": "460 Oak Street", "sale_price": 368000},
            {"id": "COMP-202", "address": "470 Oak Street", "sale_price": 382500}
        ]
    }
]

# Sample ETL status data
ETL_STATUS = {
    "status": "completed",
    "last_run": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(),
    "sources_processed": [
        {"name": "MLS", "status": "success", "records": 1250},
        {"name": "NARRPR", "status": "success", "records": 875},
        {"name": "PACS", "status": "warning", "records": 432}
    ],
    "records_processed": 2557,
    "validation_status": "passed_with_warnings",
    "validation_details": {
        "completeness": {"status": "passed", "score": 98.2},
        "data_types": {"status": "passed", "score": 100.0},
        "numeric_ranges": {"status": "warning", "issues": 17},
        "dates": {"status": "passed", "score": 99.5},
        "duplicates": {"status": "warning", "issues": 5},
        "cross_source": {"status": "passed", "score": 97.8}
    },
    "errors": [
        {
            "source": "PACS",
            "error_type": "validation_warning",
            "message": "15 properties have lot_size outside expected range",
            "severity": "warning"
        },
        {
            "source": "MLS",
            "error_type": "validation_warning",
            "message": "5 properties have duplicate parcel IDs",
            "severity": "warning"
        }
    ]
}

# Sample agent status data
AGENT_STATUS = {
    "agents": [
        {
            "agent_id": "bcbs-bootstrap-commander",
            "name": "BCBS Bootstrap Commander",
            "status": "active",
            "last_active": (datetime.datetime.now() - datetime.timedelta(minutes=15)).isoformat(),
            "current_task": "verifying_dependencies",
            "queue_size": 3,
            "performance_metrics": {
                "tasks_completed": 248,
                "avg_task_time": 35.2,
                "success_rate": 99.2
            }
        },
        {
            "agent_id": "bcbs-cascade-operator",
            "name": "BCBS Cascade Operator",
            "status": "active",
            "last_active": (datetime.datetime.now() - datetime.timedelta(minutes=2)).isoformat(),
            "current_task": "orchestrating_etl_workflow",
            "queue_size": 1,
            "performance_metrics": {
                "tasks_completed": 412,
                "avg_task_time": 127.8,
                "success_rate": 98.7
            }
        },
        {
            "agent_id": "bcbs-tdd-validator",
            "name": "BCBS TDD Validator",
            "status": "idle",
            "last_active": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat(),
            "current_task": None,
            "queue_size": 0,
            "performance_metrics": {
                "tasks_completed": 189,
                "avg_task_time": 45.3,
                "success_rate": 96.8
            }
        }
    ],
    "system_status": "operational",
    "active_agents": 2,
    "tasks_in_progress": 2,
    "tasks_completed_today": 27
}

class BCBSValuesAPIHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for BCBS Values API requests"""
    
    def _set_headers(self, status_code=200, content_type="application/json"):
        """Set the response headers"""
        self.send_response(status_code)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")  # Enable CORS
        self.end_headers()
    
    def _send_json_response(self, data, status_code=200):
        """Send a JSON response"""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Handle API routes
        if path == "/":
            # API root endpoint
            self._send_json_response({
                "name": "BCBS_Values API",
                "version": "1.0.0",
                "description": "Real estate valuation API for Benton County, WA"
            })
            
        elif path == "/api/valuations":
            # Get property valuations
            query_params = urllib.parse.parse_qs(parsed_path.query)
            limit = int(query_params.get("limit", ["10"])[0])
            
            # Apply simple filtering
            result = SAMPLE_VALUATIONS[:limit]
            self._send_json_response(result)
            
        elif path.startswith("/api/valuations/"):
            # Get valuation for specific property
            property_id = path.split("/")[-1]
            
            # Find property by ID
            property_data = next(
                (p for p in SAMPLE_VALUATIONS if p["property_id"] == property_id), 
                None
            )
            
            if property_data:
                self._send_json_response(property_data)
            else:
                self._send_json_response(
                    {"error": f"Property {property_id} not found"}, 
                    404
                )
                
        elif path == "/api/etl-status":
            # Get ETL status
            self._send_json_response(ETL_STATUS)
            
        elif path == "/api/agent-status":
            # Get agent status
            self._send_json_response(AGENT_STATUS)
            
        else:
            # Handle 404 for unknown routes
            self._send_json_response(
                {"error": "Not found"}, 
                404
            )
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def main():
    """Run the API server"""
    with socketserver.TCPServer(("0.0.0.0", PORT), BCBSValuesAPIHandler) as httpd:
        print(f"Starting BCBS Values API server on port {PORT}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user")
        finally:
            httpd.server_close()

if __name__ == "__main__":
    main()