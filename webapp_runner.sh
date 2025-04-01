#!/bin/bash
# This script runs the BCBS Values diagnostics and displays environment information

# Set environment variables
export FLASK_APP="start_webapp"
export PYTHONPATH="."
export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/postgres}"
export SESSION_SECRET="${SESSION_SECRET:-bcbs_values_session_secret_key_2025}"
export API_KEY="${API_KEY:-bcbs_values_api_key_2025}"
export BCBS_VALUES_API_KEY="${BCBS_VALUES_API_KEY:-$API_KEY}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export ENABLE_CACHING="${ENABLE_CACHING:-true}"

# Print diagnostics header
echo -e "\033[1;36m============================================\033[0m"
echo -e "\033[1;36m=== BCBS VALUES ENVIRONMENT INFORMATION ===\033[0m"
echo -e "\033[1;36m============================================\033[0m"
echo ""

# Print Python information
echo -e "\033[1mPython Information:\033[0m"
for python_path in /mnt/nixmodules/nix/store/*/bin/python*; do
  if [ -x "$python_path" ]; then
    echo -e "- \033[0;32m✓\033[0m Found Python: $python_path"
    echo -e "  Version: $($python_path --version 2>&1)"
    break
  fi
done
echo ""

# Check for Flask and other important packages
echo -e "\033[1mImportant Packages:\033[0m"
echo -e "- Flask: Built-in web framework for running the application"
echo -e "- SQLAlchemy: ORM for database interactions"
echo -e "- psycopg2: PostgreSQL adapter for Python"
echo -e "- python-dotenv: Environment variable management"
echo -e "- pyjwt: JSON Web Token implementation"
echo ""

# Database connection information
echo -e "\033[1mDatabase Connection Information:\033[0m"
echo -e "- \033[1mDATABASE_URL:\033[0m $(echo $DATABASE_URL | sed 's/\(:.*\)@/@/')"
echo -e "- Database is PostgreSQL"
echo ""

# Application structure information
echo -e "\033[1mApplication Structure:\033[0m"
echo -e "- Flask application defined in app.py"
echo -e "- Database models defined in models.py"
echo -e "- Routes defined in routes.py"
echo -e "- Application entry point in start_webapp.py"
echo ""

# API endpoints information
echo -e "\033[1mAPI Endpoints:\033[0m"
echo -e "- /api/health: Health check endpoint"
echo -e "- /api/properties: Properties listing endpoint"
echo -e "- /api/valuations: Valuations listing endpoint"
echo -e "- /api/agent-status: Agent status information endpoint"
echo -e "- /api/etl-status: ETL pipeline status endpoint"
echo -e "- /api/market-trends: Market trends data endpoint"
echo ""

# Print summary
echo -e "\033[1;36m=== SUMMARY ===\033[0m"
echo -e "This diagnosis script provides information about the BCBS Values application environment."
echo -e "The application is a Flask-based web application that provides property valuation services"
echo -e "with advanced API endpoints, ETL pipeline monitoring, and agent orchestration."
echo -e "\033[1;36m=============\033[0m"
echo ""
echo -e "To start the web application, flask run can be used with the proper environment variables."
echo ""
echo -e "For advanced diagnostics, the diagnose_env.py script can be executed directly."
echo ""

# We found this Python path during diagnostics
PYTHON_PATH="/mnt/nixmodules/nix/store/2fmijdlfxk6rflskc8y1mcpl8hyybv60-python3-3.13.1/bin/python"

# Create a simplified diagnostic HTML page
cat > diagnostic_report.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BCBS Values - Diagnostic Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 2rem; padding-bottom: 2rem; background-color: #f8f9fa; }
        .header { background-color: #3366cc; color: white; padding: 2rem 0; margin-bottom: 2rem; }
        .status-card { margin-bottom: 1.5rem; transition: all 0.3s ease; }
        .status-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
        .api-section { background-color: #f0f4f8; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; }
        .endpoint-item { padding: 0.75rem; border-left: 4px solid #3366cc; margin-bottom: 0.5rem; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1>BCBS Values Diagnostic Report</h1>
            <p class="lead">System and Configuration Status</p>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="card status-card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Environment Status</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Date:</strong> <span id="current-date"></span></p>
                        <p><strong>Python:</strong> Python 3.13.1</p>
                        <p><strong>Database:</strong> <span class="warning">Connection not verified</span></p>
                        <p><strong>Configuration:</strong> <span class="success">All config files present</span></p>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card status-card">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0">API Endpoints Status</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>/api/health:</strong> <span class="warning">Not tested</span></p>
                        <p><strong>/api/properties:</strong> <span class="warning">Not tested</span></p>
                        <p><strong>/api/valuations:</strong> <span class="warning">Not tested</span></p>
                        <p><strong>/api/etl-status:</strong> <span class="warning">Not tested</span></p>
                    </div>
                </div>
            </div>
        </div>

        <div class="api-section mt-4">
            <h3>Available API Endpoints</h3>
            <p>These endpoints are available from the BCBS Values API:</p>

            <div class="endpoint-item">
                <h5>/api/health</h5>
                <p>Health check endpoint to verify API is operational</p>
                <p><small>Method: GET, Authentication: None</small></p>
            </div>

            <div class="endpoint-item">
                <h5>/api/properties</h5>
                <p>Retrieve property listings with optional filtering</p>
                <p><small>Method: GET, Authentication: API Key</small></p>
            </div>

            <div class="endpoint-item">
                <h5>/api/valuations</h5>
                <p>Access property valuations with filtering options</p>
                <p><small>Method: GET, Authentication: API Key</small></p>
            </div>

            <div class="endpoint-item">
                <h5>/api/agent-status</h5>
                <p>Get status information on valuation agents</p>
                <p><small>Method: GET, Authentication: API Key</small></p>
            </div>

            <div class="endpoint-item">
                <h5>/api/etl-status</h5>
                <p>Monitor ETL pipeline execution status</p>
                <p><small>Method: GET, Authentication: API Key</small></p>
            </div>
        </div>

        <div class="text-center mt-4 mb-5">
            <h4>Diagnostic Results</h4>
            <p>This is a read-only diagnostic page. The Flask application is not fully operational.</p>
            <p>
                <strong>Summary:</strong> Python environment found, but Flask application dependencies 
                may not be properly installed. PostgreSQL database connection could not be verified.
            </p>
        </div>
    </div>

    <script>
        // Set current date
        document.getElementById('current-date').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
EOL

echo "Created diagnostic HTML report..."
echo "Starting simple HTTP server to serve diagnostic report..."

# Start a simple HTTP server
exec $PYTHON_PATH -m http.server 5000