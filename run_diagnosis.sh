#!/bin/bash
# This script runs diagnostics and then launches a simple diagnostic server

# Set environment variables
export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/postgres}"
export SESSION_SECRET="${SESSION_SECRET:-bcbs_values_session_secret_key_2025}"
export API_KEY="${API_KEY:-bcbs_values_api_key_2025}"
export BCBS_VALUES_API_KEY="${BCBS_VALUES_API_KEY:-$API_KEY}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export ENABLE_CACHING="${ENABLE_CACHING:-true}"

# Print diagnostics header
echo -e "\033[1;36m============================================\033[0m"
echo -e "\033[1;36m=== BCBS VALUES ENVIRONMENT DIAGNOSTICS ===\033[0m"
echo -e "\033[1;36m============================================\033[0m"
echo ""

# Find Python executable - looking specifically for Python 3.11
PYTHON_PATH=""
# First try the specific Python 3.11 we found
PYTHON311="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
if [ -x "$PYTHON311" ]; then
    echo -e "\033[0;32m✓\033[0m Found Python 3.11: $PYTHON311"
    echo -e "  Version: $($PYTHON311 --version 2>&1)"
    PYTHON_PATH="$PYTHON311"
else
    # Fall back to any available Python
    for python_exe in /mnt/nixmodules/nix/store/*/bin/python3.11 /mnt/nixmodules/nix/store/*/bin/python3 /mnt/nixmodules/nix/store/*/bin/python; do
        if [ -x "$python_exe" ]; then
            echo -e "\033[0;32m✓\033[0m Found Python: $python_exe"
            echo -e "  Version: $($python_exe --version 2>&1)"
            PYTHON_PATH="$python_exe"
            break
        fi
    done
fi

if [ -z "$PYTHON_PATH" ]; then
    echo -e "\033[0;31m✗\033[0m No Python executable found."
    echo "Creating a static diagnostic HTML page instead..."
    
    # Create a static HTML page as fallback
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
            <p class="lead">Static Diagnostic Page - Python Not Found</p>
        </div>
        <div class="alert alert-danger">
            <h4 class="alert-heading">Environment Error</h4>
            <p>Python executable could not be found in this environment.</p>
            <p>This is a static fallback page that was generated because the diagnostic server could not be started.</p>
        </div>
    </div>
</body>
</html>
EOL

    echo "Starting a basic file server using busybox..."
    if command -v busybox &> /dev/null; then
        exec busybox httpd -f -p 5000
    else
        echo -e "\033[0;31m✗\033[0m Cannot start any HTTP server. Both Python and busybox are missing."
        exit 1
    fi
else
    # We have Python, so try running our diagnostic server
    echo -e "\033[0;32m✓\033[0m Starting diagnostic server using Python..."
    
    # Make sure the server script is executable
    chmod +x quick_diagnostic_server.py
    
    # Run the diagnostic server
    exec $PYTHON_PATH quick_diagnostic_server.py
fi