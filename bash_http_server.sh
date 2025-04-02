#!/bin/bash
# Pure Bash HTTP server for BCBS Values Platform
# This script implements a basic HTTP server using only Bash and netcat
# It doesn't require any external dependencies other than netcat

# Configuration
PORT=5002
HOST=0.0.0.0
RESPONSE_COUNT=0
INDEX_HTML="index.html"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to create the index.html file if it doesn't exist
create_index_html() {
    if [ ! -f "$INDEX_HTML" ]; then
        log "Creating minimal $INDEX_HTML file"
        cat > "$INDEX_HTML" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
        .success { background: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    
    <div class="card">
        <h2>System Status</h2>
        <p><span class="status success">Server is running</span></p>
        <p>Server started at: <span id="server-time">loading...</span></p>
    </div>
    
    <div class="card">
        <h2>Server Information</h2>
        <p>This is a Bash-based HTTP server for the BCBS Values Platform.</p>
        <p>It provides minimal functionality for serving static files and API responses.</p>
    </div>
    
    <script>
        // Set current time
        document.getElementById('server-time').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
EOF
    fi
}

# Function to create a 404 page
get_404_page() {
    local path="$1"
    
    echo "<!DOCTYPE html>
<html>
<head>
    <title>404 Not Found</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #721c24; }
        .card { border: 1px solid #f5c6cb; border-radius: 8px; padding: 20px; 
               background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>404 Not Found</h1>
    <div class=\"card\">
        <p>The requested URL $path was not found on this server.</p>
        <p><a href=\"/\">Return to Homepage</a></p>
    </div>
</body>
</html>"
}

# Function to handle API requests
handle_api_request() {
    local path="$1"
    local endpoint=""
    
    # Extract the endpoint
    if [[ "$path" =~ ^/api/([^/]+) ]]; then
        endpoint="${BASH_REMATCH[1]}"
    fi
    
    # Generate the response based on the endpoint
    if [ "$endpoint" == "status" ]; then
        echo "{
  \"status\": \"online\",
  \"uptime\": \"$(uptime)\",
  \"requests_handled\": $RESPONSE_COUNT,
  \"timestamp\": \"$(date -Iseconds)\"
}"
    elif [ "$endpoint" == "agent-status" ]; then
        echo "{
  \"status\": \"success\",
  \"message\": \"Agent status retrieved successfully\",
  \"data\": {
    \"agents\": [
      {
        \"id\": \"agent-001\",
        \"name\": \"ETL-Controller\",
        \"status\": \"active\",
        \"last_heartbeat\": \"$(date -Iseconds)\",
        \"queue_size\": 12,
        \"success_rate\": 0.97
      },
      {
        \"id\": \"agent-002\",
        \"name\": \"Model-Executor\",
        \"status\": \"active\",
        \"last_heartbeat\": \"$(date -Iseconds)\",
        \"queue_size\": 5,
        \"success_rate\": 0.99
      },
      {
        \"id\": \"agent-003\",
        \"name\": \"API-Gateway\",
        \"status\": \"active\",
        \"last_heartbeat\": \"$(date -Iseconds)\",
        \"queue_size\": 0,
        \"success_rate\": 1.0
      }
    ],
    \"timestamp\": \"$(date -Iseconds)\"
  }
}"
    else
        echo "{
  \"status\": \"success\",
  \"message\": \"API endpoint: $endpoint\",
  \"path\": \"$path\",
  \"timestamp\": \"$(date -Iseconds)\"
}"
    fi
}

# Function to handle HTTP requests
handle_request() {
    local request="$1"
    local response=""
    local status_code="200 OK"
    local content_type="text/html"
    local body=""
    
    # Extract the path from the request line
    local path=""
    if [[ "$request" =~ ^GET[[:space:]]+([^[:space:]]+) ]]; then
        path="${BASH_REMATCH[1]}"
    else
        path="/"
    fi
    
    # Increment response count
    RESPONSE_COUNT=$((RESPONSE_COUNT + 1))
    
    # Log the request
    log "Request #$RESPONSE_COUNT: GET $path"
    
    # Handle different paths
    if [ "$path" == "/" ]; then
        # Serve the index page
        if [ -f "$INDEX_HTML" ]; then
            body=$(cat "$INDEX_HTML")
        else
            body="<html><body><h1>BCBS Values Platform</h1><p>Welcome to the server!</p></body></html>"
        fi
    elif [[ "$path" =~ ^/api/ ]]; then
        # Handle API requests
        content_type="application/json"
        body=$(handle_api_request "$path")
    else
        # Check if the file exists
        local file="${path:1}" # Remove leading slash
        if [ -f "$file" ]; then
            body=$(cat "$file")
            
            # Set content type based on file extension
            if [[ "$file" =~ \.html$ ]]; then
                content_type="text/html"
            elif [[ "$file" =~ \.css$ ]]; then
                content_type="text/css"
            elif [[ "$file" =~ \.js$ ]]; then
                content_type="application/javascript"
            elif [[ "$file" =~ \.json$ ]]; then
                content_type="application/json"
            elif [[ "$file" =~ \.(jpg|jpeg)$ ]]; then
                content_type="image/jpeg"
            elif [[ "$file" =~ \.png$ ]]; then
                content_type="image/png"
            elif [[ "$file" =~ \.svg$ ]]; then
                content_type="image/svg+xml"
            fi
        else
            # File not found
            status_code="404 Not Found"
            body=$(get_404_page "$path")
        fi
    fi
    
    # Construct the HTTP response
    local response_length=${#body}
    response="HTTP/1.1 $status_code
Server: BCBS-Bash-Server
Content-Type: $content_type
Content-Length: $response_length
Connection: close

$body"
    
    echo "$response"
}

# Main function to run the server
run_server() {
    log "Starting BCBS Values Platform HTTP Server"
    log "========================================"
    log "Server running at http://$HOST:$PORT/"
    
    # Create index.html if it doesn't exist
    create_index_html
    
    # List available HTML files
    html_files=$(find . -maxdepth 1 -name "*.html" | sort)
    if [ -n "$html_files" ]; then
        log "Available HTML files:"
        for file in $html_files; do
            log "  - $(basename "$file")"
        done
    else
        log "No HTML files found"
    fi
    
    log "Ready to accept connections. Press Ctrl+C to stop."
    
    # Main server loop
    while true; do
        # The command below creates a TCP server on the specified port using netcat,
        # reads the incoming HTTP request, passes it to handle_request function,
        # and sends the response back to the client
        echo "$(handle_request "$(nc -l -p $PORT)")" | nc -l -p $PORT
    done
}

# Start the server
run_server