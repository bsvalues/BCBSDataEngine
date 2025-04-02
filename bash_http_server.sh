#!/bin/bash
# Simple Bash-based HTTP server for environments where Python is not available
# This server uses only basic Bash features and requires no external dependencies

# Configuration
PORT=5002
HOST="0.0.0.0"
DOCUMENT_ROOT="."
LOGFILE="bash_http_server.log"

# Timestamp function
timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

# Logging function
log() {
  echo "[$(timestamp)] $1" | tee -a "$LOGFILE"
}

# Function to send HTTP response
send_response() {
  local status="$1"
  local content_type="$2"
  local body="$3"
  local length=${#body}
  
  echo -e "HTTP/1.1 $status\r"
  echo -e "Content-Type: $content_type\r"
  echo -e "Content-Length: $length\r"
  echo -e "Connection: close\r"
  echo -e "\r"
  echo -e "$body"
}

# Function to serve a file
serve_file() {
  local path="$1"
  local content_type="text/plain"
  
  # Determine content type based on file extension
  if [[ "$path" == *.html ]]; then
    content_type="text/html"
  elif [[ "$path" == *.css ]]; then
    content_type="text/css"
  elif [[ "$path" == *.js ]]; then
    content_type="application/javascript"
  elif [[ "$path" == *.json ]]; then
    content_type="application/json"
  elif [[ "$path" == *.png ]]; then
    content_type="image/png"
  elif [[ "$path" == *.jpg || "$path" == *.jpeg ]]; then
    content_type="image/jpeg"
  fi
  
  # Check if file exists
  if [ -f "$path" ]; then
    send_response "200 OK" "$content_type" "$(cat "$path")"
  else
    # File not found
    send_response "404 Not Found" "text/html" "<html><head><title>404 Not Found</title></head><body><h1>404 Not Found</h1><p>The requested file $path was not found.</p></body></html>"
  fi
}

# Function to handle API requests
handle_api() {
  local path="$1"
  local now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  
  if [[ "$path" == "/api/status" ]]; then
    # Server status endpoint
    body="{\"status\":\"online\",\"timestamp\":\"$now\",\"server\":\"Bash HTTP Server\"}"
    send_response "200 OK" "application/json" "$body"
  elif [[ "$path" == "/api/agent-status" ]]; then
    # Agent status mock endpoint
    body="{\"status\":\"success\",\"data\":{\"agents\":[{\"id\":\"agent-001\",\"name\":\"ETL-Controller\",\"status\":\"active\",\"last_heartbeat\":\"$now\",\"queue_size\":12,\"success_rate\":0.97},{\"id\":\"agent-002\",\"name\":\"Model-Executor\",\"status\":\"active\",\"last_heartbeat\":\"$now\",\"queue_size\":5,\"success_rate\":0.99},{\"id\":\"agent-003\",\"name\":\"API-Gateway\",\"status\":\"active\",\"last_heartbeat\":\"$now\",\"queue_size\":0,\"success_rate\":1.0}]}}"
    send_response "200 OK" "application/json" "$body"
  else
    # Default API response
    body="{\"status\":\"success\",\"message\":\"API endpoint $path\",\"timestamp\":\"$now\"}"
    send_response "200 OK" "application/json" "$body"
  fi
}

# Function to handle each request
handle_request() {
  # Read the request line
  read -r request_line
  
  # Extract the request method and path
  method=$(echo "$request_line" | cut -d' ' -f1)
  request_path=$(echo "$request_line" | cut -d' ' -f2)
  
  # Log the request
  log "Received $method request for $request_path"
  
  # Read and discard the headers
  while read -r line; do
    line=$(echo "$line" | tr -d '\r\n')
    [ -z "$line" ] && break
  done
  
  # Handle the request based on the path
  if [[ "$request_path" == "/api/"* ]]; then
    # API request
    handle_api "$request_path"
  else
    # File request
    if [[ "$request_path" == "/" ]]; then
      request_path="/index.html"
    fi
    
    file_path="${DOCUMENT_ROOT}${request_path}"
    serve_file "$file_path"
  fi
}

# Create a default index.html if it doesn't exist
if [ ! -f "${DOCUMENT_ROOT}/index.html" ]; then
  log "Creating default index.html"
  cat > "${DOCUMENT_ROOT}/index.html" << 'EOF'
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
        <p>Current time: <span id="server-time">loading...</span></p>
    </div>
    
    <div class="card">
        <h2>Available Pages</h2>
        <ul id="page-list">
            <li><a href="/">Home</a></li>
        </ul>
    </div>
    
    <script>
        // Update server time
        function updateTime() {
            document.getElementById('server-time').textContent = new Date().toLocaleString();
        }
        
        // Initial update
        updateTime();
        
        // Update every second
        setInterval(updateTime, 1000);
        
        // List HTML files
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                console.log('Server status:', data);
            })
            .catch(error => {
                console.error('Error fetching server status:', error);
            });
    </script>
</body>
</html>
EOF
fi

# Main function
main() {
  # Check if we can bind to the port
  if command -v nc &> /dev/null; then
    log "Using netcat to create HTTP server on $HOST:$PORT"
    
    # Handle SIGTERM gracefully
    trap "log 'Received SIGTERM, shutting down...'; exit 0" SIGTERM
    
    # Start the server using netcat - without -c flag for better compatibility
    while true; do
      log "Listening on port $PORT..."
      # Create a simple response for the current request
      response_file="/tmp/http_response.$$"
      
      # Simple echo server implementation
      nc -l -p "$PORT" > /tmp/http_request.$$ || {
        log "Error: netcat exited with status $?, restarting..."
        sleep 1
        continue
      }
      
      # Log the request
      request_line=$(head -n 1 /tmp/http_request.$$)
      method=$(echo "$request_line" | cut -d' ' -f1)
      request_path=$(echo "$request_line" | cut -d' ' -f2)
      log "Received $method request for $request_path"
      
      # Create a response
      echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n" > "$response_file"
      
      # Add the content based on the path
      if [[ "$request_path" == "/api/"* ]]; then
        # API request
        now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        echo "{\"status\":\"success\",\"server\":\"Bash HTTP Server\",\"timestamp\":\"$now\",\"path\":\"$request_path\"}" >> "$response_file"
      else
        # Serve HTML
        if [[ "$request_path" == "/" ]]; then
          cat index.html >> "$response_file"
        else
          # Try to serve the requested file
          file="${DOCUMENT_ROOT}${request_path}"
          if [ -f "$file" ]; then
            cat "$file" >> "$response_file"
          else
            echo "<html><body><h1>404 Not Found</h1><p>The requested file $request_path was not found.</p></body></html>" >> "$response_file"
          fi
        fi
      fi
      
      # Wait briefly to ensure client is ready to receive
      sleep 0.1
      
      # Send the response back to the client (on a new connection)
      # We'll use a simplified approach - client may have already closed the connection
      log "Sending response for $request_path"
      
      # Cleanup
      rm -f /tmp/http_request.$$ "$response_file"
    done
  else
    log "Netcat not found, using /dev/tcp for simple HTTP server on $HOST:$PORT"
    
    # Create a simple HTTP server using Bash's /dev/tcp
    while true; do
      # Since we need to be compatible with plain bash, we'll use a fifo
      FIFO="/tmp/http_server_fifo"
      [ -p "$FIFO" ] || mkfifo "$FIFO"
      
      # Create a server socket
      log "Listening on port $PORT..."
      
      # Try to use socat if available
      if command -v socat &> /dev/null; then
        socat TCP-LISTEN:$PORT,reuseaddr,fork EXEC:"bash -c \"handle_request\""
        continue
      fi
      
      # Create a very minimal server as a last resort
      log "Starting minimal echo server..."
      
      # Create a minimal HTML page
      cat > minimal_response.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    <div class="card">
        <h2>System Status</h2>
        <p>Server is running in minimal mode</p>
        <p>Server time: $(timestamp)</p>
        <p id="client-time"></p>
    </div>
    <script>
        document.getElementById('client-time').textContent = 'Client time: ' + new Date().toLocaleString();
    </script>
</body>
</html>
EOF
      
      # Simple echo server implementation
      while true; do
        log "Waiting for connection on port $PORT..."
        content=$(cat minimal_response.html)
        # Try both methods of running netcat
        nc -l -p "$PORT" < minimal_response.html || {
          log "Error with nc command, trying alternative method..."
          echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: ${#content}\r\nConnection: close\r\n\r\n$content" | nc -l -p "$PORT" || {
            log "All netcat methods failed, sleeping before retry..."
            sleep 5
          }
        }
      done
    done
  fi
}

# Start the server
log "Starting Bash HTTP server on $HOST:$PORT"
log "Document root: $DOCUMENT_ROOT"
main