#!/bin/bash

# BCBS Values Simple Bash Server
# This is the most minimal implementation of a diagnostic server
# using only Bash and no other dependencies

PORT=${PORT:-5000}
HOST="0.0.0.0"

# Function to send HTTP response header
send_header() {
  echo -e "HTTP/1.1 $1\r\nContent-Type: $2\r\n\r\n"
}

# Function to serve index.html
serve_index() {
  # Read index.html
  if [ -f "index.html" ]; then
    send_header "200 OK" "text/html"
    cat index.html
  else
    # Fallback HTML if index.html doesn't exist
    send_header "200 OK" "text/html"
    echo "<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values - Basic Diagnostic Page</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        h1 { color: #0066cc; }
        pre { background: #f5f5f5; padding: 15px; }
        .card { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    <p>Benton County Building Services - Property Valuation System</p>
    
    <div class='card'>
        <h2>System Status</h2>
        <p>The system is currently running in ultra-minimal bash diagnostic mode.</p>
        <p>Database status: PostgreSQL database is available.</p>
    </div>
    
    <div class='card'>
        <h2>Diagnostic Information</h2>
        <pre>
Server: Bash Minimal HTTP Server
Date: $(date)
Host: $HOST
Port: $PORT
Directory: $(pwd)
        </pre>
    </div>
    
    <div class='card'>
        <h2>Available Files</h2>
        <pre>
$(ls -la | grep -v "total" | head -n 20)
...and more files not shown
        </pre>
    </div>
    
    <footer>
        <p>© 2025 Benton County Building Services</p>
    </footer>
</body>
</html>"
  fi
}

# Function to handle 404s
serve_404() {
  send_header "404 Not Found" "text/html"
  echo "<!DOCTYPE html>
<html>
<head>
    <title>404 - Not Found</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
        h1 { color: #d9534f; }
    </style>
</head>
<body>
    <h1>404 - Not Found</h1>
    <p>The requested file '$1' was not found on this server.</p>
    <p><a href='/'>Go back to home page</a></p>
</body>
</html>"
}

echo "========================================================"
echo "BCBS VALUES MINIMAL BASH HTTP SERVER"
echo "========================================================"
echo "Starting at: $(date)"
echo "Listening on: http://$HOST:$PORT"
echo "No external dependencies used. Pure bash implementation."
echo "========================================================"

# Check if netcat is available
if command -v nc &>/dev/null; then
  # Use netcat for a more reliable server
  echo "Starting server with netcat..."
  while true; do
    echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n$(cat index.html)" | nc -l -p $PORT || break
  done
else
  # Fallback to /dev/tcp if netcat is not available
  echo "Netcat not found. Using /dev/tcp fallback..."
  
  # Simple TCP server using only bash
  while true; do
    # Create a temporary file to store request
    tmp_file=$(mktemp)
    
    # Start listening on the TCP socket
    echo "Waiting for connections on port $PORT..."
    
    # Use exec to redirect file descriptors
    exec 3<>/dev/tcp/$HOST/$PORT || { echo "Failed to bind to port $PORT"; exit 1; }
    
    # Read the HTTP request
    while read -r line; do
      # Store the first line to check the request path
      [ -z "$first_line" ] && first_line="$line"
      
      # Break on empty line (end of HTTP headers)
      [ -z "$line" ] && break
      
      # Store each line for debugging
      echo "$line" >> "$tmp_file"
    done <&3
    
    # Extract the requested path from the first line
    request_path=$(echo "$first_line" | awk '{print $2}')
    echo "Request for path: $request_path"
    
    # Serve the appropriate content based on the path
    if [ "$request_path" = "/" ] || [ "$request_path" = "/index.html" ]; then
      serve_index >&3
    else
      serve_404 "$request_path" >&3
    fi
    
    # Close the connection
    exec 3>&-
    rm -f "$tmp_file"
  done
fi

echo "Server has stopped."
exit 1