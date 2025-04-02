#!/bin/bash

# Server configuration
PORT=5002
HOST="0.0.0.0"

echo "Starting simple bash HTTP server on port $PORT..."
echo "This will serve the BCBS dashboard files."
echo "Access the dashboard at: https://${REPL_SLUG}.${REPL_OWNER}.repl.co/"
echo "Press Ctrl+C to stop the server."

# Default document to serve for root requests
DEFAULT_DOC="dashboard_static.html"

# Function to parse HTTP headers
read_http_headers() {
  local line=""
  local path=""
  
  # Read the first line of the HTTP request
  read -r line
  
  # Extract the requested path
  path=$(echo "$line" | awk '{print $2}')
  
  # Read all headers until an empty line
  while read -r line && [ "$line" != $'\r' ] && [ "$line" != "" ]; do
    :
  done
  
  # Return the path
  echo "$path"
}

# Function to generate HTTP response
send_response() {
  local statuscode="$1"
  local contenttype="$2"
  local content="$3"
  
  echo -e "HTTP/1.1 $statuscode\r\nContent-Type: $contenttype\r\nContent-Length: ${#content}\r\nConnection: close\r\n\r\n$content"
}

# Function to determine content type
get_content_type() {
  local file="$1"
  local ext="${file##*.}"
  
  case "$ext" in
    html) echo "text/html" ;;
    css) echo "text/css" ;;
    js) echo "application/javascript" ;;
    json) echo "application/json" ;;
    png) echo "image/png" ;;
    jpg|jpeg) echo "image/jpeg" ;;
    gif) echo "image/gif" ;;
    svg) echo "image/svg+xml" ;;
    *) echo "text/plain" ;;
  esac
}

# Create a TCP server socket
if command -v nc &> /dev/null; then
  echo "Using netcat for serving HTTP requests..."
  
  while true; do
    # Use netcat to listen on the specified port
    echo -e "Listening on $HOST:$PORT...\n"
    
    (
      path=$(read_http_headers)
      
      # Normalize the path
      path="${path//\.\.\//}" # Remove directory traversal attempts
      path="${path//\/\//\/}" # Normalize double slashes
      path="${path#/}"       # Remove leading slash
      
      # Map specific routes
      case "$path" in
        ""|"index.html") file="$DEFAULT_DOC" ;;
        "dashboard") file="dashboard_static.html" ;;
        "bcbs-dashboard") file="bcbs_dashboard.html" ;;
        "interactive-dashboard") file="templates/reactive_dashboard.html" ;;
        "demo") file="dashboard_demo.html" ;;
        *) file="$path" ;;
      esac
      
      # Check if file exists
      if [ ! -f "$file" ]; then
        if [ -f "404.html" ]; then
          send_response "404 Not Found" "text/html" "$(cat 404.html)"
        else
          send_response "404 Not Found" "text/plain" "404 Not Found: $file"
        fi
      else
        content_type=$(get_content_type "$file")
        send_response "200 OK" "$content_type" "$(cat "$file")"
      fi
    ) | nc -l -p $PORT -q 1
  done
else
  echo "Error: netcat (nc) is not available. Please install it to run this server."
  exit 1
fi