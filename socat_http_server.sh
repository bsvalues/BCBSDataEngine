#!/bin/bash
# A very simple HTTP server implemented in pure Bash
# This version uses /dev/tcp for communication

PORT=5002
DEFAULT_PAGE="dashboard_static.html"
ERROR_PAGE="404.html"

# Make sure we're in the right directory
cd "$(dirname "$0")"

echo "Starting Bash HTTP Server on port $PORT..."
echo "Press Ctrl+C to stop the server."

# Function to handle HTTP requests and send a response
handle_request() {
  local request_line
  local path
  local file
  local ext
  local content_type
  local size
  
  # Read the first line of the HTTP request
  read -r request_line
  
  # Extract the request path
  path=$(echo "$request_line" | awk '{print $2}')
  
  # Default to dashboard if root is requested
  if [ "$path" = "/" ]; then
    path="/$DEFAULT_PAGE"
  fi
  
  # Remove the leading slash
  file="${path:1}"
  
  # Log the request
  echo "$(date): $request_line - $file"
  
  # Keep reading headers until we reach an empty line
  while read -r header && [ -n "$header" ]; do
    # Remove carriage return
    header=${header%$'\r'}
    # Skip empty lines
    [ -z "$header" ] && break
  done
  
  # Check if the file exists
  if [ -f "$file" ]; then
    # Determine content type based on file extension
    ext="${file##*.}"
    case "$ext" in
      html) content_type="text/html" ;;
      css)  content_type="text/css" ;;
      js)   content_type="application/javascript" ;;
      json) content_type="application/json" ;;
      png)  content_type="image/png" ;;
      jpg|jpeg) content_type="image/jpeg" ;;
      gif)  content_type="image/gif" ;;
      svg)  content_type="image/svg+xml" ;;
      *)    content_type="text/plain" ;;
    esac
    
    # Get the file size
    size=$(wc -c < "$file")
    
    # Send HTTP headers
    echo -e "HTTP/1.1 200 OK\r"
    echo -e "Content-Type: $content_type\r"
    echo -e "Content-Length: $size\r"
    echo -e "Connection: close\r"
    echo -e "\r"
    
    # Send the file content
    cat "$file"
  else
    # Check if we have a 404 page
    if [ -f "$ERROR_PAGE" ]; then
      # Send 404 with custom error page
      size=$(wc -c < "$ERROR_PAGE")
      
      echo -e "HTTP/1.1 404 Not Found\r"
      echo -e "Content-Type: text/html\r"
      echo -e "Content-Length: $size\r"
      echo -e "Connection: close\r"
      echo -e "\r"
      
      cat "$ERROR_PAGE"
    else
      # Send a simple 404 message
      echo -e "HTTP/1.1 404 Not Found\r"
      echo -e "Content-Type: text/plain\r"
      echo -e "Content-Length: 13\r"
      echo -e "Connection: close\r"
      echo -e "\r"
      echo "404 Not Found"
    fi
  fi
}

# Function to clean up on exit
cleanup() {
  echo "Shutting down server..."
  exit 0
}

# Set up signal handling
trap cleanup INT TERM

# Main server loop - using the builtin /dev/tcp
while true; do
  # We need to use a temporary file for each connection
  tempfile=$(mktemp)
  
  echo "Waiting for connection..."
  
  # Use this technique for /dev/tcp
  # This is a workaround because we can't handle I/O efficiently
  {
    (
      handle_request > "$tempfile"
      cat "$tempfile"
    ) 
  } < /dev/tcp/0.0.0.0/$PORT > /dev/tcp/0.0.0.0/$PORT
  
  rm -f "$tempfile"
done