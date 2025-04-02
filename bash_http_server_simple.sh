#!/bin/bash
# A very simple HTTP server implemented in pure Bash
# This is a fallback server that should work in most environments

PORT=5002
DEFAULT_PAGE="dashboard_static.html"
ERROR_PAGE="404.html"

# Make sure we're in the right directory
cd "$(dirname "$0")"

echo "Starting Bash HTTP Server on port $PORT..."
echo "Press Ctrl+C to stop the server."

# Function to handle HTTP requests
handle_request() {
  read -r request
  
  # Extract the request path
  path=$(echo "$request" | awk '{print $2}')
  
  # Default to the dashboard file
  if [ "$path" = "/" ]; then
    path="/$DEFAULT_PAGE"
  fi
  
  # Remove the leading slash
  file="${path:1}"
  
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

# Create a fifo for communication
FIFO="/tmp/bash_http_server_$$"
rm -f "$FIFO"
mkfifo "$FIFO"

# Function to clean up on exit
cleanup() {
  echo "Shutting down server..."
  rm -f "$FIFO"
  exit 0
}

# Set up signal handling
trap cleanup INT TERM

# Start the server loop
while true; do
  # Listen on the specified port
  nc -l -p $PORT < "$FIFO" | handle_request > "$FIFO"
done