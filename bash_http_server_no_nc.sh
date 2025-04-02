#!/bin/bash

# Server configuration
PORT=5002
HOST="0.0.0.0"

echo "Starting simple bash HTTP server on port $PORT using /dev/tcp..."
echo "This will serve the BCBS dashboard files."
echo "Access the dashboard at: https://${REPL_SLUG}.${REPL_OWNER}.repl.co/"
echo "Press Ctrl+C to stop the server."

# Default document to serve for root requests
DEFAULT_DOC="dashboard_static.html"

# Function to parse HTTP headers and path
parse_http_request() {
  local request=""
  local line=""
  local path=""
  
  # Read the first chunk of data
  while IFS= read -r line; do
    request+="$line"$'\n'
    if [[ "$line" == $'\r' || "$line" == "" ]]; then
      break
    fi
  done
  
  # Extract the path from the request
  path=$(echo "$request" | head -1 | awk '{print $2}')
  
  echo "$path"
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

# Create a simple server using /dev/tcp
echo "Using /dev/tcp for serving HTTP requests..."

# Check if some basic requirements are available
if ! command -v bash &> /dev/null || ! command -v cat &> /dev/null; then
  echo "Error: Basic requirements (bash, cat) are not available."
  exit 1
fi

# Check if dashboard file exists
if [ ! -f "$DEFAULT_DOC" ]; then
  echo "Warning: Default document '$DEFAULT_DOC' not found. The server will return 404 for root requests."
fi

# Create a simplified Python HTTP server if Python is available
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
  PYTHON_CMD=""
  if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
  else
    PYTHON_CMD="python"
  fi
  
  echo "Python found. Using Python's built-in HTTP server..."
  exec $PYTHON_CMD -m http.server $PORT 2>/dev/null || \
  exec $PYTHON_CMD -m SimpleHTTPServer $PORT
  exit 0
fi

# Fallback message if nothing works
echo "Error: Unable to start an HTTP server. Please install Python or nc (netcat)."
echo "You can try: ./run_dashboard.sh instead."
exit 1