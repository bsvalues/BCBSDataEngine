#!/bin/bash

# BCBS Values Ultra-Ultra-Simple Minimal Diagnostic Server
# This script creates the simplest possible emergency response

echo "==================================================="
echo "BCBS VALUES MINIMAL EMERGENCY SERVER"
echo "==================================================="
echo "Starting at: $(date)"
echo ""

# Set port (always use 5000 for simplicity)
PORT=5000

# Create a very basic message
echo "Content-Type: text/html"
echo ""
echo "<!DOCTYPE html>"
echo "<html>"
echo "<head><title>BCBS Values</title></head>"
echo "<body>"
echo "<h1>BCBS Values - Minimal Emergency Page</h1>"
echo "<p>This is the minimal emergency fallback page.</p>"
echo "<p>Date: $(date)</p>"
echo "<p>Working Directory: $(pwd)</p>"
echo "</body>"
echo "</html>"

# Try to serve it with whatever is available
echo "Trying to start emergency server on port $PORT..."

# Attempt several methods, from simplest to more complex
echo "Attempting various methods to start a minimal web server..."

# Create a one-line HTML file to be served
echo "<!DOCTYPE html><html><head><title>BCBS Values</title></head><body><h1>BCBS Values - Minimal Emergency Page</h1><p>This is the minimal emergency fallback page.</p><p>Date: $(date)</p><p>Working Directory: $(pwd)</p></body></html>" > emergency.html

# Try netcat in various modes
if command -v nc >/dev/null 2>&1; then
  echo "* Using netcat..."
  while true; do
    echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n$(cat emergency.html)" | nc -l -p $PORT || break
    sleep 1
  done
  exit $?
fi

# Try Python if available
if command -v python3 >/dev/null 2>&1; then
  echo "* Using Python..."
  cd $(dirname $(readlink -f $0))
  python3 -m http.server $PORT
  exit $?
fi

if command -v python >/dev/null 2>&1; then
  echo "* Using Python 2..."
  cd $(dirname $(readlink -f $0))
  python -m SimpleHTTPServer $PORT
  exit $?
fi

# Final fallback: just print the HTML to console
echo "* No server methods available. Printing HTML that would have been served:"
cat emergency.html

# Keep the script running
echo "Unable to start any type of web server."
echo "Entering sleep mode to keep the workflow alive..."
while true; do
  sleep 600
done