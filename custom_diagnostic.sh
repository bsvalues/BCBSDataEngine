#!/bin/bash

# BCBS Values Custom Diagnostic Script
# This script performs more comprehensive diagnostics and tries to start various servers

echo "========================================================"
echo "BCBS VALUES CUSTOM DIAGNOSTIC SCRIPT"
echo "========================================================"
echo "Current directory: $(pwd)"
echo "Current time: $(date)"

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Set default port
PORT=${PORT:-5000}
export PORT

echo "========================================================"
echo "ENVIRONMENT INFORMATION"
echo "========================================================"

# Check Python
if command_exists python3; then
  echo "Python version: $(python3 --version 2>&1)"
elif command_exists python; then
  echo "Python version: $(python --version 2>&1)"
else
  echo "Python version: No Python found"
fi

# Check Node.js
if command_exists node; then
  echo "Node.js version: $(node --version 2>&1)"
else
  echo "Node.js version: No Node.js found"
fi

# System information
echo "System: $(uname -a)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "Server port: $PORT"

echo ""
echo "========================================================"
echo "TRYING PYTHON SERVERS"
echo "========================================================"

# Try Python-based diagnostic server
echo "Attempting to run simple Python diagnostic server..."
if command_exists python3 && [ -f "simple_diagnostic.py" ]; then
  echo "Running simple_diagnostic.py with Python 3..."
  python3 simple_diagnostic.py &
  # Store the PID to kill it later if needed
  PYTHON_PID=$!
  # Wait a moment for server to start
  sleep 2
  # Check if server is still running
  if kill -0 $PYTHON_PID 2>/dev/null; then
    echo "Python diagnostic server started successfully with PID $PYTHON_PID."
    wait $PYTHON_PID
    exit $?
  else
    echo "Python diagnostic server failed to start or crashed."
  fi
elif command_exists python && [ -f "simple_diagnostic.py" ]; then
  echo "Running simple_diagnostic.py with Python..."
  python simple_diagnostic.py &
  # Store the PID to kill it later if needed
  PYTHON_PID=$!
  # Wait a moment for server to start
  sleep 2
  # Check if server is still running
  if kill -0 $PYTHON_PID 2>/dev/null; then
    echo "Python diagnostic server started successfully with PID $PYTHON_PID."
    wait $PYTHON_PID
    exit $?
  else
    echo "Python diagnostic server failed to start or crashed."
  fi
else
  echo "Cannot run simple_diagnostic.py: Python not found in PATH."
fi

echo ""
echo "========================================================"
echo "TRYING NODE.JS SERVERS"
echo "========================================================"

# Try Node.js-based servers
echo "Attempting to run main Node.js server..."
if command_exists node && [ -f "server.js" ]; then
  echo "Running server.js with Node.js..."
  node server.js &
  # Store the PID to kill it later if needed
  NODE_PID=$!
  # Wait a moment for server to start
  sleep 2
  # Check if server is still running
  if kill -0 $NODE_PID 2>/dev/null; then
    echo "Node.js server started successfully with PID $NODE_PID."
    wait $NODE_PID
    exit $?
  else
    echo "Node.js server failed to start or crashed."
  fi
else
  echo "Cannot run server.js: Node.js not found in PATH."
fi

echo "Attempting to run simple Node.js server..."
if command_exists node && [ -f "simple_http_server.js" ]; then
  echo "Running simple_http_server.js with Node.js..."
  node simple_http_server.js &
  # Store the PID to kill it later if needed
  NODE_PID=$!
  # Wait a moment for server to start
  sleep 2
  # Check if server is still running
  if kill -0 $NODE_PID 2>/dev/null; then
    echo "Simple Node.js server started successfully with PID $NODE_PID."
    wait $NODE_PID
    exit $?
  else
    echo "Simple Node.js server failed to start or crashed."
  fi
else
  echo "Cannot run simple_http_server.js: Node.js not found in PATH."
fi

echo ""
echo "========================================================"
echo "DIRECT DIAGNOSTIC INFORMATION"
echo "========================================================"
echo "Since no server could be started, here's some diagnostic information:"
echo ""

# Memory information
echo "Memory information:"
free -h

# Disk space
echo ""
echo "Disk space:"
df -h .

# List files in current directory
echo ""
echo "Directory listing:"
ls -la | head -50

# Create a minimal index.html if it doesn't exist
if [ ! -f "index.html" ]; then
  echo "index.html not found. Creating a minimal diagnostic page..."
  cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BCBS Values System Diagnostic</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      margin: 0;
      padding: 20px;
      color: #333;
      max-width: 800px;
      margin: 0 auto;
    }
    h1 {
      color: #2c3e50;
      border-bottom: 2px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      color: #3498db;
      margin-top: 30px;
    }
    .card {
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status {
      padding: 8px 12px;
      border-radius: 4px;
      display: inline-block;
      font-weight: bold;
    }
    .error {
      background-color: #f8d7da;
      color: #721c24;
    }
    .warning {
      background-color: #fff3cd;
      color: #856404;
    }
    .info {
      background-color: #d1ecf1;
      color: #0c5460;
    }
    pre {
      background-color: #f5f5f5;
      padding: 10px;
      border-radius: 4px;
      overflow-x: auto;
    }
    .footer {
      margin-top: 40px;
      padding-top: 10px;
      border-top: 1px solid #eee;
      font-size: 0.9em;
      color: #777;
    }
  </style>
</head>
<body>
  <h1>BCBS Values System Diagnostic</h1>
  
  <div class="card">
    <h2>System Status</h2>
    <p><span class="status warning">DIAGNOSTIC MODE</span></p>
    <p>This is a static diagnostic page. The main application could not be started.</p>
  </div>
  
  <div class="card">
    <h2>Diagnostic Information</h2>
    <p>The BCBS Values system is experiencing technical difficulties. Here are some diagnostic details that may help troubleshoot the issue:</p>
    
    <h3>Environment</h3>
    <ul>
      <li><strong>Date:</strong> <span id="current-date">Unknown</span></li>
      <li><strong>Python:</strong> Not available</li>
      <li><strong>Node.js:</strong> Not available</li>
    </ul>
    
    <h3>Server Status</h3>
    <p><span class="status error">OFFLINE</span> The application server is not running.</p>
    
    <h3>Database Status</h3>
    <p><span class="status info">UNKNOWN</span> Cannot check database connection status at this time.</p>
  </div>
  
  <div class="card">
    <h2>Troubleshooting Steps</h2>
    <ol>
      <li>Check if Python and Node.js are properly installed and available in the PATH</li>
      <li>Verify that all required dependencies are installed</li>
      <li>Ensure database connection parameters are correct</li>
      <li>Check application logs for specific error messages</li>
    </ol>
  </div>
  
  <div class="footer">
    <p>BCBS Values Diagnostic - Generated at <span id="generated-time">Unknown</span></p>
  </div>
  
  <script>
    // Update dates
    document.getElementById('current-date').textContent = new Date().toLocaleString();
    document.getElementById('generated-time').textContent = new Date().toLocaleString();
  </script>
</body>
</html>
EOF
  echo "Created minimal index.html"
fi

echo ""
echo "========================================================"
echo "DIAGNOSTIC FALLBACK"
echo "========================================================"
echo "All server startup attempts failed. A static HTML diagnostic page has been created."
echo "If you're seeing this message in the terminal, try accessing the application via the web browser directly."

# Return an error code since no server could be started
exit 1