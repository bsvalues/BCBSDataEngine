#!/bin/bash

# BCBS Values Diagnostic Script
# This script attempts to run various diagnostics with a focus on Node.js

echo "========================================================"
echo "BCBS VALUES DIAGNOSTIC SCRIPT"
echo "========================================================"
echo "Current directory: $(pwd)"
echo "Date: $(date)"

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Set default port
PORT=${PORT:-5000}
export PORT

# Define multiple possible Node.js paths
NODE_PATHS=(
  "/nix/store/*/bin/node"  # Try to find any Node in Nix store
  "/usr/bin/node"
  "/usr/local/bin/node"
  "node"
)

# Try our custom diagnostic script first
echo "Trying custom diagnostic script..."
if [ -f "custom_diagnostic.sh" ]; then
  echo "Found custom_diagnostic.sh. Running..."
  if bash custom_diagnostic.sh; then
    echo "Custom diagnostic script completed successfully."
    exit 0
  else
    echo "Custom diagnostic script failed. Trying alternative methods..."
  fi
else
  echo "custom_diagnostic.sh not found. Trying alternative methods..."
fi

# Try Python-based server
echo "Trying Python-based diagnostic server..."

# Function to try running simple_python_server.py with a given Python interpreter
try_python_diagnostic() {
  local python_cmd="$1"
  echo "Checking Python at: $python_cmd"
  
  if [ -x "$python_cmd" ] || command_exists "$python_cmd"; then
    echo "Python found at $python_cmd."
    
    # Try our Python server first
    if [ -f "simple_python_server.py" ]; then
      echo "Found simple_python_server.py. Running..."
      echo "You can access the diagnostic server at: http://0.0.0.0:5000"
      chmod +x simple_python_server.py
      $python_cmd simple_python_server.py
      local exit_code=$?
      echo "Python server has stopped with exit code $exit_code."
      return $exit_code
    fi
    
    echo "No Python server scripts found."
    return 1
  else
    echo "Python not found at $python_cmd."
    return 1
  fi
}

# Define Python common paths
PYTHON_COMMON_PATHS=(
  "/nix/store/*/bin/python3"
  "/usr/bin/python3"
  "/usr/local/bin/python3"
  "python3"
)

# Try each Python path
python_success=false
for python_path in "${PYTHON_COMMON_PATHS[@]}"; do
  # If path contains a wildcard, try to expand it
  if [[ "$python_path" == *"*"* ]]; then
    # Try to find actual paths matching the pattern
    for actual_path in $python_path; do
      if [ -x "$actual_path" ]; then
        echo "Found Python at: $actual_path"
        if try_python_diagnostic "$actual_path"; then
          python_success=true
          break 2
        fi
      fi
    done
  else
    # Regular path, no wildcard
    if try_python_diagnostic "$python_path"; then
      python_success=true
      break
    fi
  fi
done

# If Python was successful, exit now
if $python_success; then
  exit 0
fi

# Try Node.js-based server
echo "Trying Node.js diagnostic server..."

# Function to try running server.js with a given Node interpreter
try_node_diagnostic() {
  local node_cmd="$1"
  echo "Checking Node.js at: $node_cmd"
  
  if [ -x "$node_cmd" ] || command_exists "$node_cmd"; then
    echo "Node.js found at $node_cmd."
    
    # First try server.js
    if [ -f "server.js" ]; then
      echo "Found server.js. Running..."
      echo "You can access the diagnostic server at: http://0.0.0.0:5000"
      $node_cmd server.js
      local exit_code=$?
      echo "Node.js server has stopped with exit code $exit_code."
      return $exit_code
    fi
    
    # Then try simple_http_server.js
    if [ -f "simple_http_server.js" ]; then
      echo "Found simple_http_server.js. Running..."
      echo "You can access the diagnostic server at: http://0.0.0.0:5000"
      $node_cmd simple_http_server.js
      local exit_code=$?
      echo "Simple Node.js server has stopped with exit code $exit_code."
      return $exit_code
    fi
    
    echo "No Node.js server scripts found."
    return 1
  else
    echo "Node.js not found at $node_cmd."
    return 1
  fi
}

# Try each Node.js path
node_success=false
for node_path in "${NODE_PATHS[@]}"; do
  # If path contains a wildcard, try to expand it
  if [[ "$node_path" == *"*"* ]]; then
    # Try to find actual paths matching the pattern
    for actual_path in $node_path; do
      if [ -x "$actual_path" ]; then
        echo "Found Node.js at: $actual_path"
        if try_node_diagnostic "$actual_path"; then
          node_success=true
          break 2
        fi
      fi
    done
  else
    # Regular path, no wildcard
    if try_node_diagnostic "$node_path"; then
      node_success=true
      break
    fi
  fi
done

# If Node.js was successful, exit now
if $node_success; then
  exit 0
fi

# Check if we have direct_server.js available
if [ -f "direct_server.js" ]; then
  echo "Found direct_server.js. Running with bash directly..."
  # This is a special fallback server designed to work with minimal dependencies
  bash -c "cat direct_server.js"
  bash ./direct_server.js
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Direct server completed successfully."
    exit 0
  else
    echo "Direct server failed with exit code $exit_code."
  fi
fi

# Try our bash-based diagnostic server
if [ -f "simple_bash_server.sh" ]; then
  echo "Found simple_bash_server.sh. Running..."
  echo "You can access the diagnostic server at: http://0.0.0.0:5000"
  bash ./simple_bash_server.sh
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Bash-based server completed successfully."
    exit 0
  else
    echo "Bash-based server failed with exit code $exit_code."
  fi
fi

# Final fallback: Just show plain text diagnostic info
echo "========================================================"
echo "STATIC DIAGNOSTIC INFORMATION"
echo "========================================================"
echo "No server could be started. Here's some basic diagnostic information:"

echo "System:"
uname -a

echo "Memory:"
free -h || echo "free command not available"

echo "Disk space:"
df -h . || echo "df command not available" 

echo "Python paths:"
# Check some common Python paths
PYTHON_COMMON_PATHS=(
  "/nix/store/*/bin/python3"
  "/usr/bin/python3"
  "/usr/local/bin/python3"
  "python3"
)

for path in "${PYTHON_COMMON_PATHS[@]}"; do
  if [[ "$path" == *"*"* ]]; then
    echo "- Checking wildcard path: $path"
    # List any matching paths
    for actual_path in $path 2>/dev/null; do
      if [ -e "$actual_path" ]; then
        echo "  - Found: $actual_path"
      fi
    done
  else
    if command_exists "$path"; then
      echo "- Exists: $path"
    else
      echo "- Missing: $path"
    fi
  fi
done

echo "Environment:"
echo "- PATH: $PATH"
echo "- PWD: $PWD"
echo "- PORT: $PORT"

echo "Files:"
for file in "index.html" "simple_diagnostic.py" "server.js" "simple_http_server.js" "direct_server.js" "simple_bash_server.sh"; do
  if [ -f "$file" ]; then
    echo "$file exists"
  else 
    echo "$file does not exist"
  fi
done

echo "========================================================"
echo "DIAGNOSTICS COMPLETE"
echo "========================================================"
echo "All diagnostic options have been exhausted."
echo "Please check the diagnostic information above."
echo "Try running one of these commands directly:"
echo "1. /mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3 simple_diagnostic.py"
echo "2. bash direct_server.js (if available)"
echo "3. bash ./simple_bash_server.sh (if available)"

# Try to serve a static HTML page as last resort
if [ -f "index.html" ]; then
  echo "Serving index.html with a simple inline server..."
  echo "You can access the static page at: http://0.0.0.0:5000"
  (echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n$(cat index.html)" | nc -l -p 5000 || echo "Failed to start netcat server")
fi

exit 1