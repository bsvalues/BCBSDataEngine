#!/bin/bash
# reliable_startup.sh
# Reliable startup script for BCBS Values Platform server
# Features:
# - Robust Python path detection
# - Process management
# - Multiple startup methods
# - Error logging

# Configuration
PORT=5002
MAX_RETRIES=5
RETRY_DELAY=2
LOG_FILE="server_startup.log"
PID_FILE="server.pid"

# Known Python paths to try
PYTHON_PATHS=(
  "/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
  "/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"
  "/nix/store/clx0mcir7qw8zk36zbr4jra789g3knf6-python3-3.11.10/bin/python3.11"
  "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3.11"
  "/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"
  "python3.11"
  "python3"
  "python"
)

# Clear log file
> "$LOG_FILE"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE"
}

check_port() {
  log "Checking if port $PORT is available..."
  if netstat -tuln | grep -q ":$PORT"; then
    error_log "Port $PORT is already in use."
    
    log "Attempting to identify process using port $PORT..."
    if command -v lsof >/dev/null 2>&1; then
      lsof -i :"$PORT" | tee -a "$LOG_FILE"
    elif command -v ss >/dev/null 2>&1; then
      ss -lptn "sport = :$PORT" | tee -a "$LOG_FILE"
    elif command -v netstat >/dev/null 2>&1; then
      netstat -tlpn | grep ":$PORT" | tee -a "$LOG_FILE"
    fi
    
    log "Attempting to kill process using port $PORT..."
    for pid in $(lsof -t -i:"$PORT" 2>/dev/null || ss -lptn "sport = :$PORT" 2>/dev/null | awk 'NR>1 {print $6}' | cut -d',' -f2 | cut -d'=' -f2 || netstat -tlpn | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1); do
      if [ -n "$pid" ] && [ "$pid" != "" ]; then
        log "Killing process $pid using port $PORT..."
        kill -15 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
        sleep 1
      fi
    done
    
    # Double check
    if netstat -tuln | grep -q ":$PORT"; then
      error_log "Failed to free port $PORT. Server might not start correctly."
      return 1
    else
      log "Successfully freed port $PORT"
      return 0
    fi
  else
    log "Port $PORT is available"
    return 0
  fi
}

find_python() {
  log "Searching for Python interpreter..."
  
  for path in "${PYTHON_PATHS[@]}"; do
    log "Trying $path..."
    if [ -x "$path" ]; then
      if "$path" -c "print('Python works')" &>/dev/null; then
        PYTHON="$path"
        log "✅ Found working Python at $PYTHON"
        return 0
      else
        log "❌ $path exists but failed to run Python"
      fi
    else
      log "❌ $path not found or not executable"
    fi
  done
  
  # Last resort: try the which command
  if command -v python3 &>/dev/null; then
    PYTHON=$(which python3)
    log "✅ Found Python via which: $PYTHON"
    return 0
  elif command -v python &>/dev/null; then
    PYTHON=$(which python)
    log "✅ Found Python via which: $PYTHON"
    return 0
  fi
  
  error_log "No working Python interpreter found."
  return 1
}

check_server_files() {
  # Check for the server Python script
  if [ -f "enhanced_server.py" ]; then
    SERVER_SCRIPT="enhanced_server.py"
    log "Found enhanced server script: $SERVER_SCRIPT"
  elif [ -f "simple_server.py" ]; then
    SERVER_SCRIPT="simple_server.py"
    log "Found simple server script: $SERVER_SCRIPT"
  else
    error_log "No server script found (enhanced_server.py or simple_server.py)"
    return 1
  fi
  
  # Ensure the script is executable
  chmod +x "$SERVER_SCRIPT"
  
  return 0
}

kill_existing_server() {
  log "Checking for existing server processes..."
  
  # Try PID file first
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [ -n "$PID" ] && ps -p "$PID" &>/dev/null; then
      log "Found existing server process with PID $PID, killing it..."
      kill -15 "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
      sleep 1
      
      # Remove stale PID file
      rm -f "$PID_FILE"
    else
      log "PID file exists but process is not running."
      rm -f "$PID_FILE"
    fi
  fi
  
  # Kill any Python processes running our server scripts
  for script in "enhanced_server.py" "simple_server.py"; do
    if pgrep -f "python.*$script" &>/dev/null; then
      log "Killing existing $script processes..."
      pkill -9 -f "python.*$script" || true
    fi
  done
  
  sleep 1
}

wait_for_server() {
  log "Waiting for server to start (port $PORT)..."
  
  # Try to connect to the server
  for i in $(seq 1 "$MAX_RETRIES"); do
    if curl -s "http://localhost:$PORT/" &>/dev/null; then
      log "✅ Server is up and responding"
      return 0
    else
      log "Waiting for server (attempt $i/$MAX_RETRIES)..."
      sleep "$RETRY_DELAY"
    fi
  done
  
  error_log "Server did not respond after $MAX_RETRIES attempts."
  return 1
}

start_server() {
  log "Starting BCBS Values Platform server..."
  
  # Double check if we have all we need
  if [ -z "$PYTHON" ] || [ -z "$SERVER_SCRIPT" ]; then
    error_log "Missing Python interpreter or server script."
    return 1
  fi
  
  # Start the server in the background
  log "Executing: $PYTHON $SERVER_SCRIPT"
  nohup "$PYTHON" "$SERVER_SCRIPT" > server_output.log 2>&1 &
  
  # Verify server started
  if [ $? -ne 0 ]; then
    error_log "Failed to start server."
    return 1
  fi
  
  log "Server process started."
  
  # Wait for server to be responsive
  if wait_for_server; then
    log "✅ Server started successfully!"
    
    # Print access information
    SERVER_URL="http://localhost:$PORT"
    log "Server is running at $SERVER_URL"
    log "Dashboard URL: $SERVER_URL/dashboard.html"
    return 0
  else
    error_log "Server failed to respond. Check server_output.log for details."
    tail -20 server_output.log | tee -a "$LOG_FILE"
    return 1
  fi
}

main() {
  log "=== BCBS Values Platform Server Startup ==="
  log "Starting server initialization process..."
  
  # Kill any existing server processes
  kill_existing_server
  
  # Check if port is available
  check_port
  
  # Find Python interpreter
  if ! find_python; then
    exit 1
  fi
  
  # Check server files
  if ! check_server_files; then
    exit 1
  fi
  
  # Start server
  if start_server; then
    log "Server startup completed successfully."
    exit 0
  else
    error_log "Server startup failed."
    exit 1
  fi
}

# Run the main function
main