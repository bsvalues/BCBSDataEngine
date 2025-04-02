#!/bin/bash
# Minimal BASH HTTP Server without any dependencies

# Configuration
PORT=5002
HOST="0.0.0.0"

# Create a status HTML page
cat > index.html << 'EOF'
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
        <p>Server time: CURRENT_TIME</p>
        <p>Client time: <script>document.write(new Date().toLocaleString());</script></p>
    </div>
</body>
</html>
EOF

# Log start
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting minimal BASH HTTP server on $HOST:$PORT"

# Check if netcat is available
if command -v nc &> /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found netcat, using it for server"
    
    # Keep running the server
    while true; do
        # Update timestamp
        CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        sed "s/CURRENT_TIME/$CURRENT_TIME/g" index.html > response.html
        
        # Serve the page
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for connection..."
        RESPONSE=$(cat response.html)
        HTTP_RESPONSE="HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: ${#RESPONSE}\r\nConnection: close\r\n\r\n$RESPONSE"
        echo -e "$HTTP_RESPONSE" | nc -l -p "$PORT" || {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Netcat error, retrying..."
            sleep 2
        }
    done
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Netcat not found, cannot start server"
    # Just keep the script running
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Keeping workflow active..."
    sleep 86400
fi