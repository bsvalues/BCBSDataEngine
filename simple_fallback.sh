#!/bin/bash

echo "BCBS Values Platform Simple Fallback Server"
echo "=========================================="

# Try to find the python executable
for py_path in $(which python3) /nix/store/*python*3.11*/bin/python3.11 /usr/bin/python3.11 /usr/local/bin/python3.11 /mnt/nixmodules/nix/store/*python*3.11*/bin/python3.11; do
  if [ -x "$py_path" ]; then
    echo "Using Python at: $py_path"
    
    # Check if we have the simple server
    if [ -f "simple_python_server.py" ]; then
      echo "Starting simple Python server..."
      chmod +x simple_python_server.py
      exec "$py_path" simple_python_server.py
      exit 0
    fi
    
    # If not, use the built-in http.server module
    echo "Starting built-in Python HTTP server..."
    exec "$py_path" -m http.server 5000 --bind 0.0.0.0
    exit 0
  fi
done

echo "Python not found. Trying Node.js..."

# Try to find the node executable
for node_path in $(which node) /nix/store/*nodejs*20*/bin/node /nix/store/*nodejs*18*/bin/node /usr/bin/node /usr/local/bin/node; do
  if [ -x "$node_path" ]; then
    echo "Using Node.js at: $node_path"
    
    # Check if we have the node server
    if [ -f "server.js" ]; then
      echo "Starting Node.js server..."
      exec "$node_path" server.js
      exit 0
    fi
    
    # If not, create a minimal HTTP server
    echo "Creating minimal Node.js HTTP server..."
    cat > minimal_server.js << 'EOF'
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
const server = http.createServer((req, res) => {
  console.log('Request:', req.url);
  let filePath = '.' + req.url;
  if (filePath === './') filePath = './index.html';
  
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('File not found!');
      return;
    }
    res.writeHead(200);
    res.end(data);
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
});
EOF
    
    exec "$node_path" minimal_server.js
    exit 0
  fi
done

echo "Error: No Python or Node.js found!"
echo "Falling back to static file listing:"
ls -la *.html

exit 1