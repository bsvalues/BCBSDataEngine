#!/bin/bash

# Simplified workflow runner that doesn't try to use python directly
echo "Starting BCBS Values Platform..."

# Create a very minimal Node.js server script
cat > minimal_server.js << 'EOF'
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Convert URL to file path
  let filePath = '.' + req.url;
  if (filePath === './') {
    filePath = './index.html';
  }
  
  // Get file extension
  const extname = path.extname(filePath);
  let contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Read the file
  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // Page not found - try to serve static_fallback.html
        fs.readFile('./static_fallback.html', (err, content) => {
          if (err) {
            res.writeHead(404);
            res.end('Page not found');
          } else {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(content, 'utf-8');
          }
        });
      } else {
        // Server error
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      }
    } else {
      // Success
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`BCBS Values Platform server running at http://0.0.0.0:${PORT}/`);
  console.log('Available pages:');
  console.log(`- Home: http://0.0.0.0:${PORT}/`);
  console.log(`- Dashboard: http://0.0.0.0:${PORT}/dashboard.html`);
  console.log(`- Static Fallback: http://0.0.0.0:${PORT}/static_fallback.html`);
});
EOF

# Try to run with node
echo "Attempting to start Node.js server..."
node minimal_server.js