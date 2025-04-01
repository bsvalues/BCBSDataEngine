// Simple Node.js server for the BCBS Values Platform Dashboard with Micro-animations
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5000;

// MIME types for different file extensions
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// Create the HTTP server
const server = http.createServer((req, res) => {
  console.log(`Request received: ${req.method} ${req.url}`);
  
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle OPTIONS requests for CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  
  // Parse the URL
  let url = req.url;
  
  // If the URL is just '/', serve the index.html file
  if (url === '/') {
    url = '/index.html';
  } else if (url === '/dashboard') {
    url = '/dashboard.html';
  }
  
  // Resolve the file path
  const filePath = path.resolve(__dirname, '.' + url);
  
  // Get the file extension
  const extname = path.extname(filePath);
  
  // Default content type
  let contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Read the file
  fs.readFile(filePath, (err, content) => {
    if (err) {
      // If the file is not found
      if (err.code === 'ENOENT') {
        // Check if we have an index.html fallback
        if (url !== '/index.html') {
          console.log(`File not found: ${filePath}, redirecting to /index.html`);
          
          // Redirect to index.html
          res.writeHead(302, {
            'Location': '/index.html'
          });
          res.end();
        } else {
          // No fallback available
          console.error(`No fallback available for ${filePath}`);
          res.writeHead(404);
          res.end('404 Not Found');
        }
      } else {
        // Server error
        console.error(`Server error: ${err.code}`);
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      }
    } else {
      // Success - file found
      console.log(`Serving file: ${filePath} with content type: ${contentType}`);
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf8');
    }
  });
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log('BCBS Values Platform Dashboard Server');
  console.log('====================================');
  console.log(`Server running on http://0.0.0.0:${PORT}/`);
  console.log(`Access the enhanced dashboard at http://0.0.0.0:${PORT}/dashboard.html`);
  
  // List HTML files found
  console.log('\nHTML files found:');
  fs.readdir('.', (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      return;
    }
    
    files.filter(file => file.endsWith('.html')).forEach(file => {
      console.log(`  - ${file}`);
    });
  });
});