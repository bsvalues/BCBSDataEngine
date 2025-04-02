const fs = require('fs');
const http = require('http');
const path = require('path');

// Configuration
const PORT = 5002;
const DEFAULT_FILE = 'dashboard_static.html';
const ERROR_FILE = '404.html';

// MIME types
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml'
};

// Create HTTP server
const server = http.createServer((req, res) => {
  console.log(`Request: ${req.method} ${req.url}`);
  
  // Handle the request
  handleRequest(req, res);
});

// Start server
server.listen(PORT, () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
  console.log(`Default file: ${DEFAULT_FILE}`);
});

// Handle requests
function handleRequest(req, res) {
  // Get the file path
  let filePath = req.url;
  
  // Default to index file for root requests
  if (filePath === '/') {
    filePath = `/${DEFAULT_FILE}`;
  }
  
  // Get absolute path
  filePath = path.join(process.cwd(), filePath.substring(1));
  
  // Get file extension
  const extname = path.extname(filePath);
  
  // Set default content type if no extension
  let contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Read the file
  fs.readFile(filePath, (err, content) => {
    if (err) {
      // Handle error (file not found, etc.)
      handleError(res, err);
      return;
    }
    
    // Success - return the file
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content, 'utf-8');
  });
}

// Handle errors
function handleError(res, err) {
  if (err.code === 'ENOENT') {
    // File not found - try to serve 404 page
    fs.readFile(path.join(process.cwd(), ERROR_FILE), (error, content) => {
      if (error) {
        // No custom 404 page, send basic response
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<html><body><h1>404 Not Found</h1><p>The resource you requested could not be found.</p></body></html>', 'utf-8');
      } else {
        // Send custom 404 page
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end(content, 'utf-8');
      }
    });
  } else {
    // Server error
    res.writeHead(500, { 'Content-Type': 'text/html' });
    res.end(`<html><body><h1>500 Server Error</h1><p>${err.code}</p></body></html>`, 'utf-8');
  }
}