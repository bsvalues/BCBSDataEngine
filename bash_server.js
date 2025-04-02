// Simple HTTP server for serving static files
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5002;
const DEFAULT_FILE = 'dashboard_static.html';
const ERROR_FILE = '404.html';

// MIME types for common file extensions
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

// Create the server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Parse the URL
  let url = req.url;
  
  // Default to the dashboard file if root is requested
  if (url === '/' || url === '') {
    url = `/${DEFAULT_FILE}`;
  }
  
  // Get the file path
  let filePath = path.join(__dirname, url);
  
  // Check if file exists
  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      // Try to serve the error page
      const errorPath = path.join(__dirname, ERROR_FILE);
      
      fs.access(errorPath, fs.constants.F_OK, (errCheck) => {
        if (errCheck) {
          // Error page doesn't exist, send plain 404
          res.writeHead(404, { 'Content-Type': 'text/plain' });
          res.end('404 Not Found');
          return;
        }
        
        // Serve the error page
        const extname = path.extname(ERROR_FILE);
        const contentType = MIME_TYPES[extname] || 'text/plain';
        
        res.writeHead(404, { 'Content-Type': contentType });
        fs.createReadStream(errorPath).pipe(res);
      });
      
      return;
    }
    
    // Get the file extension and content type
    const extname = path.extname(filePath);
    const contentType = MIME_TYPES[extname] || 'text/plain';
    
    // Serve the file
    res.writeHead(200, { 'Content-Type': contentType });
    fs.createReadStream(filePath).pipe(res);
  });
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
  console.log(`Access the dashboard at http://localhost:${PORT}/`);
});