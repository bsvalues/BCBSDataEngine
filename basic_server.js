// Basic Node.js HTTP server for BCBS Values Platform
const http = require('http');
const fs = require('fs');
const path = require('path');

// Port to listen on
const PORT = 5000;

// MIME type mapping for different file extensions
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// Default file to serve if no path is specified
const DEFAULT_FILE = 'index.html';

// Create the HTTP server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Get the file path from the URL
  let filePath = req.url === '/' ? DEFAULT_FILE : req.url;
  
  // Remove query parameters if any
  filePath = filePath.split('?')[0];
  
  // Resolve the file path to the current directory
  filePath = path.resolve(filePath.slice(1));
  
  // Get the file extension
  const extname = path.extname(filePath).toLowerCase();
  
  // Get the MIME type based on the file extension
  const contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Read the file
  fs.readFile(filePath, (err, data) => {
    if (err) {
      // If the file doesn't exist, serve the static fallback page
      if (err.code === 'ENOENT') {
        console.log(`File not found: ${filePath}, serving static_fallback.html`);
        fs.readFile('static_fallback.html', (err, data) => {
          if (err) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('404 Not Found');
            return;
          }
          res.writeHead(200, { 'Content-Type': 'text/html' });
          res.end(data, 'utf-8');
        });
        return;
      }
      
      // For other errors, return a 500 error
      console.error(`Error reading file: ${err}`);
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('500 Internal Server Error');
      return;
    }
    
    // Serve the file with the appropriate content type
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data, 'utf-8');
  });
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`BCBS Values Platform server running at http://0.0.0.0:${PORT}/`);
  console.log('Available pages:');
  console.log(`- Home: http://0.0.0.0:${PORT}/`);
  console.log(`- Static Fallback: http://0.0.0.0:${PORT}/static_fallback.html`);
  console.log(`- Dashboard: http://0.0.0.0:${PORT}/dashboard.html`);
  console.log(`- What-If Analysis: http://0.0.0.0:${PORT}/what-if-analysis.html`);
  console.log(`- Agent Dashboard: http://0.0.0.0:${PORT}/agent-dashboard.html`);
});