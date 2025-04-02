/**
 * Simple Node.js HTTP server for BCBS Dashboard
 * This server requires Node.js but doesn't need any external dependencies
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Server configuration
const PORT = 8080;
const DEFAULT_PAGE = 'dashboard_static.html';

// MIME types for common file extensions
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// Create HTTP server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Parse URL and handle root path
  let url = req.url;
  if (url === '/' || url === '') {
    url = `/${DEFAULT_PAGE}`;
    console.log(`Serving default page: ${url}`);
  }

  // Remove query parameters if present
  const urlPath = url.split('?')[0];
  
  // Get file path
  let filePath = path.join(process.cwd(), urlPath);
  
  // Check if file exists
  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      console.error(`File not found: ${filePath}`);
      res.writeHead(404);
      res.end('404 Not Found');
      return;
    }
    
    // Get file extension and set content type
    const ext = path.extname(filePath);
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    
    // Read and serve file
    fs.readFile(filePath, (err, data) => {
      if (err) {
        console.error(`Error reading file: ${err.message}`);
        res.writeHead(500);
        res.end('500 Internal Server Error');
        return;
      }
      
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(data);
    });
  });
});

// Start server
console.log(`Starting HTTP server on http://0.0.0.0:${PORT}/`);
console.log(`Default page: ${DEFAULT_PAGE}`);
console.log(`Current time: ${new Date().toISOString()}`);
console.log("Server will remain running until manually stopped");

// Verify default page exists
try {
  fs.accessSync(DEFAULT_PAGE, fs.constants.F_OK);
  console.log(`Default page exists: ${DEFAULT_PAGE}`);
} catch (err) {
  console.warn(`WARNING: Default page '${DEFAULT_PAGE}' not found!`);
  console.log(`Current directory contains: ${fs.readdirSync('.').join(', ')}`);
}

// Start listening
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
});