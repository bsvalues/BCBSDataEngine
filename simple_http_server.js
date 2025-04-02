/**
 * Simple HTTP Server for BCBS Dashboard
 * This server works with Node.js only
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Constants
const PORT = 5002;
const DEFAULT_PAGE = 'dashboard_static.html';
const ERROR_PAGE = '404.html';

// MIME types for different file extensions
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
  '.txt': 'text/plain'
};

/**
 * Create and start the HTTP server
 */
const server = http.createServer((req, res) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  
  // Get the requested path
  let filePath = req.url;
  
  // Handle root request
  if (filePath === '/') {
    filePath = `/${DEFAULT_PAGE}`;
  }
  
  // Remove query string if present
  filePath = filePath.split('?')[0];
  
  // Convert URL path to file path
  filePath = path.join(process.cwd(), filePath);
  
  // Get file extension to determine content type
  const extname = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Check if the file exists
  fs.readFile(filePath, (err, content) => {
    if (err) {
      // File not found
      console.log(`File not found: ${filePath}`);
      
      // Try to serve the error page
      const errorPagePath = path.join(process.cwd(), ERROR_PAGE);
      fs.readFile(errorPagePath, (err2, errorContent) => {
        if (err2) {
          // No error page, send simple 404 response
          res.writeHead(404, { 'Content-Type': 'text/plain' });
          res.end('404 Not Found');
        } else {
          // Send error page
          res.writeHead(404, { 'Content-Type': 'text/html' });
          res.end(errorContent);
        }
      });
    } else {
      // Success - send file content
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    }
  });
});

// Start the server and listen on the specified port
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
  console.log(`Serving files from ${process.cwd()}`);
  console.log(`Default page: ${DEFAULT_PAGE}`);
  console.log('Press Ctrl+C to stop the server');
});

// Handle server errors
server.on('error', (err) => {
  console.error('Server error:', err);
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  server.close(() => {
    console.log('Server stopped');
    process.exit(0);
  });
});