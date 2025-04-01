// Simple Node.js HTTP server as an emergency fallback
// This is used when the Python server cannot start in the workflow environment

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5002;

// Create HTTP server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Set default URL to index.html
  let url = req.url;
  if (url === '/') {
    url = '/index.html';
  }
  
  // Handle API requests
  if (url.startsWith('/api/')) {
    return handleApiRequest(req, res);
  }
  
  // Determine content type based on file extension
  const ext = path.extname(url);
  let contentType = 'text/html';
  
  switch (ext) {
    case '.js':
      contentType = 'text/javascript';
      break;
    case '.css':
      contentType = 'text/css';
      break;
    case '.json':
      contentType = 'application/json';
      break;
    case '.png':
      contentType = 'image/png';
      break;
    case '.jpg':
      contentType = 'image/jpeg';
      break;
    case '.svg':
      contentType = 'image/svg+xml';
      break;
  }
  
  // Read file from filesystem
  fs.readFile(`.${url}`, (err, data) => {
    if (err) {
      // If file not found, serve 404 page if available, or a simple message
      if (err.code === 'ENOENT') {
        fs.readFile('./404.html', (err404, data404) => {
          if (err404) {
            res.writeHead(404, { 'Content-Type': 'text/html' });
            res.end('<h1>404 Not Found</h1><p>The requested file was not found on the server.</p>');
          } else {
            res.writeHead(404, { 'Content-Type': 'text/html' });
            res.end(data404);
          }
        });
      } else {
        res.writeHead(500, { 'Content-Type': 'text/html' });
        res.end('<h1>500 Server Error</h1><p>Internal server error occurred.</p>');
      }
      return;
    }
    
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
});

// Handle API request and respond with JSON
function handleApiRequest(req, res) {
  const url = req.url;
  const parts = url.split('/');
  const endpoint = parts[2] || '';
  
  let responseData = {};
  
  // Basic API endpoints
  if (endpoint === 'data') {
    responseData = {
      status: 'success',
      data: {
        propertyCount: 12548,
        averageValue: 452000,
        recentProperties: [
          {address: "123 Main St", value: 350000},
          {address: "456 Oak Ave", value: 475000},
          {address: "789 Pine Blvd", value: 560000}
        ]
      }
    };
  } else if (endpoint === 'status') {
    responseData = {
      status: 'success',
      serverStatus: 'running',
      uptime: process.uptime(),
      timestamp: new Date().toISOString()
    };
  } else {
    responseData = {
      status: 'error',
      message: `Unknown endpoint: ${endpoint}`
    };
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(responseData, null, 2));
    return;
  }
  
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(responseData, null, 2));
}

// Start the server
server.listen(PORT, () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
  console.log(`Current working directory: ${process.cwd()}`);
  
  // List available HTML files
  const htmlFiles = fs.readdirSync('.').filter(file => file.endsWith('.html'));
  console.log(`HTML files available: ${htmlFiles.join(', ')}`);
});