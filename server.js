/**
 * Simple HTTP server for BCBS Values static diagnostic page
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

// Configuration
const PORT = process.env.PORT || 5000;

// Create a server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Basic routing
  let filePath = '';
  let contentType = 'text/html';
  
  // Map URL path to file path
  if (req.url === '/' || req.url === '/index.html') {
    filePath = 'index.html';
  } else if (req.url === '/api/health') {
    contentType = 'application/json';
    const healthData = {
      status: 'ok',
      message: 'Diagnostic server is running in Node.js mode',
      timestamp: new Date().toISOString(),
      serverInfo: {
        nodeVersion: process.version,
        platform: process.platform,
        hostname: require('os').hostname()
      }
    };
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(JSON.stringify(healthData, null, 2));
    return;
  } else {
    filePath = req.url.substring(1);
  }
  
  // Check if the file exists
  const fullPath = path.join(__dirname, filePath);
  fs.access(fullPath, fs.constants.F_OK, (err) => {
    if (err) {
      // File not found, serve 404
      serveFile('404.html', res, 404) || serve404Fallback(res);
      return;
    }
    
    // Set content type based on file extension
    const extname = path.extname(filePath);
    switch (extname) {
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
      case '.jpeg':
        contentType = 'image/jpeg';
        break;
      case '.svg':
        contentType = 'image/svg+xml';
        break;
    }
    
    // Read and serve the file
    fs.readFile(fullPath, (err, content) => {
      if (err) {
        if (err.code === 'ENOENT') {
          // File not found
          serveFile('404.html', res, 404) || serve404Fallback(res);
        } else {
          // Server error
          res.writeHead(500);
          res.end(`Server Error: ${err.code}`);
        }
        return;
      }
      
      // Success - serve the file
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    });
  });
});

// Helper function to serve a file
function serveFile(filePath, res, statusCode = 200) {
  try {
    const fullPath = path.join(__dirname, filePath);
    const content = fs.readFileSync(fullPath, 'utf-8');
    res.writeHead(statusCode, { 'Content-Type': 'text/html' });
    res.end(content);
    return true;
  } catch (err) {
    return false;
  }
}

// Fallback 404 page if 404.html doesn't exist
function serve404Fallback(res) {
  const html = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>404 - Not Found</title>
      <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 650px; margin: 40px auto; padding: 0 10px; }
        h1 { color: #e74c3c; }
        a { color: #3498db; }
      </style>
    </head>
    <body>
      <h1>404 - Not Found</h1>
      <p>The requested resource was not found on this server.</p>
      <p><a href="/">Return to home page</a></p>
    </body>
    </html>
  `;
  res.writeHead(404, { 'Content-Type': 'text/html' });
  res.end(html);
}

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Diagnostic server running on http://0.0.0.0:${PORT}`);
  console.log(`Server started at: ${new Date().toISOString()}`);
});