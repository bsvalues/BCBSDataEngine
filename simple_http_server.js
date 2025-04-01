/**
 * BCBS Ultra-Simple HTTP Server
 * A minimal server with no dependencies to run basic diagnostics
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = process.env.PORT || 5000;
const HOST = '0.0.0.0';

console.log("===================================");
console.log("BCBS VALUES SIMPLE HTTP SERVER");
console.log("===================================");
console.log("Starting server on " + HOST + ":" + PORT);
console.log("Current directory: " + process.cwd());
console.log("Node.js version: " + process.version);
console.log("Started at: " + new Date().toISOString());
console.log("===================================");

// Check if index.html exists, create a basic one if not
if (!fs.existsSync('index.html')) {
  console.log('Creating basic index.html file...');
  const basicHtml = `
<!DOCTYPE html>
<html>
  <head>
    <title>BCBS Values Diagnostic Server</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; line-height: 1.6; }
      h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
      h2 { color: #0066cc; margin-top: 30px; }
      pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
      .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
      .info { color: #004085; background-color: #cce5ff; padding: 10px; border-radius: 5px; }
    </style>
  </head>
  <body>
    <h1>BCBS Values Diagnostic Server</h1>
    <div class="info">
      <p>This is a minimal diagnostic page created by the BCBS Values Ultra-Simple HTTP Server.</p>
      <p>Server started at: ${new Date().toISOString()}</p>
    </div>

    <div class="card">
      <h2>Server Information</h2>
      <pre>
Node.js Version: ${process.version}
Working Directory: ${process.cwd()}
Port: ${PORT}
Date: ${new Date().toISOString()}
      </pre>
    </div>

    <div class="card">
      <h2>Available Files</h2>
      <pre>
${fs.readdirSync('.').join('\n')}
      </pre>
    </div>
  </body>
</html>
  `;
  fs.writeFileSync('index.html', basicHtml);
  console.log('Basic index.html file created.');
}

// Create and start HTTP server
const server = http.createServer((req, res) => {
  console.log(new Date().toISOString() + " - " + req.method + " " + req.url);
  
  // Parse URL
  const parsedUrl = url.parse(req.url);
  let pathname = parsedUrl.pathname;
  
  // Normalize pathname to prevent directory traversal
  const normalizedPath = path.normalize(pathname).replace(/^(\.\.[\/\\])+/, '');
  
  // Map pathname to local filesystem path
  let filePath = path.join(process.cwd(), normalizedPath === '/' ? '/index.html' : normalizedPath);
  
  // Determine content type based on file extension
  const extname = path.extname(filePath);
  let contentType = 'text/html';
  
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

  // Read the file and serve it
  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // Page not found - return 404
        console.log('File not found: ' + filePath);
        
        const notFoundMessage = `
          <!DOCTYPE html>
          <html>
            <head>
              <title>404 Not Found</title>
              <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; line-height: 1.6; }
                h1 { color: #dc3545; }
                pre { background: #f8d7da; padding: 15px; color: #721c24; border-radius: 5px; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
              </style>
            </head>
            <body>
              <h1>404 - File Not Found</h1>
              <p>The requested URL ${pathname} was not found on this server.</p>
              <p><a href="/">Go to Homepage</a></p>
              <pre>
Path: ${filePath}
Current directory: ${process.cwd()}
              </pre>
            </body>
          </html>
        `;
        
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end(notFoundMessage);
      } else {
        // Server error - return 500
        console.error('Server error:', err);
        
        const errorMessage = `
          <!DOCTYPE html>
          <html>
            <head>
              <title>500 Server Error</title>
              <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; line-height: 1.6; }
                h1 { color: #dc3545; }
                pre { background: #f8d7da; padding: 15px; color: #721c24; border-radius: 5px; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
              </style>
            </head>
            <body>
              <h1>500 - Internal Server Error</h1>
              <p>An error occurred while serving the requested file.</p>
              <p><a href="/">Go to Homepage</a></p>
              <pre>
Error: ${err.message}
Path: ${filePath}
              </pre>
            </body>
          </html>
        `;
        
        res.writeHead(500, { 'Content-Type': 'text/html' });
        res.end(errorMessage);
      }
    } else {
      // Success - serve the file
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    }
  });
});

// Start the server and register error handlers
server.listen(PORT, HOST, () => {
  console.log(`Server running at http://${HOST}:${PORT}/`);

  // Report files in current directory for diagnostic purposes
  console.log("\nCurrent directory files:");
  try {
    const files = fs.readdirSync('.');
    files.forEach(file => {
      const stats = fs.statSync(file);
      console.log(`- ${file} (${stats.isDirectory() ? 'directory' : 'file'}, ${stats.size} bytes)`);
    });
  } catch (error) {
    console.error('Error reading directory:', error);
  }
});

// Handle server errors
server.on('error', (err) => {
  console.error('Server error:', err);
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Please try a different port.`);
    process.exit(1);
  }
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('\nServer shutting down...');
  server.close(() => {
    console.log('Server stopped.');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('\nServer shutting down...');
  server.close(() => {
    console.log('Server stopped.');
    process.exit(0);
  });
});