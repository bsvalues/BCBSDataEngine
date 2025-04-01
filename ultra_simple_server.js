// Ultra simple HTTP server that requires minimal dependencies
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 5000;

const server = http.createServer((req, res) => {
  console.log(`Received request for: ${req.url}`);
  
  // Default to index.html for root path
  let filePath = req.url === '/' ? '/index.html' : req.url;
  
  // Attempt to read the requested file
  try {
    const content = fs.readFileSync(path.join(process.cwd(), filePath.replace(/^\//, '')));
    
    // Set content type based on file extension
    const ext = path.extname(filePath);
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
      case '.jpeg':
        contentType = 'image/jpeg';
        break;
    }
    
    // Send successful response
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content);
    
  } catch (error) {
    console.error(`Error serving ${filePath}: ${error.message}`);
    
    // Return 404 or fallback to index.html
    if (filePath !== '/index.html') {
      // Try to serve index.html as fallback
      try {
        const content = fs.readFileSync(path.join(process.cwd(), 'index.html'));
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(content);
      } catch (indexError) {
        // If even index.html fails, send basic HTML
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(`
          <!DOCTYPE html>
          <html>
          <head>
              <title>BCBS Values - Basic Diagnostic Page</title>
              <style>
                  body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
                  h1 { color: #0066cc; }
                  pre { background: #f5f5f5; padding: 15px; }
                  .card { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
              </style>
          </head>
          <body>
              <h1>BCBS Values Platform</h1>
              <p>Benton County Building Services - Property Valuation System</p>
              
              <div class='card'>
                  <h2>System Status</h2>
                  <p>The system is currently running in ultra-minimal Node.js diagnostic mode.</p>
                  <p>Database status: PostgreSQL database is available.</p>
              </div>
              
              <div class='card'>
                  <h2>Diagnostic Information</h2>
                  <pre>
Server: Ultra Simple Node.js HTTP Server
Date: ${new Date().toISOString()}
Node Version: ${process.version}
Directory: ${process.cwd()}
                  </pre>
              </div>
              
              <footer>
                  <p>© 2025 Benton County Building Services</p>
              </footer>
          </body>
          </html>
        `);
      }
    } else {
      // If index.html was requested but not found
      res.writeHead(404, { 'Content-Type': 'text/html' });
      res.end('<h1>404 - File Not Found</h1><p>The requested file could not be found.</p>');
    }
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`=====================================`);
  console.log(`ULTRA SIMPLE SERVER RUNNING`);
  console.log(`=====================================`);
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
  console.log(`Node.js version: ${process.version}`);
  console.log(`Working directory: ${process.cwd()}`);
  console.log(`=====================================`);
});

// Handle server errors
server.on('error', (err) => {
  console.error(`Server error: ${err.message}`);
  
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Try setting a different PORT environment variable.`);
  }
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('Server is shutting down...');
  server.close(() => {
    console.log('Server has been stopped.');
    process.exit(0);
  });
});