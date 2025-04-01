// Ultra-simple Node.js HTTP server

const http = require('http');
const fs = require('fs');
const PORT = process.env.PORT || 5000;

console.log("Starting ultra-simple Node.js server on port", PORT);

// Create a very basic HTML page
const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <title>BCBS Values Emergency Server</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
    h1 { color: #d9534f; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  </style>
</head>
<body>
  <h1>BCBS Values - Emergency Node.js Server</h1>
  
  <div class="card">
    <p>This is an emergency diagnostic page served by a minimal Node.js server.</p>
    <p>Server started at: ${new Date().toLocaleString()}</p>
  </div>
  
  <div class="card">
    <h2>Server Information</h2>
    <pre>
Node.js Version: ${process.version}
Working Directory: ${process.cwd()}
Date: ${new Date().toLocaleString()}
    </pre>
  </div>
</body>
</html>
`;

// Create the server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toLocaleString()} - Request: ${req.method} ${req.url}`);
  
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(htmlContent);
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
});

// Handle server errors
server.on('error', (err) => {
  console.error('Server error:', err);
  
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use.`);
  }
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('Server shutting down...');
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });
});