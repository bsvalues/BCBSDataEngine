#!/usr/bin/env bash
# This file is dual purpose - it's a bash script that extracts and executes embedded JavaScript code
# It allows us to embed a Node.js server in a bash script for flexibility

# Check if node exists, extract the JS part and run it
if command -v node >/dev/null 2>&1; then
  echo "Node.js found, executing server..."
  # Extract JS code (everything after the marker) and pipe to node
  sed -n '/^\/\/ --- BEGIN JAVASCRIPT ---$/,$ p' "$0" | tail -n +2 | node -
  exit $?
fi

# If node isn't available, try to find another node
NODE_PATHS=(
  "/nix/store/*/bin/node"
  "/usr/local/bin/node"
  "/usr/bin/node"
)

for path_pattern in "${NODE_PATHS[@]}"; do
  # If pattern has wildcards, expand them
  if [[ "$path_pattern" == *"*"* ]]; then
    for node_path in $path_pattern; do
      if [ -x "$node_path" ]; then
        echo "Found Node.js at $node_path"
        sed -n '/^\/\/ --- BEGIN JAVASCRIPT ---$/,$ p' "$0" | tail -n +2 | "$node_path" -
        exit $?
      fi
    done
  elif [ -x "$path_pattern" ]; then
    echo "Found Node.js at $path_pattern"
    sed -n '/^\/\/ --- BEGIN JAVASCRIPT ---$/,$ p' "$0" | tail -n +2 | "$path_pattern" -
    exit $?
  fi
done

echo "Node.js not found. Cannot run server."
exit 1

// --- BEGIN JAVASCRIPT ---
/**
 * BCBS Direct Server - Minimal Node.js HTTP Server
 * This server is embedded in a bash script for flexibility
 */

// Create a basic HTTP server with just the built-in modules
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Configuration
const PORT = process.env.PORT || 5000;
const HOST = '0.0.0.0';

// Map of MIME types
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain',
};

// Function to get MIME type based on file extension
function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return MIME_TYPES[ext] || 'application/octet-stream';
}

// Function to generate diagnostics HTML
function generateDiagnosticsHtml() {
  const fileList = [];
  try {
    const files = fs.readdirSync('.');
    files.forEach(file => {
      const stats = fs.statSync(file);
      fileList.push({
        name: file,
        size: stats.size,
        isDirectory: stats.isDirectory(),
        modified: stats.mtime
      });
    });
  } catch (err) {
    console.error('Error reading directory:', err);
  }

  return `
<!DOCTYPE html>
<html>
<head>
  <title>BCBS Diagnostic - Direct Server</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
    h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
    h2 { color: #0066cc; margin-top: 30px; }
    pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
    .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
    .warning { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; }
    table th, table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
    table th { background-color: #f2f2f2; }
    .directory { color: #0066cc; font-weight: bold; }
  </style>
</head>
<body>
  <h1>BCBS Values - Direct Server Diagnostic</h1>
  
  <div class="card">
    <p class="warning"><strong>Note:</strong> You are viewing the direct fallback server. This indicates that primary servers failed to start.</p>
    <p>This page is served by a minimal Node.js HTTP server.</p>
    <p>Server started at: ${new Date().toLocaleString()}</p>
  </div>
  
  <div class="card">
    <h2>System Information</h2>
    <pre>
Hostname: ${os.hostname()}
Platform: ${os.platform()}
Architecture: ${os.arch()}
OS Version: ${os.version ? os.version() : 'unknown'}
Memory Total: ${Math.round(os.totalmem() / (1024 * 1024))} MB
Memory Free: ${Math.round(os.freemem() / (1024 * 1024))} MB
Uptime: ${Math.floor(os.uptime() / 3600)} hours, ${Math.floor((os.uptime() % 3600) / 60)} minutes
    </pre>
  </div>
  
  <div class="card">
    <h2>Node.js Information</h2>
    <pre>
Node.js Version: ${process.version}
V8 Version: ${process.versions.v8}
Working Directory: ${process.cwd()}
    </pre>
  </div>
  
  <div class="card">
    <h2>Environment Variables</h2>
    <pre>
PORT=${process.env.PORT || 'Not set'}
PATH=${process.env.PATH || 'Not set'}
PWD=${process.env.PWD || 'Not set'}
    </pre>
  </div>
  
  <div class="card">
    <h2>Directory Contents</h2>
    <table>
      <tr>
        <th>Name</th>
        <th>Type</th>
        <th>Size</th>
        <th>Modified</th>
      </tr>
      ${fileList.map(file => `
      <tr>
        <td${file.isDirectory ? ' class="directory"' : ''}>${file.name}</td>
        <td>${file.isDirectory ? 'Directory' : 'File'}</td>
        <td>${file.isDirectory ? '-' : file.size + ' bytes'}</td>
        <td>${file.modified.toLocaleString()}</td>
      </tr>`).join('')}
    </table>
  </div>
  
  <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
    <p>BCBS Values Direct Server Diagnostic</p>
    <p>Generated at: ${new Date().toLocaleString()}</p>
  </footer>
</body>
</html>
  `;
}

// Create the HTTP server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Parse the URL
  const url = new URL(req.url, `http://${req.headers.host}`);
  let filePath = '.' + url.pathname;
  
  // Default to index.html if the path ends with /
  if (filePath.endsWith('/')) {
    filePath += 'index.html';
  }
  
  // If the request is for the root, serve our diagnostic page
  if (filePath === './index.html' && !fs.existsSync(filePath)) {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(generateDiagnosticsHtml());
    return;
  }
  
  // Check if the file exists
  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      // File not found, serve the diagnostic page
      res.writeHead(404, { 'Content-Type': 'text/html' });
      res.end(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>404 - File Not Found</title>
          <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
            h1 { color: #d9534f; }
            .back { margin-top: 20px; }
          </style>
        </head>
        <body>
          <h1>404 - File Not Found</h1>
          <p>The requested file "${filePath}" was not found on the server.</p>
          <div class="back">
            <a href="/">Back to Diagnostics</a>
          </div>
        </body>
        </html>
      `);
      return;
    }
    
    // Read and serve the file
    fs.readFile(filePath, (err, content) => {
      if (err) {
        res.writeHead(500, { 'Content-Type': 'text/html' });
        res.end(`
          <!DOCTYPE html>
          <html>
          <head>
            <title>500 - Server Error</title>
            <style>
              body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
              h1 { color: #d9534f; }
              .back { margin-top: 20px; }
              pre { background: #f5f5f5; padding: 15px; }
            </style>
          </head>
          <body>
            <h1>500 - Server Error</h1>
            <p>An error occurred while reading the file:</p>
            <pre>${err.message}</pre>
            <div class="back">
              <a href="/">Back to Diagnostics</a>
            </div>
          </body>
          </html>
        `);
        return;
      }
      
      // Get the content type based on file extension
      const contentType = getMimeType(filePath);
      
      // Send the response
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    });
  });
});

// Start the server
server.listen(PORT, HOST, () => {
  console.log(`BCBS Direct Server running at http://${HOST}:${PORT}/`);
  console.log(`Serving files from: ${process.cwd()}`);
  console.log(`Server started at: ${new Date().toLocaleString()}`);
});

// Handle server errors
server.on('error', (err) => {
  console.error('Server error:', err);
  
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Trying to use another port.`);
    setTimeout(() => {
      server.close();
      server.listen(0, HOST);
    }, 1000);
  }
});

// Log when the server closes
process.on('SIGINT', () => {
  console.log('Server shutting down...');
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });
});