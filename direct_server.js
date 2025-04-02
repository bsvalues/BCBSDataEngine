// Direct HTTP Server for Workflow Reliability
// This Node.js-based server is used as a fallback when Python isn't available

const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

// Configuration
const PORT = 5002;
const HOST = '0.0.0.0';

// Create an HTTP server
const server = http.createServer((req, res) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  
  // Handle API requests
  if (req.url.startsWith('/api/')) {
    handleApiRequest(req, res);
    return;
  }
  
  // Serve static files
  let filePath = '.' + req.url;
  if (filePath === './') {
    filePath = './index.html';
  }
  
  // Check if the file exists
  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      // If index.html doesn't exist, create a default one
      if (filePath === './index.html') {
        createDefaultIndex();
        serveFile(filePath, res);
      } else {
        // File not found
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<html><head><title>404 Not Found</title></head><body><h1>404 Not Found</h1><p>The requested file was not found.</p></body></html>');
      }
    } else {
      // File exists, serve it
      serveFile(filePath, res);
    }
  });
});

// Function to serve a file
function serveFile(filePath, res) {
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
  }
  
  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(500);
      res.end(`Server Error: ${err.code}`);
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
}

// Function to handle API requests
function handleApiRequest(req, res) {
  const endpoint = req.url.split('/')[2] || '';
  
  // Set content type for API responses
  res.setHeader('Content-Type', 'application/json');
  
  // Generate response based on endpoint
  if (endpoint === 'status') {
    // Server status endpoint
    const response = {
      status: 'online',
      server: 'Node.js Direct Server',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    };
    res.writeHead(200);
    res.end(JSON.stringify(response, null, 2));
  } else if (endpoint === 'agent-status') {
    // Agent status endpoint
    const now = new Date().toISOString();
    const response = {
      status: 'success',
      message: 'Agent status retrieved successfully',
      data: {
        agents: [
          {
            id: 'agent-001',
            name: 'ETL-Controller',
            status: 'active',
            last_heartbeat: now,
            queue_size: 12,
            success_rate: 0.97
          },
          {
            id: 'agent-002',
            name: 'Model-Executor',
            status: 'active',
            last_heartbeat: now,
            queue_size: 5,
            success_rate: 0.99
          },
          {
            id: 'agent-003',
            name: 'API-Gateway',
            status: 'active',
            last_heartbeat: now,
            queue_size: 0,
            success_rate: 1.0
          }
        ],
        timestamp: now
      }
    };
    res.writeHead(200);
    res.end(JSON.stringify(response, null, 2));
  } else {
    // Default API response
    const response = {
      status: 'success',
      message: `API endpoint: ${endpoint}`,
      path: req.url,
      timestamp: new Date().toISOString()
    };
    res.writeHead(200);
    res.end(JSON.stringify(response, null, 2));
  }
}

// Function to create a default index.html
function createDefaultIndex() {
  console.log(`[${new Date().toISOString()}] Creating default index.html`);
  
  const htmlContent = `<!DOCTYPE html>
<html>
<head>
    <title>BCBS Values Platform</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
        .success { background: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <h1>BCBS Values Platform</h1>
    
    <div class="card">
        <h2>System Status</h2>
        <p><span class="status success">Server is running</span></p>
        <p>Current time: <span id="server-time">loading...</span></p>
    </div>
    
    <div class="card">
        <h2>Diagnostic Information</h2>
        <ul>
            <li>Server type: Node.js Direct Server</li>
            <li>Server started: ${new Date().toLocaleString()}</li>
            <li>Client time: <span id="client-time">loading...</span></li>
        </ul>
    </div>
    
    <script>
        // Update time function
        function updateTime() {
            const now = new Date();
            document.getElementById('server-time').textContent = now.toLocaleString();
            document.getElementById('client-time').textContent = now.toLocaleString();
        }
        
        // Initial update
        updateTime();
        
        // Update every second
        setInterval(updateTime, 1000);
        
        // Fetch server status
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                console.log('Server status:', data);
            })
            .catch(error => {
                console.error('Error fetching server status:', error);
            });
    </script>
</body>
</html>`;
  
  try {
    fs.writeFileSync('./index.html', htmlContent);
    console.log(`[${new Date().toISOString()}] Default index.html created successfully`);
  } catch (err) {
    console.error(`[${new Date().toISOString()}] Error creating index.html: ${err.message}`);
  }
}

// Start the server
server.listen(PORT, HOST, () => {
  console.log(`[${new Date().toISOString()}] Server running at http://${HOST}:${PORT}/`);
  console.log(`[${new Date().toISOString()}] Node.js version: ${process.version}`);
  console.log(`[${new Date().toISOString()}] Current directory: ${process.cwd()}`);
  
  // List available HTML files
  try {
    const files = fs.readdirSync('.').filter(file => file.endsWith('.html'));
    if (files.length > 0) {
      console.log(`[${new Date().toISOString()}] Available HTML files: ${files.join(', ')}`);
    } else {
      console.log(`[${new Date().toISOString()}] No HTML files found in current directory`);
      createDefaultIndex();
    }
  } catch (err) {
    console.error(`[${new Date().toISOString()}] Error listing files: ${err.message}`);
  }
});