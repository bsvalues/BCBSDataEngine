// Emergency Minimal HTTP Server for BCBS Values Platform
// This simple server uses only standard Node.js modules

const http = require('http');
const fs = require('fs');

// Configuration
const PORT = 5002;

// Create simple HTML content
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
        <h2>Emergency Server</h2>
        <p>This is a minimal emergency server for the BCBS Values Platform.</p>
        <p id="client-time"></p>
    </div>
    
    <script>
        function updateTime() {
            const now = new Date();
            document.getElementById('server-time').textContent = now.toLocaleString();
            document.getElementById('client-time').textContent = 'Client time: ' + now.toLocaleString();
        }
        updateTime();
        setInterval(updateTime, 1000);
    </script>
</body>
</html>`;

// Create index.html for direct file serving
try {
    fs.writeFileSync('index.html', htmlContent);
    console.log('Created index.html');
} catch (err) {
    console.error('Error creating index.html:', err);
}

// Create HTTP server
const server = http.createServer((req, res) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    
    // Simple routing
    if (req.url === '/' || req.url === '/index.html') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(htmlContent);
    } else if (req.url.startsWith('/api/')) {
        // API endpoint
        const response = {
            status: 'success',
            server: 'Emergency Server',
            timestamp: new Date().toISOString(),
            url: req.url
        };
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(response));
    } else {
        // 404 for anything else
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<html><body><h1>404 Not Found</h1></body></html>');
    }
});

// Start the server
server.listen(PORT, () => {
    console.log(`${new Date().toISOString()} - Emergency server running at http://0.0.0.0:${PORT}/`);
});