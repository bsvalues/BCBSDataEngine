const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5002;
const HOST = '0.0.0.0';

// Track server statistics
const stats = {
    startTime: new Date(),
    requests: 0,
    errors: 0
};

// Simple MIME type lookup
function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon'
    };
    
    return mimeTypes[ext] || 'text/plain';
}

// Log with timestamp
function log(message) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${message}`);
}

// Create the server
const server = http.createServer((req, res) => {
    stats.requests++;
    
    // Log the request
    log(`${req.method} ${req.url}`);
    
    // Handle API requests
    if (req.url.startsWith('/api/')) {
        handleApiRequest(req, res);
        return;
    }
    
    // Handle static file requests
    let filePath = '.' + req.url;
    if (filePath === './') {
        filePath = './index.html';
    }
    
    // Check if file exists
    fs.access(filePath, fs.constants.F_OK, (err) => {
        if (err) {
            // File not found
            stats.errors++;
            res.writeHead(404, { 'Content-Type': 'text/html' });
            res.end(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>404 Not Found</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        h1 { color: #721c24; }
                        .card { border: 1px solid #f5c6cb; border-radius: 8px; padding: 20px; background-color: #f8d7da; color: #721c24; }
                    </style>
                </head>
                <body>
                    <h1>404 Not Found</h1>
                    <div class="card">
                        <p>The requested URL ${req.url} was not found on this server.</p>
                        <p><a href="/">Return to Homepage</a></p>
                    </div>
                </body>
                </html>
            `);
            return;
        }
        
        // Read and serve the file
        fs.readFile(filePath, (err, content) => {
            if (err) {
                stats.errors++;
                res.writeHead(500, { 'Content-Type': 'text/html' });
                res.end(`
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Server Error</title>
                        <style>
                            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                            h1 { color: #721c24; }
                            .card { border: 1px solid #f5c6cb; border-radius: 8px; padding: 20px; background-color: #f8d7da; color: #721c24; }
                        </style>
                    </head>
                    <body>
                        <h1>500 Server Error</h1>
                        <div class="card">
                            <p>Sorry, there was a problem reading the requested file.</p>
                            <p><a href="/">Return to Homepage</a></p>
                        </div>
                    </body>
                    </html>
                `);
                return;
            }
            
            // Serve the file
            res.writeHead(200, { 'Content-Type': getMimeType(filePath) });
            res.end(content);
        });
    });
});

// Handle API requests
function handleApiRequest(req, res) {
    const urlParts = req.url.split('/');
    const endpoint = urlParts[2] || '';
    
    res.setHeader('Content-Type', 'application/json');
    
    // Status endpoint
    if (endpoint === 'status') {
        res.writeHead(200);
        res.end(JSON.stringify({
            status: 'online',
            uptime: Math.floor((new Date() - stats.startTime) / 1000),
            requests: stats.requests,
            errors: stats.errors,
            timestamp: new Date().toISOString()
        }));
        return;
    }
    
    // Agent status endpoint
    if (endpoint === 'agent-status') {
        res.writeHead(200);
        res.end(JSON.stringify({
            status: 'success',
            message: 'Agent status retrieved successfully',
            data: {
                agents: [
                    {
                        id: 'agent-001',
                        name: 'ETL-Controller',
                        status: 'active',
                        last_heartbeat: new Date().toISOString(),
                        queue_size: 12,
                        success_rate: 0.97
                    },
                    {
                        id: 'agent-002',
                        name: 'Model-Executor',
                        status: 'active',
                        last_heartbeat: new Date().toISOString(),
                        queue_size: 5,
                        success_rate: 0.99
                    },
                    {
                        id: 'agent-003',
                        name: 'API-Gateway',
                        status: 'active',
                        last_heartbeat: new Date().toISOString(),
                        queue_size: 0,
                        success_rate: 1.0
                    }
                ],
                timestamp: new Date().toISOString()
            }
        }));
        return;
    }
    
    // Default response for other endpoints
    res.writeHead(200);
    res.end(JSON.stringify({
        status: 'success',
        message: `API endpoint: ${endpoint}`,
        path: req.url,
        timestamp: new Date().toISOString()
    }));
}

// Start the server
server.listen(PORT, HOST, () => {
    log(`BCBS Values Platform Server running at http://${HOST}:${PORT}/`);
    
    // List HTML files in the directory
    fs.readdir('.', (err, files) => {
        if (err) {
            log(`Error reading directory: ${err.message}`);
            return;
        }
        
        const htmlFiles = files.filter(file => file.endsWith('.html'));
        if (htmlFiles.length > 0) {
            log(`Available HTML files: ${htmlFiles.join(', ')}`);
        } else {
            log('No HTML files found in the directory');
        }
    });
});
