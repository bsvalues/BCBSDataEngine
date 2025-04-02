const http = require('http');
const fs = require('fs');
const path = require('path');

// Define common MIME types
const mimeTypes = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

// Create HTTP server
const server = http.createServer((req, res) => {
    console.log(`Request: ${req.method} ${req.url}`);
    
    // Handle root URL
    let filePath = req.url === '/' ? './dashboard_static.html' : '.' + req.url;
    
    // Handle /interactive-dashboard route
    if (req.url === '/interactive-dashboard') {
        filePath = './templates/reactive_dashboard.html';
    }
    
    // Handle /dashboard route
    if (req.url === '/dashboard') {
        filePath = './dashboard_static.html';
    }
    
    // Handle /demo route
    if (req.url === '/demo') {
        filePath = './dashboard_demo.html';
    }

    // Get file extension
    const extname = path.extname(filePath);
    const contentType = mimeTypes[extname] || 'application/octet-stream';

    // Read the file
    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                // If file not found, try to serve 404.html
                fs.readFile('./404.html', (err, content) => {
                    if (err) {
                        // If 404.html doesn't exist, send basic 404 response
                        res.writeHead(404);
                        res.end('404 Not Found');
                    } else {
                        res.writeHead(404, { 'Content-Type': 'text/html' });
                        res.end(content, 'utf-8');
                    }
                });
            } else {
                // For other errors, send 500 response
                res.writeHead(500);
                res.end(`Server Error: ${err.code}`);
            }
        } else {
            // File found, serve it
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

// Set server port and start listening
const PORT = process.env.PORT || 5002;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}/`);
});