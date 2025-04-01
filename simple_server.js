/**
 * Ultra-minimal diagnostic server for BCBS Values
 */

// Create a simple HTTP server
const http = require('http');
const os = require('os');
const fs = require('fs');

const PORT = process.env.PORT || 5000;

const server = http.createServer((req, res) => {
  // Basic HTML diagnostic page
  res.writeHead(200, { 'Content-Type': 'text/html' });
  
  // Generate HTML content
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>BCBS Diagnostic</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #0066cc; margin-top: 30px; }
        pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
      </style>
    </head>
    <body>
      <h1>BCBS Values Diagnostic Server</h1>
      
      <div class="card">
        <h2>System Information</h2>
        <pre>
Time: ${new Date().toISOString()}
Hostname: ${os.hostname()}
Platform: ${os.platform()} ${os.release()}
Memory: ${Math.round(os.totalmem() / (1024 * 1024))} MB total, ${Math.round(os.freemem() / (1024 * 1024))} MB free
Node.js: ${process.version}
        </pre>
      </div>
      
      <div class="card">
        <h2>File System Access</h2>
        <pre>
Current Directory: ${process.cwd()}
Files: ${fs.readdirSync('.').join(', ')}
        </pre>
      </div>
      
      <div class="card">
        <h2>Environment Variables</h2>
        <pre>
PORT: ${process.env.PORT || 'Not set'}
NODE_ENV: ${process.env.NODE_ENV || 'Not set'}
        </pre>
      </div>
      
      <div class="card">
        <h2>Next Steps</h2>
        <p>This is a minimal diagnostic server to confirm that Node.js is working in this environment.</p>
        <p>For a more complete diagnostic, the full application needs to be properly installed.</p>
      </div>
      
      <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
        <p>BCBS Values Minimal Diagnostic Server</p>
        <p>Generated: ${new Date().toISOString()}</p>
      </footer>
    </body>
    </html>
  `;
  
  // Send response
  res.end(html);
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Minimal Diagnostic Server running at http://0.0.0.0:${PORT}/`);
});